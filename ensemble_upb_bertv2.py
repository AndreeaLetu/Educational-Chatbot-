import os, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from scipy import stats
from sklearn.metrics import (
    accuracy_score, f1_score,
    classification_report, confusion_matrix
)

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
ROBERT_PT      = os.path.join(BASE_DIR, "models", "robert", "robert_model.pt")
ROBERT_TOK     = os.path.join(BASE_DIR, "models", "robert", "tokenizer")
BERT_V2_DIR    = os.path.join(BASE_DIR, "models", "final", "tinylogic_bert_romanian_v2")
DATA_DIR       = os.path.join(BASE_DIR, "data", "augmented")
OUT_DIR        = os.path.join(BASE_DIR, "diagrams", "ensemble_robert_bert_v2")
os.makedirs(OUT_DIR, exist_ok=True)

MAX_LENGTH       = 128
BATCH_SIZE       = 16
NUM_DIFFICULTIES = 3
CMAP = LinearSegmentedColormap.from_list(
    "purple", ["#ffffff", "#e8d5f5", "#9c4dcc", "#4a148c"], N=256
)
ID_TO_DIFF = {0: "beginner", 1: "intermediate", 2: "advanced"}
DIFF_NAMES = [ID_TO_DIFF[i] for i in range(NUM_DIFFICULTIES)]


class TinyLogicDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.texts      = df["text_clean"].tolist()
        self.labels     = df["label_id"].tolist()
        self.difficulty = df["difficulty"].tolist()
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx], max_length=self.max_length,
            padding="max_length", truncation=True, return_tensors="pt"
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":      torch.tensor(self.labels[idx],     dtype=torch.long),
            "difficulty": torch.tensor(self.difficulty[idx], dtype=torch.long),
        }


class MultiTaskRoBERT(nn.Module):
    def __init__(self, model_name, num_labels, num_difficulties=3, dropout=0.3):
        super().__init__()
        self.robert  = AutoModel.from_pretrained(model_name)
        h            = self.robert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.head_label = nn.Sequential(
            nn.Linear(h, 256), nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, num_labels)
        )
        self.head_difficulty = nn.Sequential(
            nn.Linear(h, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, num_difficulties)
        )

    def forward(self, input_ids, attention_mask):
        out = self.robert(input_ids=input_ids, attention_mask=attention_mask)
        cls = self.dropout(out.last_hidden_state[:, 0, :])
        return self.head_label(cls), self.head_difficulty(cls)


class MultiTaskBERT_v2(nn.Module):
    def __init__(self, model_name, num_labels, num_difficulties, freeze_layers=4):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        for i, layer in enumerate(self.bert.encoder.layer):
            if i < freeze_layers:
                for p in layer.parameters(): p.requires_grad = False
        h = self.bert.config.hidden_size
        self.label_head = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(h, 256), nn.ReLU(),
            nn.Dropout(0.1), nn.Linear(256, num_labels)
        )
        self.difficulty_head = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(h, 256), nn.ReLU(),
            nn.Dropout(0.1), nn.Linear(256, num_difficulties)
        )

    def mean_pooling(self, outputs, attention_mask):
        t = outputs.last_hidden_state
        m = attention_mask.unsqueeze(-1).float()
        return torch.sum(t * m, 1) / torch.clamp(m.sum(1), min=1e-9)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        p   = self.mean_pooling(out, attention_mask)
        return self.label_head(p), self.difficulty_head(p)


def get_probs(model, loader, device):
    model.eval()
    softmax = nn.Softmax(dim=1)
    lp, dp, yt_l, yt_d = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            ll, dl = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device)
            )
            lp.append(softmax(ll).cpu().numpy())
            dp.append(softmax(dl).cpu().numpy())
            yt_l.extend(batch["label"].tolist())
            yt_d.extend(batch["difficulty"].tolist())
    return {
        "label_probs": np.vstack(lp),
        "diff_probs":  np.vstack(dp),
        "label_true":  np.array(yt_l),
        "diff_true":   np.array(yt_d),
    }


def compute_metrics(yp_l, yp_d, yt_l, yt_d):
    return {
        "label_acc":   accuracy_score(yt_l, yp_l),
        "label_f1":    f1_score(yt_l, yp_l, average="macro", zero_division=0),
        "diff_acc":    accuracy_score(yt_d, yp_d),
        "diff_f1":     f1_score(yt_d, yp_d, average="macro", zero_division=0),
        "overall_acc": (accuracy_score(yt_l, yp_l) + accuracy_score(yt_d, yp_d)) / 2,
    }


def majority_voting(preds_l, preds_d):
    sl = np.stack(preds_l, axis=1)
    sd = np.stack(preds_d, axis=1)
    return (stats.mode(sl, axis=1, keepdims=False).mode,
            stats.mode(sd, axis=1, keepdims=False).mode)

def average_probs(lp_list, dp_list):
    return (np.argmax(np.mean(lp_list, axis=0), axis=1),
            np.argmax(np.mean(dp_list,  axis=0), axis=1))

def weighted_probs(lp_list, dp_list, weights):
    w = np.array(weights) / np.sum(weights)
    wl = sum(wi * lp for wi, lp in zip(w, lp_list))
    wd = sum(wi * dp for wi, dp in zip(w, dp_list))
    return np.argmax(wl, axis=1), np.argmax(wd, axis=1)


def plot_cm(cm, classes, title, path):
    n = len(classes)
    fig, ax = plt.subplots(figsize=(max(6, n*1.5), max(5, n*1.3)), facecolor="white")
    ax.set_facecolor("white")
    im = ax.imshow(cm, cmap=CMAP, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=8)
    ax.set_xticks(range(n)); ax.set_xticklabels(classes, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(n)); ax.set_yticklabels(classes, fontsize=9)
    ax.set_xlabel("Etichetă prezisă", fontsize=10)
    ax.set_ylabel("Etichetă reală",   fontsize=10)
    ax.set_title(title, fontweight="bold", pad=12, fontsize=11)
    vmax = cm.max()
    for i in range(n):
        for j in range(n):
            v = cm[i, j]
            color = "white" if v >= vmax * 0.55 else "#333333"
            ax.text(j, i, str(v), ha="center", va="center",
                    fontsize=11, color=color, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path + ".pdf", bbox_inches="tight", dpi=200, facecolor="white")
    plt.savefig(path + ".png", bbox_inches="tight", dpi=200, facecolor="white")
    plt.close()
    print(f"  ✓ {os.path.basename(path)}")


def main():
    print("\n" + "="*60)
    print("  Ensemble — RoBERT UPB + BERT Românesc v2")
    print("="*60)

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    print(f"  Test: {len(test_df)} exemple | Device: {device}\n")

    print("  → Încărcare RoBERT UPB...")
    ckpt       = torch.load(ROBERT_PT, map_location=device)
    label_map  = ckpt["label_map"]
    if isinstance(list(label_map.keys())[0], str):
        label_map = {int(k): v for k, v in label_map.items()}
    NUM_LABELS  = len(label_map)
    LABEL_NAMES = [label_map[i] for i in range(NUM_LABELS)]

    tok_robert  = AutoTokenizer.from_pretrained(ROBERT_TOK)
    loader_rob  = DataLoader(TinyLogicDataset(test_df, tok_robert, MAX_LENGTH),
                             batch_size=BATCH_SIZE, shuffle=False)
    model_rob   = MultiTaskRoBERT("readerbench/RoBERT-base", NUM_LABELS)
    model_rob.load_state_dict(ckpt["model_state_dict"])
    model_rob.to(device).eval()
    print("    ✓ RoBERT UPB încărcat")

    print("  → Încărcare BERT Românesc v2...")
    with open(os.path.join(BERT_V2_DIR, "config.json"), encoding="utf-8") as f:
        cfg_v2 = json.load(f)
    tok_bert    = AutoTokenizer.from_pretrained(BERT_V2_DIR)
    loader_bert = DataLoader(TinyLogicDataset(test_df, tok_bert, MAX_LENGTH),
                             batch_size=BATCH_SIZE, shuffle=False)
    model_bert  = MultiTaskBERT_v2(
        cfg_v2["model_name"], NUM_LABELS, NUM_DIFFICULTIES,
        cfg_v2.get("freeze_layers", 4)
    )
    model_bert.load_state_dict(torch.load(
        os.path.join(BERT_V2_DIR, "model_weights.pt"), map_location=device
    ))
    model_bert.to(device).eval()
    print("    ✓ BERT Românesc v2 încărcat\n")

    print("  Generez predicții...")
    preds_rob  = get_probs(model_rob,  loader_rob,  device)
    preds_bert = get_probs(model_bert, loader_bert, device)

    yt_l = preds_rob["label_true"]
    yt_d = preds_rob["diff_true"]

    lp_list = [preds_rob["label_probs"], preds_bert["label_probs"]]
    dp_list = [preds_rob["diff_probs"],  preds_bert["diff_probs"]]

    ind_rob  = compute_metrics(
        np.argmax(preds_rob["label_probs"],  axis=1),
        np.argmax(preds_rob["diff_probs"],   axis=1), yt_l, yt_d
    )
    ind_bert = compute_metrics(
        np.argmax(preds_bert["label_probs"], axis=1),
        np.argmax(preds_bert["diff_probs"],  axis=1), yt_l, yt_d
    )

    print("  Aplic metodele ensemble...\n")
    weights = [ind_rob["overall_acc"], ind_bert["overall_acc"]]

    mv_l,  mv_d  = majority_voting(
        [np.argmax(preds_rob["label_probs"], 1), np.argmax(preds_bert["label_probs"], 1)],
        [np.argmax(preds_rob["diff_probs"],  1), np.argmax(preds_bert["diff_probs"],  1)]
    )
    avg_l, avg_d = average_probs(lp_list, dp_list)
    w_l,   w_d   = weighted_probs(lp_list, dp_list, weights)

    ensemble_results = {
        "Majority Voting":       compute_metrics(mv_l,  mv_d,  yt_l, yt_d),
        "Average Probabilities": compute_metrics(avg_l, avg_d, yt_l, yt_d),
        "Weighted Ensemble":     compute_metrics(w_l,   w_d,   yt_l, yt_d),
    }

    print(f"  {'Model':<28} {'LblAcc':>8} {'LblF1':>8} {'DiffAcc':>8} {'DiffF1':>8} {'Overall':>8}")
    print(f"  {'─'*68}")
    for name, m in [("RoBERT UPB", ind_rob), ("BERT Românesc v2", ind_bert)]:
        print(f"  {name:<28} {m['label_acc']:>8.4f} {m['label_f1']:>8.4f} "
              f"{m['diff_acc']:>8.4f} {m['diff_f1']:>8.4f} {m['overall_acc']:>8.4f}")
    print(f"  {'─'*68}")
    for name, m in ensemble_results.items():
        print(f"  {name:<28} {m['label_acc']:>8.4f} {m['label_f1']:>8.4f} "
              f"{m['diff_acc']:>8.4f} {m['diff_f1']:>8.4f} {m['overall_acc']:>8.4f}")

    best_name = max(ensemble_results, key=lambda x: ensemble_results[x]["diff_acc"])
    best_m    = ensemble_results[best_name]
    best_preds_l, best_preds_d = {
        "Majority Voting":       (mv_l,  mv_d),
        "Average Probabilities": (avg_l, avg_d),
        "Weighted Ensemble":     (w_l,   w_d),
    }[best_name]

    print(f"\n  ✅ Cel mai bun ensemble pe Diff: {best_name}")
    print(f"     Diff Acc: {best_m['diff_acc']:.4f} | Overall: {best_m['overall_acc']:.4f}\n")

    print("── Classification Report — Label ──────────────────────")
    print(classification_report(yt_l, best_preds_l, target_names=LABEL_NAMES))
    print("── Classification Report — Dificultate ────────────────")
    print(classification_report(yt_d, best_preds_d, target_names=DIFF_NAMES))

    plot_cm(
        confusion_matrix(yt_l, best_preds_l), LABEL_NAMES,
        f"Ensemble (RoBERT + BERT v2) — Label\n"
        f"{best_name}  |  Acc={best_m['label_acc']:.2%}  F1={best_m['label_f1']:.2%}",
        os.path.join(OUT_DIR, "cm_label")
    )
    plot_cm(
        confusion_matrix(yt_d, best_preds_d), DIFF_NAMES,
        f"Ensemble (RoBERT + BERT v2) — Dificultate\n"
        f"{best_name}  |  Acc={best_m['diff_acc']:.2%}  F1={best_m['diff_f1']:.2%}",
        os.path.join(OUT_DIR, "cm_diff")
    )

    report = {
        "individual": {"RoBERT UPB": ind_rob, "BERT Românesc v2": ind_bert},
        "ensemble":   ensemble_results,
        "best":       best_name,
    }
    with open(os.path.join(OUT_DIR, "ensemble_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n  ✅ Gata! Fișiere în: {OUT_DIR}")


if __name__ == "__main__":
    main()