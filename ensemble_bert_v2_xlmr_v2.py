import os, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import (
    accuracy_score, f1_score,
    classification_report, confusion_matrix
)

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
BERT_V2_DIR  = os.path.join(BASE_DIR, "models", "final", "tinylogic_bert_romanian_v2")
XLMR_V2_DIR  = os.path.join(BASE_DIR, "models", "final", "tinylogic_xlmroberta_v2")
DATA_DIR     = os.path.join(BASE_DIR, "data", "augmented")
OUT_DIR      = os.path.join(BASE_DIR, "diagrams", "ensemble_bert_v2_xlmr_v2")
os.makedirs(OUT_DIR, exist_ok=True)

MAX_LENGTH       = 128
BATCH_SIZE       = 16
NUM_DIFFICULTIES = 3

CMAP = LinearSegmentedColormap.from_list(
    "purple", ["#ffffff", "#e8d5f5", "#9c4dcc", "#4a148c"], N=256
)
ID_TO_DIFF = {0: "beginner", 1: "intermediate", 2: "advanced"}
DIFF_NAMES = [ID_TO_DIFF[i] for i in range(NUM_DIFFICULTIES)]

with open(os.path.join(BERT_V2_DIR, "config.json"), encoding="utf-8") as f:
    cfg_bert = json.load(f)

NUM_LABELS  = cfg_bert["num_labels"]
ID_TO_LABEL = {int(k): v for k, v in cfg_bert["label_map"].items()}
LABEL_NAMES = [ID_TO_LABEL[i] for i in range(NUM_LABELS)]


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


class MultiTaskMeanPool(nn.Module):
    def __init__(self, model_name, num_labels, num_difficulties, freeze_layers=4):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        try:
            for i, layer in enumerate(self.bert.encoder.layer):
                if i < freeze_layers:
                    for p in layer.parameters(): p.requires_grad = False
        except AttributeError:
            pass
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


def plot_cm(cm, classes, title, path):
    n = len(classes)
    fig, ax = plt.subplots(
        figsize=(max(6, n*1.5), max(5, n*1.3)), facecolor="white"
    )
    ax.set_facecolor("white")
    im = ax.imshow(cm, cmap=CMAP, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=8)
    ax.set_xticks(range(n))
    ax.set_xticklabels(classes, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(n))
    ax.set_yticklabels(classes, fontsize=9)
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
    print("  Ensemble — BERT Românesc v2 + XLM-RoBERTa v2")
    print("  Metodă: Average Probabilities")
    print("="*60)

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    print(f"  Test: {len(test_df)} exemple | Device: {device}\n")

    print("  → Încărcare BERT Românesc v2...")
    tok_bert    = AutoTokenizer.from_pretrained(BERT_V2_DIR)
    loader_bert = DataLoader(
        TinyLogicDataset(test_df, tok_bert, MAX_LENGTH),
        batch_size=BATCH_SIZE, shuffle=False
    )
    model_bert = MultiTaskMeanPool(
        cfg_bert["model_name"], NUM_LABELS, NUM_DIFFICULTIES,
        cfg_bert.get("freeze_layers", 4)
    )
    model_bert.load_state_dict(torch.load(
        os.path.join(BERT_V2_DIR, "model_weights.pt"), map_location=device
    ))
    model_bert.to(device)
    print("    ✓ BERT Românesc v2 încărcat")

    print("  → Încărcare XLM-RoBERTa v2...")
    with open(os.path.join(XLMR_V2_DIR, "config.json"), encoding="utf-8") as f:
        cfg_xlmr = json.load(f)
    tok_xlmr    = AutoTokenizer.from_pretrained(XLMR_V2_DIR)
    loader_xlmr = DataLoader(
        TinyLogicDataset(test_df, tok_xlmr, MAX_LENGTH),
        batch_size=BATCH_SIZE, shuffle=False
    )
    model_xlmr = MultiTaskMeanPool(
        cfg_xlmr["model_name"], NUM_LABELS, NUM_DIFFICULTIES,
        cfg_xlmr.get("freeze_layers", 4)
    )
    model_xlmr.load_state_dict(torch.load(
        os.path.join(XLMR_V2_DIR, "model_weights.pt"), map_location=device
    ))
    model_xlmr.to(device)
    print("    ✓ XLM-RoBERTa v2 încărcat\n")

    print("  Generez predicții...")
    preds_bert = get_probs(model_bert, loader_bert, device)
    preds_xlmr = get_probs(model_xlmr, loader_xlmr, device)

    yt_l = preds_bert["label_true"]
    yt_d = preds_bert["diff_true"]

    avg_label_probs = (preds_bert["label_probs"] + preds_xlmr["label_probs"]) / 2
    avg_diff_probs  = (preds_bert["diff_probs"]  + preds_xlmr["diff_probs"])  / 2

    yp_l = np.argmax(avg_label_probs, axis=1)
    yp_d = np.argmax(avg_diff_probs,  axis=1)

    label_acc = accuracy_score(yt_l, yp_l)
    label_f1  = f1_score(yt_l, yp_l, average="macro", zero_division=0)
    diff_acc  = accuracy_score(yt_d, yp_d)
    diff_f1   = f1_score(yt_d, yp_d,  average="macro", zero_division=0)

    print(f"\n  Label — Acc: {label_acc:.4f} | F1: {label_f1:.4f}")
    print(f"  Diff  — Acc: {diff_acc:.4f}  | F1: {diff_f1:.4f}\n")

    print("── Classification Report — Label ──────────────────────")
    print(classification_report(yt_l, yp_l, target_names=LABEL_NAMES))
    print("── Classification Report — Dificultate ────────────────")
    print(classification_report(yt_d, yp_d, target_names=DIFF_NAMES))

    print("  Generez heatmap-uri...")
    plot_cm(
        confusion_matrix(yt_l, yp_l), LABEL_NAMES,
        f"Ensemble BERT v2 + XLM-R v2 — Label\n"
        f"Avg. Probabilities  |  Acc={label_acc:.2%}  F1={label_f1:.2%}",
        os.path.join(OUT_DIR, "cm_label")
    )
    plot_cm(
        confusion_matrix(yt_d, yp_d), DIFF_NAMES,
        f"Ensemble BERT v2 + XLM-R v2 — Dificultate\n"
        f"Avg. Probabilities  |  Acc={diff_acc:.2%}  F1={diff_f1:.2%}",
        os.path.join(OUT_DIR, "cm_diff")
    )

    print(f"\n  ✅ Gata! Fișiere în: {OUT_DIR}")


if __name__ == "__main__":
    main()