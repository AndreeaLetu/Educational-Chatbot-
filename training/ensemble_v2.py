

import os
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score, f1_score, classification_report
from scipy import stats


BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(BASE_DIR, "data", "augmented")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

MAX_LENGTH = 128
BATCH_SIZE = 16

MODELS_CONFIG = [
    {
        "name":       "BERT Românesc v2",
        "save_name":  "tinylogic_bert_romanian_v2",
        "model_name": "dumitrescustefan/bert-base-romanian-cased-v1",
        "pooling":    "mean",
    },
    {
        "name":       "XLM-RoBERTa v2",
        "save_name":  "tinylogic_xlmroberta_v2",
        "model_name": "xlm-roberta-base",
        "pooling":    "mean",
    },
]

ID_TO_DIFF  = {0: "beginner", 1: "intermediate", 2: "advanced"}

class TinyLogicDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.texts      = df["text_clean"].tolist()
        self.labels     = df["label_id"].tolist()
        self.difficulty = df["difficulty"].tolist()
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long),
            "difficulty":     torch.tensor(self.difficulty[idx], dtype=torch.long),
        }



class MultiTaskModel_MeanPool(nn.Module):

    def __init__(self, model_name, num_labels, num_difficulties):
        super().__init__()
        self.bert       = AutoModel.from_pretrained(model_name)
        hidden_size     = self.bert.config.hidden_size  # 768

        self.label_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_labels)
        )
        self.difficulty_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_difficulties)
        )

    def mean_pooling(self, outputs, attention_mask):
        token_embeddings    = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).float()
        sum_embeddings      = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask            = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(self, input_ids, attention_mask):
        outputs      = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled       = self.mean_pooling(outputs, attention_mask)
        label_logits = self.label_head(pooled)
        diff_logits  = self.difficulty_head(pooled)
        return label_logits, diff_logits


def load_model(cfg, num_labels, num_difficulties, device):
    final_path   = os.path.join(BASE_DIR, "models", "final", cfg["save_name"])
    weights_path = os.path.join(final_path, "model_weights.pt")
    config_path  = os.path.join(final_path, "config.json")

    if not os.path.exists(weights_path):
        print(f"  ❌ Nu găsesc: {weights_path}")
        print(f"     Rulează mai întâi antrenamentul pentru {cfg['name']}")
        return None, None, {}

    tokenizer = AutoTokenizer.from_pretrained(final_path)

    model = MultiTaskModel_MeanPool(cfg["model_name"], num_labels, num_difficulties)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    metrics = {}
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        metrics = {
            "label_acc":  config.get("test_label_acc", 0),
            "label_f1":   config.get("test_label_f1", 0),
            "diff_acc":   config.get("test_diff_acc", 0),
            "diff_f1":    config.get("test_diff_f1", 0),
            "overall_f1": (config.get("test_label_f1", 0) +
                           config.get("test_diff_f1", 0)) / 2,
            "epochs":     config.get("epochs", "?"),
            "pooling":    config.get("pooling", "mean"),
        }

    print(f"   {cfg['name']} încărcat")
    print(f"     Pooling: {cfg['pooling']} | "
          f"Label F1: {metrics.get('label_f1', 0):.4f} | "
          f"Diff F1: {metrics.get('diff_f1', 0):.4f} | "
          f"Epoci: {metrics.get('epochs', '?')}")

    return (model, tokenizer), metrics


def get_predictions(model, loader, device):
    model.eval()
    all_label_probs = []
    all_diff_probs  = []
    all_label_true  = []
    all_diff_true   = []
    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            label_logits, diff_logits = model(input_ids, attention_mask)

            all_label_probs.append(softmax(label_logits).cpu().numpy())
            all_diff_probs.append(softmax(diff_logits).cpu().numpy())
            all_label_true.extend(batch["label"].tolist())
            all_diff_true.extend(batch["difficulty"].tolist())

    return {
        "label_probs": np.vstack(all_label_probs),
        "diff_probs":  np.vstack(all_diff_probs),
        "label_true":  np.array(all_label_true),
        "diff_true":   np.array(all_diff_true),
    }


def compute_metrics(preds_label, preds_diff, true_label, true_diff):
    label_acc  = accuracy_score(true_label, preds_label)
    label_f1   = f1_score(true_label, preds_label, average="macro", zero_division=0)
    diff_acc   = accuracy_score(true_diff, preds_diff)
    diff_f1    = f1_score(true_diff, preds_diff, average="macro", zero_division=0)
    overall    = (label_acc + diff_acc) / 2
    overall_f1 = (label_f1 + diff_f1) / 2

    return {
        "label_acc": label_acc, "label_f1":  label_f1,
        "diff_acc":  diff_acc,  "diff_f1":   diff_f1,
        "overall_acc": overall, "overall_f1": overall_f1,
    }


def majority_voting(all_preds_label, all_preds_diff):
    stacked_label = np.stack(all_preds_label, axis=1)
    stacked_diff  = np.stack(all_preds_diff,  axis=1)
    final_label   = stats.mode(stacked_label, axis=1, keepdims=False).mode
    final_diff    = stats.mode(stacked_diff,  axis=1, keepdims=False).mode
    return final_label, final_diff


def average_probabilities(all_label_probs, all_diff_probs):
    avg_label = np.mean(all_label_probs, axis=0)
    avg_diff  = np.mean(all_diff_probs,  axis=0)
    return np.argmax(avg_label, axis=1), np.argmax(avg_diff, axis=1)


def weighted_ensemble(all_label_probs, all_diff_probs, weights, names):
    weights = np.array(weights)
    weights = weights / weights.sum()

    print(f"\n  Ponderi pentru Weighted Ensemble:")
    for name, w in zip(names, weights):
        print(f"    {name}: {w:.3f}")

    weighted_label = sum(w * lp for w, lp in zip(weights, all_label_probs))
    weighted_diff  = sum(w * dp for w, dp in zip(weights, all_diff_probs))

    return np.argmax(weighted_label, axis=1), np.argmax(weighted_diff, axis=1)

def main():
    print("\n" + "="*65)
    print("  TinyLogic — Ensemble BERT Românesc v2 + XLM-RoBERTa v2")
    print("  Ambele modele: Mean Pooling + Label Smoothing + Cosine")
    print("  Scop: maximizare acuratețe difficulty")
    print("="*65 + "\n")

    os.makedirs(REPORTS_DIR, exist_ok=True)

    device = torch.device("cpu")
    print(f"  Dispozitiv: {device}")

    print(f"\n📂 Încărcăm test set din: {DATA_DIR}")
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

    with open(os.path.join(DATA_DIR, "label_map.json")) as f:
        label_map = json.load(f)

    num_labels       = len(label_map)
    num_difficulties = 3
    id_to_label      = {v: k for k, v in label_map.items()}

    print(f"  Test: {len(test_df)} exemple")

    print(f"\nÎncărcăm modelele v2...")
    loaded_models      = []
    individual_results = {}
    weights            = []
    names              = []

    for cfg in MODELS_CONFIG:
        print(f"\n  → {cfg['name']}")
        result = load_model(cfg, num_labels, num_difficulties, device)

        if result[0] is None:
            continue

        (model, tokenizer), metrics = result
        loaded_models.append((model, tokenizer, cfg))
        individual_results[cfg["name"]] = metrics
        weights.append(metrics.get("overall_f1", 0.5))
        names.append(cfg["name"])

    if len(loaded_models) < 2:
        print("\n❌ Ai nevoie de ambele modele v2 antrenate!")
        print("   Verifică că există:")
        for cfg in MODELS_CONFIG:
            path = os.path.join(BASE_DIR, "models", "final",
                                cfg["save_name"], "model_weights.pt")
            exists = "✅" if os.path.exists(path) else "❌"
            print(f"   {exists} {path}")
        return

    print(f"\n  ✅ {len(loaded_models)} modele încărcate")

    # ── Generăm predicțiile ──
    print(f"\nGenerăm predicțiile pe test set...")

    all_label_probs = []
    all_diff_probs  = []
    all_label_preds = []
    all_diff_preds  = []
    true_label      = None
    true_diff       = None

    for model, tokenizer, cfg in loaded_models:
        print(f"  → {cfg['name']}...")

        test_dataset = TinyLogicDataset(test_df, tokenizer, MAX_LENGTH)
        test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        preds = get_predictions(model, test_loader, device)

        all_label_probs.append(preds["label_probs"])
        all_diff_probs.append(preds["diff_probs"])
        all_label_preds.append(np.argmax(preds["label_probs"], axis=1))
        all_diff_preds.append(np.argmax(preds["diff_probs"],  axis=1))

        if true_label is None:
            true_label = preds["label_true"]
            true_diff  = preds["diff_true"]

        m = compute_metrics(
            all_label_preds[-1], all_diff_preds[-1], true_label, true_diff
        )
        individual_results[cfg["name"]] = m
        print(f"     Label F1: {m['label_f1']:.4f} | "
              f"Diff F1: {m['diff_f1']:.4f} | "
              f"Diff Acc: {m['diff_acc']:.4f} ({m['diff_acc']*100:.2f}%)")

    print(f"\n Aplicăm metodele de ensemble...")
    ensemble_results = {}

    print("  1. Majority Voting...")
    mv_label, mv_diff = majority_voting(all_label_preds, all_diff_preds)
    ensemble_results["Majority Voting"] = compute_metrics(
        mv_label, mv_diff, true_label, true_diff
    )

    print("  2. Average Probabilities...")
    avg_label, avg_diff = average_probabilities(all_label_probs, all_diff_probs)
    ensemble_results["Average Probabilities"] = compute_metrics(
        avg_label, avg_diff, true_label, true_diff
    )

    print("  3. Weighted Ensemble...")
    w_label, w_diff = weighted_ensemble(
        all_label_probs, all_diff_probs, weights, names
    )
    ensemble_results["Weighted Ensemble"] = compute_metrics(
        w_label, w_diff, true_label, true_diff
    )

    print(f"\n{'='*70}")
    print(f"  TABEL COMPARATIV — BERT v2 + XLM-RoBERTa v2")
    print(f"{'='*70}")
    print(f"  {'Model':<28} {'LblAcc':>8} {'LblF1':>8} "
          f"{'DiffAcc':>8} {'DiffF1':>8} {'AccGen':>8}")
    print(f"  {'─'*68}")

    best_ind_diff = 0
    print(f"  MODELE INDIVIDUALE:")
    for name, m in individual_results.items():
        if m["diff_acc"] > best_ind_diff:
            best_ind_diff = m["diff_acc"]
        print(f"  {name:<28} {m['label_acc']:>8.4f} {m['label_f1']:>8.4f} "
              f"{m['diff_acc']:>8.4f} {m['diff_f1']:>8.4f} "
              f"{m['overall_acc']:>8.4f}")

    print(f"  {'─'*68}")
    print(f"  ENSEMBLE:")

    best_ens_diff = 0
    best_method   = ""
    best_overall  = 0

    for name, m in ensemble_results.items():
        marker = ""
        if m["diff_acc"] > best_ind_diff:
            marker = " ✅ DIFF îmbunătățit!"
        if m["diff_acc"] > best_ens_diff:
            best_ens_diff = m["diff_acc"]
            best_method   = name
        if m["overall_acc"] > best_overall:
            best_overall = m["overall_acc"]

        print(f"  {name:<28} {m['label_acc']:>8.4f} {m['label_f1']:>8.4f} "
              f"{m['diff_acc']:>8.4f} {m['diff_f1']:>8.4f} "
              f"{m['overall_acc']:>8.4f}{marker}")

    print(f"{'='*70}")

    improvement = (best_ens_diff - best_ind_diff) * 100
    sign = "+" if improvement >= 0 else ""
    print(f"\n  Difficulty individual (cel mai bun): "
          f"{best_ind_diff:.4f} ({best_ind_diff*100:.2f}%)")
    print(f"  Difficulty ensemble (cel mai bun):   "
          f"{best_ens_diff:.4f} ({best_ens_diff*100:.2f}%)")
    print(f"  Îmbunătățire:                        {sign}{improvement:.2f}%")

    best_ens_name = max(ensemble_results, key=lambda x: ensemble_results[x]["diff_acc"])
    print(f"\n RAPORT DETALIAT — {best_ens_name}:")

    if best_ens_name == "Majority Voting":
        best_label_p, best_diff_p = mv_label, mv_diff
    elif best_ens_name == "Average Probabilities":
        best_label_p, best_diff_p = avg_label, avg_diff
    else:
        best_label_p, best_diff_p = w_label, w_diff

    print("\n  DIFFICULTY:")
    diff_names = [ID_TO_DIFF[i] for i in sorted(ID_TO_DIFF.keys())]
    print(classification_report(
        true_diff, best_diff_p, target_names=diff_names
    ))

    print("\n  LABEL:")
    label_names = [id_to_label[i] for i in sorted(id_to_label.keys())]
    print(classification_report(
        true_label, best_label_p, target_names=label_names
    ))

    print("\nSalvăm rapoartele...")

    report = {
        "timestamp":          time.strftime("%Y-%m-%d %H:%M:%S"),
        "models":             [cfg["name"] for _, _, cfg in loaded_models],
        "pooling":            "mean (ambele modele)",
        "individual_results": individual_results,
        "ensemble_results":   ensemble_results,
        "best_ensemble":      best_ens_name,
        "best_diff_acc":      best_ens_diff,
        "improvement":        improvement,
    }

    json_path = os.path.join(REPORTS_DIR, "ensemble_v2_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    txt_lines = [
        "=" * 65,
        "RAPORT ENSEMBLE v2 — BERT Românesc v2 + XLM-RoBERTa v2",
        f"Generat: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "Ambele modele: Mean Pooling + Label Smoothing + Cosine Scheduler",
        "=" * 65,
        "",
        "MODELE INDIVIDUALE:",
    ]
    for name, m in individual_results.items():
        txt_lines += [
            f"  {name}:",
            f"    Label Accuracy:      {m['label_acc']:.4f} ({m['label_acc']*100:.2f}%)",
            f"    Label F1:            {m['label_f1']:.4f}",
            f"    Difficulty Accuracy: {m['diff_acc']:.4f} ({m['diff_acc']*100:.2f}%)",
            f"    Difficulty F1:       {m['diff_f1']:.4f}",
            f"    Acuratețe generală:  {m['overall_acc']:.4f} ({m['overall_acc']*100:.2f}%)",
            "",
        ]

    txt_lines += ["ENSEMBLE:"]
    for name, m in ensemble_results.items():
        txt_lines += [
            f"  {name}:",
            f"    Label Accuracy:      {m['label_acc']:.4f} ({m['label_acc']*100:.2f}%)",
            f"    Label F1:            {m['label_f1']:.4f}",
            f"    Difficulty Accuracy: {m['diff_acc']:.4f} ({m['diff_acc']*100:.2f}%)",
            f"    Difficulty F1:       {m['diff_f1']:.4f}",
            f"    Acuratețe generală:  {m['overall_acc']:.4f} ({m['overall_acc']*100:.2f}%)",
            "",
        ]

    txt_lines += [
        f"CEL MAI BUN ENSEMBLE: {best_ens_name}",
        f"Difficulty Accuracy:  {best_ens_diff:.4f} ({best_ens_diff*100:.2f}%)",
        f"Îmbunătățire față de individual: {sign}{improvement:.2f}%",
    ]

    txt_path = os.path.join(REPORTS_DIR, "ensemble_v2_report.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(txt_lines))

    print(f"  Raport JSON: {json_path}")
    print(f"  Raport TXT:  {txt_path}")

    print(f"\n{'='*65}")
    print(f"  ENSEMBLE COMPLET!")
    print(f"  Difficulty individual: {best_ind_diff:.4f} ({best_ind_diff*100:.2f}%)")
    print(f"  Difficulty ensemble:   {best_ens_diff:.4f} ({best_ens_diff*100:.2f}%)")
    print(f"  Îmbunătățire:         {sign}{improvement:.2f}%")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()