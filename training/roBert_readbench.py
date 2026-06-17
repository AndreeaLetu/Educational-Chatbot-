
import os
import json
import time
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_CONFIG = {
    "model_name": "readerbench/RoBERT-base",
    "data_dir": os.path.join(BASE_DIR, "data", "augmented"),
    "output_dir": os.path.join(BASE_DIR, "models", "robert"),
    "checkpoint_dir": os.path.join(BASE_DIR, "models", "checkpoints", "robert"),
    "reports_dir": os.path.join(BASE_DIR, "reports"),

    "max_length": 128,
    "batch_size": 16,
    "num_epochs": 15,
    "learning_rate": 5e-6,
    "weight_decay": 0.001,
    "warmup_ratio": 0.15,
    "dropout": 0.2,

    "label_loss_w":    0.4,
    "diff_loss_w":     0.6,

    "seed": 42,
}

DIFFICULTY_MAP = {0: "beginner", 1: "intermediate", 2: "advanced"}


def set_seed(seed):
    torch.manual_seed(seed)
    import numpy as np
    np.random.seed(seed)


class TinyLogicDataset(Dataset):
    def __init__(self, texts, labels, difficulties, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.difficulties = difficulties
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label_id": torch.tensor(self.labels[idx], dtype=torch.long),
            "difficulty_id": torch.tensor(self.difficulties[idx], dtype=torch.long),
        }



class MultiTaskRoBERT(nn.Module):


    def __init__(self, model_name, num_labels, num_difficulties=3, dropout=0.3):
        super().__init__()

        print(f"   🔄 Încărcare model {model_name}...")
        self.robert = AutoModel.from_pretrained(model_name)
        hidden_size = self.robert.config.hidden_size

        if hasattr(self.robert, "encoder") and hasattr(self.robert.encoder, "layer"):
            for layer in self.robert.encoder.layer[:2]:
                for param in layer.parameters():
                    param.requires_grad = False

        self.dropout = nn.Dropout(dropout)

        self.head_label = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_labels),
        )

        self.head_difficulty = nn.Sequential(
            nn.Linear(hidden_size, 256),  # 128 → 256
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),  # layer extra
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_difficulties),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.robert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = self.dropout(outputs.last_hidden_state[:, 0, :])

        logits_label = self.head_label(cls_output)
        logits_difficulty = self.head_difficulty(cls_output)

        return logits_label, logits_difficulty


def load_data(data_dir):
    print(f"\n Încărcare date din: {data_dir}")

    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    val = pd.read_csv(os.path.join(data_dir, "val.csv"))
    test = pd.read_csv(os.path.join(data_dir, "test.csv"))

    with open(os.path.join(data_dir, "label_map.json"), encoding="utf-8") as f:
        label_map = json.load(f)

    label_map = {int(k): v for k, v in label_map.items()}

    print(f"   Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    print(f"   Labels: {list(label_map.values())}")

    return train, val, test, label_map



def compute_metrics(preds_label, preds_diff, true_label, true_diff, label_names):
    diff_names = list(DIFFICULTY_MAP.values())

    metrics = {
        "label_accuracy": round(accuracy_score(true_label, preds_label), 4),
        "label_f1_macro": round(f1_score(true_label, preds_label, average="macro", zero_division=0), 4),
        "label_f1_weighted": round(f1_score(true_label, preds_label, average="weighted", zero_division=0), 4),
        "label_precision": round(precision_score(true_label, preds_label, average="macro", zero_division=0), 4),
        "label_recall": round(recall_score(true_label, preds_label, average="macro", zero_division=0), 4),
        "label_report": classification_report(true_label, preds_label, target_names=label_names, zero_division=0),
        "label_confusion": confusion_matrix(true_label, preds_label).tolist(),

        "diff_accuracy": round(accuracy_score(true_diff, preds_diff), 4),
        "diff_f1_macro": round(f1_score(true_diff, preds_diff, average="macro", zero_division=0), 4),
        "diff_f1_weighted": round(f1_score(true_diff, preds_diff, average="weighted", zero_division=0), 4),
        "diff_precision": round(precision_score(true_diff, preds_diff, average="macro", zero_division=0), 4),
        "diff_recall": round(recall_score(true_diff, preds_diff, average="macro", zero_division=0), 4),
        "diff_report": classification_report(true_diff, preds_diff, target_names=diff_names, zero_division=0),
        "diff_confusion": confusion_matrix(true_diff, preds_diff).tolist(),
    }

    return metrics


def print_metrics(metrics, split="Validation"):
    print(f"\n{'=' * 60}")
    print(f"  METRICI — {split.upper()}")
    print(f"{'=' * 60}")

    print(f"\n   LABEL:")
    print(f"     Accuracy:    {metrics['label_accuracy']:.4f}")
    print(f"     F1 macro:    {metrics['label_f1_macro']:.4f}")
    print(f"     Precision:   {metrics['label_precision']:.4f}")
    print(f"     Recall:      {metrics['label_recall']:.4f}")

    print(f"\n DIFFICULTY:")
    print(f"     Accuracy:    {metrics['diff_accuracy']:.4f}")
    print(f"     F1 macro:    {metrics['diff_f1_macro']:.4f}")
    print(f"     Precision:   {metrics['diff_precision']:.4f}")
    print(f"     Recall:      {metrics['diff_recall']:.4f}")



def train_epoch(model, loader, optimizer, scheduler, config, device):
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0

    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        lbl = batch["label_id"].to(device)
        diff = batch["difficulty_id"].to(device)

        optimizer.zero_grad()

        lg_lbl, lg_dif = model(ids, mask)

        loss_lbl = criterion(lg_lbl, lbl)
        loss_dif = criterion(lg_dif, diff)
        loss = config["label_loss_w"] * loss_lbl + config["diff_loss_w"] * loss_dif

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, device, label_names):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0

    all_pred_lbl, all_pred_dif = [], []
    all_true_lbl, all_true_dif = [], []

    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            lbl = batch["label_id"].to(device)
            diff = batch["difficulty_id"].to(device)

            lg_lbl, lg_dif = model(ids, mask)

            loss_lbl = criterion(lg_lbl, lbl)
            loss_dif = criterion(lg_dif, diff)
            total_loss += (0.6 * loss_lbl + 0.4 * loss_dif).item()

            all_pred_lbl.extend(lg_lbl.argmax(1).cpu().numpy())
            all_pred_dif.extend(lg_dif.argmax(1).cpu().numpy())
            all_true_lbl.extend(lbl.cpu().numpy())
            all_true_dif.extend(diff.cpu().numpy())

    avg_loss = total_loss / len(loader)
    metrics = compute_metrics(
        all_pred_lbl, all_pred_dif,
        all_true_lbl, all_true_dif,
        label_names
    )

    return avg_loss, metrics


def train_model(config):
    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "🤖 " * 20)
    print("   TINYLOGIC — ANTRENAMENT RoBERT")
    print("   Model: readerbench/ro-bert")
    print("🤖 " * 20)
    print(f"\n  Device: {device}")

    train_df, val_df, test_df, label_map = load_data(config["data_dir"])
    label_names = [label_map[i] for i in sorted(label_map.keys())]
    num_labels = len(label_names)

    print(f"\n🔤 Încărcare tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

    def make_ds(df):
        return TinyLogicDataset(
            df["text_clean"].tolist(),
            df["label_id"].tolist(),
            df["difficulty"].tolist(),
            tokenizer,
            config["max_length"]
        )

    train_loader = DataLoader(make_ds(train_df), batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(make_ds(val_df), batch_size=config["batch_size"])
    test_loader = DataLoader(make_ds(test_df), batch_size=config["batch_size"])

    print(f"\n🏗️  Construire model multi-task...")
    model = MultiTaskRoBERT(
        config["model_name"],
        num_labels,
        dropout=config["dropout"]
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"   Parametri antrenabili: {trainable:,} / {total:,} ({100 * trainable / total:.1f}%)")

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )

    total_steps = len(train_loader) * config["num_epochs"]
    warmup_steps = int(total_steps * config["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    print(f"\n🚀 Antrenament {config['num_epochs']} epoci...\n")
    print(f"  {'Ep':>3} | {'TrLoss':>8} | {'VlLoss':>8} | {'LblAcc':>8} | {'DifAcc':>8} | {'F1comb':>8}")
    print(f"  {'─' * 60}")

    history = []
    best_f1 = 0.0
    best_state = None
    train_start = time.time()

    for epoch in range(1, config["num_epochs"] + 1):
        epoch_start = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, scheduler, config, device)
        val_loss, val_metrics = evaluate(model, val_loader, device, label_names)

        val_f1 = (val_metrics["label_f1_macro"] * 0.6 +
                  val_metrics["diff_f1_macro"] * 0.4)

        marker = " ✨" if val_f1 > best_f1 else ""
        print(f"  {epoch:>3} | {train_loss:>8.4f} | {val_loss:>8.4f} | "
              f"{val_metrics['label_accuracy']:>8.4f} | {val_metrics['diff_accuracy']:>8.4f} | "
              f"{val_f1:>8.4f}{marker}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "label_accuracy": val_metrics["label_accuracy"],
            "label_f1_macro": val_metrics["label_f1_macro"],
            "diff_accuracy": val_metrics["diff_accuracy"],
            "diff_f1_macro": val_metrics["diff_f1_macro"],
            "val_f1_combined": round(val_f1, 4),
            "epoch_time_s": round(time.time() - epoch_start, 1),
        })

    train_time = round(time.time() - train_start, 1)

    model.load_state_dict(best_state)

    inf_start = time.time()
    test_loss, test_metrics = evaluate(model, test_loader, device, label_names)
    inf_time = round((time.time() - inf_start) / len(test_df) * 1000, 2)

    print_metrics(test_metrics, "Test")

    print(f"\n  ✅ Best val F1 combinat: {best_f1:.4f}")
    print(f"  ⏱️  Timp total antrenament: {train_time}s")
    print(f"  ⏱️  Inferență medie: {inf_time}ms/exemplu")

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save({
        "model_state_dict": best_state,
        "label_map": label_map,
        "config": config,
        "test_metrics": {k: v for k, v in test_metrics.items()
                         if not k.endswith("_report")},
    }, output_dir / "robert_model.pt")

    tokenizer.save_pretrained(str(output_dir / "tokenizer"))

    with open(output_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"\n💾 Model salvat în: {output_dir}")
    print(f"   - robert_model.pt")
    print(f"   - tokenizer/")
    print(f"   - training_history.json")

    return model, test_metrics, history



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TinyLogic — Fine-tuning RoBERT")
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["num_epochs"])
    parser.add_argument("--batch_size", type=int, default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG["learning_rate"])
    parser.add_argument("--max_length", type=int, default=DEFAULT_CONFIG["max_length"])
    args = parser.parse_args()

    config = {**DEFAULT_CONFIG}
    config["num_epochs"] = args.epochs
    config["batch_size"] = args.batch_size
    config["learning_rate"] = args.lr
    config["max_length"] = args.max_length

    train_model(config)

    print("\n Antrenament complet!")
    print("   Modelul RoBERT e gata pentru inferență.\n")