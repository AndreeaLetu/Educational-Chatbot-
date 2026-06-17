"""
Model: dumitrescuv/bert-base-romanian-cased-v1
Task: Multi-task — clasificare label + difficulty
Date: data/augmented/

Rulare:
    python training/train_model.py

Rezultate salvate în:
    models/checkpoints/   ← cel mai bun model din fiecare epocă
    models/final/         ← modelul final
    training/logs/        ← metrici JSON pentru raport
"""

import os
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR       = os.path.join(BASE_DIR, "data", "augmented")
MODELS_DIR     = os.path.join(BASE_DIR, "models")
CHECKPOINT_DIR = os.path.join(MODELS_DIR, "checkpoints")
FINAL_DIR      = os.path.join(MODELS_DIR, "final")
LOGS_DIR       = os.path.join(BASE_DIR, "training", "logs")

MODEL_NAME  = "dumitrescustefan/bert-base-romanian-cased-v1"
MAX_LENGTH  = 128       # lungimea maximă a tokenilor
BATCH_SIZE  = 16        # numărul de exemple procesate simultan
EPOCHS       = 10      # era 5
LR           = 1e-5    # era 2e-5
FREEZE_LAYERS = 4       # primele 8 layere BERT înghețate (viteză pe CPU)
SEED        = 42


torch.manual_seed(SEED)
np.random.seed(SEED)


class TinyLogicDataset(Dataset):
    """

    Fiecare exemplu conține:
      - input_ids:      tokenii transformați în numere
      - attention_mask: 1 = token real, 0 = padding
      - label_id:       eticheta tipului de întrebare (0-5)
      - difficulty:     nivelul de dificultate (0=beginner, 1=intermediate, 2=advanced)
    """
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


# ──────────────────────────────────────────────
# PASUL 2: ARHITECTURA MODELULUI
# ──────────────────────────────────────────────
class MultiTaskBERT(nn.Module):

    def __init__(self, model_name, num_labels, num_difficulties, freeze_layers=8):
        super().__init__()

        self.bert = AutoModel.from_pretrained(model_name)

        for i, layer in enumerate(self.bert.encoder.layer):
            if i < freeze_layers:
                for param in layer.parameters():
                    param.requires_grad = False

        hidden_size = self.bert.config.hidden_size

        self.label_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_labels)
        )

        self.difficulty_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_difficulties)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        cls_output = outputs.last_hidden_state[:, 0, :]

        label_logits = self.label_head(cls_output)
        diff_logits  = self.difficulty_head(cls_output)

        return label_logits, diff_logits


def train_one_epoch(model, loader, optimizer, scheduler, device, epoch, total_epochs):
    model.train()
    total_loss = 0
    all_label_preds, all_label_true = [], []
    all_diff_preds,  all_diff_true  = [], []

    criterion = nn.CrossEntropyLoss()

    for step, batch in enumerate(loader):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["label"].to(device)
        difficulties   = batch["difficulty"].to(device)

        optimizer.zero_grad()

        label_logits, diff_logits = model(input_ids, attention_mask)

        loss_label = criterion(label_logits, labels)
        loss_diff  = criterion(diff_logits, difficulties)
        loss = loss_label + loss_diff * 2.0 # (pondere dubla)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # stabilitate
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        all_label_preds.extend(torch.argmax(label_logits, dim=1).cpu().tolist())
        all_label_true.extend(labels.cpu().tolist())
        all_diff_preds.extend(torch.argmax(diff_logits, dim=1).cpu().tolist())
        all_diff_true.extend(difficulties.cpu().tolist())

        if (step + 1) % 20 == 0:
            print(f"   Epocă {epoch}/{total_epochs} | Batch {step+1}/{len(loader)} | Loss: {loss.item():.4f}")

    avg_loss   = total_loss / len(loader)
    label_acc  = accuracy_score(all_label_true, all_label_preds)
    label_f1   = f1_score(all_label_true, all_label_preds, average="macro", zero_division=0)
    diff_acc   = accuracy_score(all_diff_true, all_diff_preds)
    diff_f1    = f1_score(all_diff_true, all_diff_preds, average="macro", zero_division=0)

    return {
        "loss": avg_loss,
        "label_acc": label_acc, "label_f1": label_f1,
        "diff_acc": diff_acc,   "diff_f1": diff_f1
    }


def evaluate(model, loader, device, split_name="Val"):
    model.eval()
    total_loss = 0
    all_label_preds, all_label_true = [], []
    all_diff_preds,  all_diff_true  = [], []

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)
            difficulties   = batch["difficulty"].to(device)

            label_logits, diff_logits = model(input_ids, attention_mask)

            loss = criterion(label_logits, labels) + criterion(diff_logits, difficulties)
            total_loss += loss.item()

            all_label_preds.extend(torch.argmax(label_logits, dim=1).cpu().tolist())
            all_label_true.extend(labels.cpu().tolist())
            all_diff_preds.extend(torch.argmax(diff_logits, dim=1).cpu().tolist())
            all_diff_true.extend(difficulties.cpu().tolist())

    avg_loss  = total_loss / len(loader)
    label_acc = accuracy_score(all_label_true, all_label_preds)
    label_f1  = f1_score(all_label_true, all_label_preds, average="macro", zero_division=0)
    diff_acc  = accuracy_score(all_diff_true, all_diff_preds)
    diff_f1   = f1_score(all_diff_true, all_diff_preds, average="macro", zero_division=0)

    return {
        "loss": avg_loss,
        "label_acc": label_acc, "label_f1": label_f1,
        "diff_acc": diff_acc,   "diff_f1": diff_f1,
        "label_preds": all_label_preds, "label_true": all_label_true,
        "diff_preds":  all_diff_preds,  "diff_true":  all_diff_true,
    }


def print_epoch_summary(epoch, total_epochs, train_m, val_m, elapsed):
    print(f"""
  ┌─────────────────────────────────────────────────────┐
  │  EPOCĂ {epoch}/{total_epochs}  ({elapsed:.0f}s)
  ├──────────────────┬──────────────┬───────────────────┤
  │  Metrică         │    TRAIN     │       VAL         │
  ├──────────────────┼──────────────┼───────────────────┤
  │  Loss            │   {train_m['loss']:.4f}     │    {val_m['loss']:.4f}        │
  │  Label Acc       │   {train_m['label_acc']:.4f}     │    {val_m['label_acc']:.4f}        │
  │  Label F1        │   {train_m['label_f1']:.4f}     │    {val_m['label_f1']:.4f}        │
  │  Difficulty Acc  │   {train_m['diff_acc']:.4f}     │    {val_m['diff_acc']:.4f}        │
  │  Difficulty F1   │   {train_m['diff_f1']:.4f}     │    {val_m['diff_f1']:.4f}        │
  └──────────────────┴──────────────┴───────────────────┘""")


def main():
    print("\n" + "="*60)
    print("  TinyLogic — Fine-Tuning BERT Românesc")
    print("  Model: dumitrescuv/bert-base-romanian-cased-v1")
    print("  Task:  Multi-task (Label + Difficulty)")
    print("="*60 + "\n")
    for d in [CHECKPOINT_DIR, FINAL_DIR, LOGS_DIR]:
        os.makedirs(d, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Dispozitiv: {device}")

    print(f"\n Încărcăm datele din: {DATA_DIR}")
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    val_df   = pd.read_csv(os.path.join(DATA_DIR, "val.csv"))
    test_df  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

    for col in ["text_clean", "label_id", "difficulty"]:
        assert col in train_df.columns, f"Coloana '{col}' lipsește din train.csv!"

    print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    label_map_path = os.path.join(DATA_DIR, "label_map.json")
    with open(label_map_path) as f:
        label_map = json.load(f)

    num_labels      = len(label_map)
    num_difficulties = 3  # beginner=0, intermediate=1, advanced=2
    id_to_label     = {v: k for k, v in label_map.items()}
    id_to_diff      = {0: "beginner", 1: "intermediate", 2: "advanced"}

    print(f"  Label-uri ({num_labels}): {list(label_map.keys())}")

    print(f"\n🔤 Descărcăm tokenizer-ul ({MODEL_NAME})...")
    print("  (prima rulare descarcă ~500MB — așteptați...)")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print("  Tokenizer gata!")

    print("\n📦 Pregătim dataset-urile...")
    train_dataset = TinyLogicDataset(train_df, tokenizer, MAX_LENGTH)
    val_dataset   = TinyLogicDataset(val_df,   tokenizer, MAX_LENGTH)
    test_dataset  = TinyLogicDataset(test_df,  tokenizer, MAX_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

    print(f"  Batch-uri train: {len(train_loader)}")
    print(f"  Batch-uri val:   {len(val_loader)}")

    print(f"\n🤖 Descărcăm modelul BERT ({MODEL_NAME})...")
    print("  (prima rulare descarcă ~500MB — așteptați...)")
    model = MultiTaskBERT(MODEL_NAME, num_labels, num_difficulties, FREEZE_LAYERS)
    model = model.to(device)

    total_params    = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parametri totali:     {total_params:,}")
    print(f"  Parametri antrenabili: {trainable_params:,}")
    print(f"  Layere înghețate:     {FREEZE_LAYERS}/12")

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=0.01
    )
    total_steps = len(train_loader) * EPOCHS
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,  # 10% warmup
        num_training_steps=total_steps
    )

    print(f"\n Start antrenament ({EPOCHS} epoci)...")
    print(f"  Estimare timp CPU: ~{len(train_loader) * EPOCHS * 3 // 60} minute\n")

    history = []
    best_val_f1   = 0.0
    best_epoch    = 0
    training_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        print(f"\n{'─'*60}")
        print(f"  EPOCĂ {epoch}/{EPOCHS}")
        print(f"{'─'*60}")

        train_metrics = train_one_epoch(model, train_loader, optimizer, scheduler, device, epoch, EPOCHS)

        val_metrics = evaluate(model, val_loader, device, "Val")

        elapsed = time.time() - epoch_start
        print_epoch_summary(epoch, EPOCHS, train_metrics, val_metrics, elapsed)

        combined_f1 = (val_metrics["label_f1"] + val_metrics["diff_f1"]) / 2
        if combined_f1 > best_val_f1:
            best_val_f1 = combined_f1
            best_epoch  = epoch
            checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")
            torch.save({
                "epoch":      epoch,
                "model_state": model.state_dict(),
                "val_f1":     combined_f1,
                "label_map":  label_map,
            }, checkpoint_path)
            print(f"  💾 Checkpoint salvat! (F1 combinat: {combined_f1:.4f})")

        history.append({
            "epoch":       epoch,
            "train":       train_metrics,
            "val":         val_metrics,
            "elapsed_sec": elapsed
        })

    print(f"\n{'='*60}")
    print("  EVALUARE FINALĂ PE TEST SET")
    print(f"{'='*60}")

    checkpoint = torch.load(os.path.join(CHECKPOINT_DIR, "best_model.pt"), map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    print(f"  Cel mai bun model: Epocă {best_epoch} (Val F1: {best_val_f1:.4f})")

    test_metrics = evaluate(model, test_loader, device, "Test")

    print(f"""
  Test Label Accuracy:      {test_metrics['label_acc']:.4f}
  Test Label F1 (macro):    {test_metrics['label_f1']:.4f}
  Test Difficulty Accuracy: {test_metrics['diff_acc']:.4f}
  Test Difficulty F1:       {test_metrics['diff_f1']:.4f}
    """)

    print("  RAPORT DETALIAT — LABEL:")
    label_names = [id_to_label[i] for i in sorted(id_to_label.keys())]
    print(classification_report(
        test_metrics["label_true"],
        test_metrics["label_preds"],
        target_names=label_names
    ))

    print("  RAPORT DETALIAT — DIFFICULTY:")
    diff_names = [id_to_diff[i] for i in sorted(id_to_diff.keys())]
    print(classification_report(
        test_metrics["diff_true"],
        test_metrics["diff_preds"],
        target_names=diff_names
    ))

    total_time = time.time() - training_start
    final_path = os.path.join(FINAL_DIR, "tinylogic_bert_romanian")
    os.makedirs(final_path, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(final_path, "model_weights.pt"))

    tokenizer.save_pretrained(final_path)

    config = {
        "model_name":       MODEL_NAME,
        "max_length":       MAX_LENGTH,
        "batch_size":       BATCH_SIZE,
        "epochs":           EPOCHS,
        "learning_rate":    LR,
        "freeze_layers":    FREEZE_LAYERS,
        "num_labels":       num_labels,
        "num_difficulties": num_difficulties,
        "label_map":        label_map,
        "best_epoch":       best_epoch,
        "best_val_f1":      best_val_f1,
        "test_label_acc":   test_metrics["label_acc"],
        "test_label_f1":    test_metrics["label_f1"],
        "test_diff_acc":    test_metrics["diff_acc"],
        "test_diff_f1":     test_metrics["diff_f1"],
        "total_time_min":   total_time / 60,
    }
    with open(os.path.join(final_path, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    log_path = os.path.join(LOGS_DIR, "training_history.json")
    with open(log_path, "w", encoding="utf-8") as f:
        clean_history = []
        for h in history:
            ch = {
                "epoch": h["epoch"],
                "elapsed_sec": h["elapsed_sec"],
                "train": {k: v for k, v in h["train"].items()},
                "val":   {k: v for k, v in h["val"].items()
                          if k not in ("label_preds", "label_true", "diff_preds", "diff_true")}
            }
            clean_history.append(ch)
        json.dump({"config": config, "history": clean_history}, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"  ANTRENAMENT COMPLET!")
    print(f"  Timp total: {total_time/60:.1f} minute")
    print(f"  Model final salvat în: {final_path}")
    print(f"  Log antrenament:       {log_path}")
    print(f"\n  Pasul următor: python training/analyze_results.py")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()