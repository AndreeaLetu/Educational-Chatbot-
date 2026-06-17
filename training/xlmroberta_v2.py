import os
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModel,
    get_cosine_with_hard_restarts_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score, classification_report

BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR       = os.path.join(BASE_DIR, "data", "augmented")
MODELS_DIR     = os.path.join(BASE_DIR, "models")
CHECKPOINT_DIR = os.path.join(MODELS_DIR, "checkpoints")
FINAL_DIR      = os.path.join(MODELS_DIR, "final")
LOGS_DIR       = os.path.join(BASE_DIR, "training", "logs")

MODEL_NAME    = "xlm-roberta-base"
SAVE_NAME     = "tinylogic_xlmroberta_v2"

MAX_LENGTH       = 128
BATCH_SIZE       = 16
EPOCHS           = 10
LR               = 1e-5
FREEZE_LAYERS    = 4
SEED             = 42
LABEL_SMOOTHING  = 0.1
COSINE_CYCLES    = 3
LOSS_DIFF_WEIGHT = 2.0

torch.manual_seed(SEED)
np.random.seed(SEED)


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


class MultiTaskXLMRoBERTa(nn.Module):
    def __init__(self, model_name, num_labels, num_difficulties, freeze_layers=4):
        super().__init__()

        self.bert = AutoModel.from_pretrained(model_name)

        try:
            layers = list(self.bert.encoder.layer)
            for i, layer in enumerate(layers):
                if i < freeze_layers:
                    for param in layer.parameters():
                        param.requires_grad = False
            print(f"  Layere înghețate: {freeze_layers}/{len(layers)}")
        except AttributeError:
            children = list(self.bert.children())
            for i, child in enumerate(children[:freeze_layers]):
                for param in child.parameters():
                    param.requires_grad = False
            print(f"  Layere înghețate: {freeze_layers} (fallback)")

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
        outputs       = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = self.mean_pooling(outputs, attention_mask)
        label_logits  = self.label_head(pooled_output)
        diff_logits   = self.difficulty_head(pooled_output)
        return label_logits, diff_logits


def train_one_epoch(model, loader, optimizer, scheduler, device, epoch, total_epochs):
    model.train()
    total_loss = 0
    all_label_preds, all_label_true = [], []
    all_diff_preds,  all_diff_true  = [], []

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    for step, batch in enumerate(loader):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["label"].to(device)
        difficulties   = batch["difficulty"].to(device)

        optimizer.zero_grad()

        label_logits, diff_logits = model(input_ids, attention_mask)

        loss_label = criterion(label_logits, labels)
        loss_diff  = criterion(diff_logits, difficulties)
        loss       = loss_label + loss_diff * LOSS_DIFF_WEIGHT

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        all_label_preds.extend(torch.argmax(label_logits, dim=1).cpu().tolist())
        all_label_true.extend(labels.cpu().tolist())
        all_diff_preds.extend(torch.argmax(diff_logits, dim=1).cpu().tolist())
        all_diff_true.extend(difficulties.cpu().tolist())

        if (step + 1) % 20 == 0:
            print(f"   Epocă {epoch}/{total_epochs} | Batch {step+1}/{len(loader)} | Loss: {loss.item():.4f}")

    avg_loss  = total_loss / len(loader)
    label_acc = accuracy_score(all_label_true, all_label_preds)
    label_f1  = f1_score(all_label_true, all_label_preds, average="macro", zero_division=0)
    diff_acc  = accuracy_score(all_diff_true, all_diff_preds)
    diff_f1   = f1_score(all_diff_true, all_diff_preds, average="macro", zero_division=0)

    return {"loss": avg_loss, "label_acc": label_acc, "label_f1": label_f1,
            "diff_acc": diff_acc, "diff_f1": diff_f1}


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

    return {"loss": avg_loss, "label_acc": label_acc, "label_f1": label_f1,
            "diff_acc": diff_acc, "diff_f1": diff_f1,
            "label_preds": all_label_preds, "label_true": all_label_true,
            "diff_preds": all_diff_preds, "diff_true": all_diff_true}


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
    print(f"  TinyLogic — Fine-Tuning XLM-RoBERTa v2")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Task:  Multi-task (Label + Difficulty)")
    print(f"  Save:  {SAVE_NAME}")
    print(f"  Îmbunătățiri: Label Smoothing + Cosine Scheduler + Mean Pooling")
    print("="*60 + "\n")

    for d in [CHECKPOINT_DIR, FINAL_DIR, LOGS_DIR]:
        os.makedirs(d, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Dispozitiv: {device}")

    print(f"\n📂 Încărcăm datele din: {DATA_DIR}")
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    val_df   = pd.read_csv(os.path.join(DATA_DIR, "val.csv"))
    test_df  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

    for col in ["text_clean", "label_id", "difficulty"]:
        assert col in train_df.columns, f"Coloana '{col}' lipsește!"

    print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    with open(os.path.join(DATA_DIR, "label_map.json")) as f:
        label_map = json.load(f)

    num_labels       = len(label_map)
    num_difficulties = 3
    id_to_label      = {v: k for k, v in label_map.items()}
    id_to_diff       = {0: "beginner", 1: "intermediate", 2: "advanced"}

    print(f"  Label-uri ({num_labels}): {list(label_map.keys())}")

    print(f"\n🔤 Încărcăm tokenizer-ul ({MODEL_NAME})...")
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

    print(f"\n🤖 Încărcăm modelul ({MODEL_NAME})...")
    model = MultiTaskXLMRoBERTa(MODEL_NAME, num_labels, num_difficulties, FREEZE_LAYERS)
    model = model.to(device)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parametri totali:      {total_params:,}")
    print(f"  Parametri antrenabili: {trainable_params:,}")

    optimizer   = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                        lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler   = get_cosine_with_hard_restarts_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps,
        num_cycles=COSINE_CYCLES
    )

    print(f"\n  Scheduler: Cosine cu {COSINE_CYCLES} restarts")
    print(f"  Label Smoothing: {LABEL_SMOOTHING}")
    print(f"  Pooling: Mean Pooling")

    print(f"\n🚀 Startăm antrenamentul ({EPOCHS} epoci)...")
    print(f"  Estimare timp CPU: ~{len(train_loader) * EPOCHS * 3 // 60} minute\n")

    history        = []
    best_val_f1    = 0.0
    best_epoch     = 0
    training_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        print(f"\n{'─'*60}")
        print(f"  EPOCĂ {epoch}/{EPOCHS}")
        print(f"{'─'*60}")

        train_metrics = train_one_epoch(model, train_loader, optimizer, scheduler, device, epoch, EPOCHS)
        val_metrics   = evaluate(model, val_loader, device, "Val")

        elapsed = time.time() - epoch_start
        print_epoch_summary(epoch, EPOCHS, train_metrics, val_metrics, elapsed)

        combined_f1 = (val_metrics["label_f1"] + val_metrics["diff_f1"]) / 2
        if combined_f1 > best_val_f1:
            best_val_f1 = combined_f1
            best_epoch  = epoch
            torch.save({
                "epoch": epoch, "model_state": model.state_dict(),
                "val_f1": combined_f1, "label_map": label_map, "model_name": MODEL_NAME,
            }, os.path.join(CHECKPOINT_DIR, f"best_model_{SAVE_NAME}.pt"))
            print(f"  💾 Checkpoint salvat! (F1 combinat: {combined_f1:.4f})")

        history.append({"epoch": epoch, "train": train_metrics,
                        "val": val_metrics, "elapsed_sec": elapsed})

    print(f"\n{'='*60}")
    print("  EVALUARE FINALĂ PE TEST SET")
    print(f"{'='*60}")

    checkpoint = torch.load(os.path.join(CHECKPOINT_DIR, f"best_model_{SAVE_NAME}.pt"),
                            map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    print(f"  Cel mai bun model: Epocă {best_epoch} (Val F1: {best_val_f1:.4f})")

    test_metrics = evaluate(model, test_loader, device, "Test")
    overall_acc  = (test_metrics["label_acc"] + test_metrics["diff_acc"]) / 2
    overall_f1   = (test_metrics["label_f1"]  + test_metrics["diff_f1"])  / 2

    print(f"""
  Test Label Accuracy:      {test_metrics['label_acc']:.4f}
  Test Label F1 (macro):    {test_metrics['label_f1']:.4f}
  Test Difficulty Accuracy: {test_metrics['diff_acc']:.4f}
  Test Difficulty F1:       {test_metrics['diff_f1']:.4f}
  ══════════════════════════════════════════
  Acuratețe generală:       {overall_acc:.4f} ({overall_acc*100:.2f}%)
  F1 general (macro):       {overall_f1:.4f} ({overall_f1*100:.2f}%)
  ══════════════════════════════════════════
    """)

    print("  RAPORT DETALIAT — LABEL:")
    label_names = [id_to_label[i] for i in sorted(id_to_label.keys())]
    print(classification_report(test_metrics["label_true"], test_metrics["label_preds"],
                                target_names=label_names))

    print("  RAPORT DETALIAT — DIFFICULTY:")
    diff_names = [id_to_diff[i] for i in sorted(id_to_diff.keys())]
    print(classification_report(test_metrics["diff_true"], test_metrics["diff_preds"],
                                target_names=diff_names))

    total_time = time.time() - training_start
    final_path = os.path.join(FINAL_DIR, SAVE_NAME)
    os.makedirs(final_path, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(final_path, "model_weights.pt"))
    tokenizer.save_pretrained(final_path)

    config = {
        "model_name": MODEL_NAME, "save_name": SAVE_NAME,
        "max_length": MAX_LENGTH, "batch_size": BATCH_SIZE,
        "epochs": EPOCHS, "learning_rate": LR, "freeze_layers": FREEZE_LAYERS,
        "label_smoothing": LABEL_SMOOTHING, "cosine_cycles": COSINE_CYCLES,
        "loss_diff_weight": LOSS_DIFF_WEIGHT, "pooling": "mean",
        "num_labels": num_labels, "num_difficulties": num_difficulties,
        "label_map": label_map, "best_epoch": best_epoch, "best_val_f1": best_val_f1,
        "test_label_acc": test_metrics["label_acc"], "test_label_f1": test_metrics["label_f1"],
        "test_diff_acc": test_metrics["diff_acc"], "test_diff_f1": test_metrics["diff_f1"],
        "overall_acc": overall_acc, "overall_f1": overall_f1,
        "total_time_min": total_time / 60,
    }
    with open(os.path.join(final_path, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    log_path = os.path.join(LOGS_DIR, f"training_history_{SAVE_NAME}.json")
    with open(log_path, "w", encoding="utf-8") as f:
        clean_history = [{"epoch": h["epoch"], "elapsed_sec": h["elapsed_sec"],
                          "train": h["train"],
                          "val": {k: v for k, v in h["val"].items()
                                  if k not in ("label_preds", "label_true", "diff_preds", "diff_true")}}
                         for h in history]
        json.dump({"config": config, "history": clean_history}, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"  ✅ ANTRENAMENT COMPLET!")
    print(f"  Timp total:   {total_time/60:.1f} minute")
    print(f"  Model salvat: {final_path}")
    print(f"  Log:          {log_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()