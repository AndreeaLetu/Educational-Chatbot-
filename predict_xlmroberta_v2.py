
import os, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models", "final", "tinylogic_xlmroberta_v2")
DATA_DIR  = os.path.join(BASE_DIR, "data", "augmented")
OUT_DIR   = os.path.join(BASE_DIR, "diagrams", "xlmroberta_v2")
os.makedirs(OUT_DIR, exist_ok=True)

MAX_LENGTH = 128
BATCH_SIZE = 16
CMAP = LinearSegmentedColormap.from_list("purple", ["#ffffff","#e8d5f5","#9c4dcc","#4a148c"], N=256)

with open(os.path.join(MODEL_DIR, "config.json"), encoding="utf-8") as f:
    config = json.load(f)
MODEL_NAME       = config["model_name"]
NUM_LABELS       = config["num_labels"]
NUM_DIFFICULTIES = config["num_difficulties"]
FREEZE_LAYERS    = config.get("freeze_layers", 4)
ID_TO_LABEL = {int(k): v for k, v in config["label_map"].items()}
ID_TO_DIFF  = {0: "beginner", 1: "intermediate", 2: "advanced"}
LABEL_NAMES = [ID_TO_LABEL[i] for i in range(NUM_LABELS)]
DIFF_NAMES  = [ID_TO_DIFF[i]  for i in range(NUM_DIFFICULTIES)]

class TinyLogicDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.texts = df["text_clean"].tolist(); self.labels = df["label_id"].tolist()
        self.difficulty = df["difficulty"].tolist(); self.tokenizer = tokenizer; self.max_length = max_length
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx], max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
        return {"input_ids": enc["input_ids"].squeeze(0), "attention_mask": enc["attention_mask"].squeeze(0),
                "label": torch.tensor(self.labels[idx], dtype=torch.long), "difficulty": torch.tensor(self.difficulty[idx], dtype=torch.long)}

class MultiTaskXLMRoBERTa(nn.Module):
    def __init__(self, model_name, num_labels, num_difficulties, freeze_layers=4):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        try:
            for i, layer in enumerate(self.bert.encoder.layer):
                if i < freeze_layers:
                    for p in layer.parameters(): p.requires_grad = False
        except AttributeError:
            for i, child in enumerate(list(self.bert.children())[:freeze_layers]):
                for p in child.parameters(): p.requires_grad = False
        h = self.bert.config.hidden_size
        self.label_head = nn.Sequential(nn.Dropout(0.3), nn.Linear(h,256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256,num_labels))
        self.difficulty_head = nn.Sequential(nn.Dropout(0.3), nn.Linear(h,256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256,num_difficulties))
    def mean_pooling(self, outputs, attention_mask):
        t = outputs.last_hidden_state; m = attention_mask.unsqueeze(-1).float()
        return torch.sum(t*m,1) / torch.clamp(m.sum(1), min=1e-9)
    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.label_head(self.mean_pooling(out, attention_mask)), self.difficulty_head(self.mean_pooling(out, attention_mask))

def run_inference(model, loader, device):
    model.eval(); yt_l,yp_l,yt_d,yp_d=[],[],[],[]
    with torch.no_grad():
        for batch in loader:
            ll,dl = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            yp_l.extend(torch.argmax(ll,1).cpu().tolist()); yt_l.extend(batch["label"].tolist())
            yp_d.extend(torch.argmax(dl,1).cpu().tolist()); yt_d.extend(batch["difficulty"].tolist())
    return np.array(yt_l), np.array(yp_l), np.array(yt_d), np.array(yp_d)

def plot_cm(cm, classes, title, path):
    n = len(classes)
    fig,ax = plt.subplots(figsize=(max(6,n*1.5), max(5,n*1.3)), facecolor="white")
    ax.set_facecolor("white")
    im = ax.imshow(cm, cmap=CMAP, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=8)
    ax.set_xticks(range(n)); ax.set_xticklabels(classes, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(n)); ax.set_yticklabels(classes, fontsize=9)
    ax.set_xlabel("Etichetă prezisă", fontsize=10); ax.set_ylabel("Etichetă reală", fontsize=10)
    ax.set_title(title, fontweight="bold", pad=12, fontsize=11)
    vmax = cm.max()
    for i in range(n):
        for j in range(n):
            v = cm[i,j]; color = "white" if v >= vmax*0.55 else "#333333"
            ax.text(j,i,str(v),ha="center",va="center",fontsize=11,color=color,fontweight="bold")
    plt.tight_layout()
    plt.savefig(path+".pdf", bbox_inches="tight", dpi=200, facecolor="white")
    plt.savefig(path+".png", bbox_inches="tight", dpi=200, facecolor="white")
    plt.close(); print(f"  ✓ {os.path.basename(path)}")

def main():
    print("\n" + "="*50 + "\n XLM-RoBERTa v2\n" + "="*50)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    test_df   = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    loader    = DataLoader(TinyLogicDataset(test_df, tokenizer, MAX_LENGTH), batch_size=BATCH_SIZE, shuffle=False)
    model = MultiTaskXLMRoBERTa(MODEL_NAME, NUM_LABELS, NUM_DIFFICULTIES, FREEZE_LAYERS)
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR,"model_weights.pt"), map_location=device))
    model.to(device); print("  Model încărcat!\n")
    yt_l,yp_l,yt_d,yp_d = run_inference(model, loader, device)
    la=accuracy_score(yt_l,yp_l); lf=f1_score(yt_l,yp_l,average="macro",zero_division=0)
    da=accuracy_score(yt_d,yp_d); df=f1_score(yt_d,yp_d,average="macro",zero_division=0)
    print(f"  Label — Acc:{la:.4f} F1:{lf:.4f}\n  Diff  — Acc:{da:.4f} F1:{df:.4f}\n")
    print("── Classification Report — Label ──────────────────────")
    print(classification_report(yt_l, yp_l, target_names=LABEL_NAMES))
    print("── Classification Report — Dificultate ────────────────")
    print(classification_report(yt_d, yp_d, target_names=DIFF_NAMES))
    plot_cm(confusion_matrix(yt_l,yp_l), LABEL_NAMES, f"XLM-RoBERTa varianta 2 — Label\nAcc={la:.2%}  F1={lf:.2%}", os.path.join(OUT_DIR,"cm_label"))
    plot_cm(confusion_matrix(yt_d,yp_d), DIFF_NAMES,  f"XLM-RoBERTa varianta 2  — Dificultate\nAcc={da:.2%}  F1={df:.2%}", os.path.join(OUT_DIR,"cm_diff"))
    print(f"\n  ✅ Gata! Fișiere în: {OUT_DIR}")

if __name__ == "__main__":
    main()