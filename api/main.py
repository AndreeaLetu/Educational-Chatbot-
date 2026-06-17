import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn

from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel

BASE_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from app.core.rag import RAGEngine

MODELS_CONFIG = [
    {
        "name":      "BERT Românesc v2",
        "save_name": "tinylogic_bert_romanian_v2",
    },
    {
        "name":      "XLM-RoBERTa v2",
        "save_name": "tinylogic_xlmroberta_v2",
    },
]

MAX_LENGTH = 128

ID_TO_LABEL = {
    0: "code_help",
    1: "concept_explanation",
    2: "exercise_request",
    3: "hint_request",
    4: "motivation_or_encouragement",
    5: "solution_request",
}
ID_TO_DIFF = {
    0: "beginner",
    1: "intermediate",
    2: "advanced",
}

MOTIVATION_KEYWORDS = {
    "obosit", "obosite", "greu", "dificil", "nu înțeleg",
    "nu pot", "renunț", "plictisit", "nu știu", "trist",
    "supărat", "dezamăgit", "frustrat", "îmi e", "ajutor",
    "nu merge", "nu reușesc"
}

CONCEPT_KEYWORDS = {
    "explică", "explica", "ce este", "ce înseamnă",
    "cum funcționează", "de ce funcționează", "ce face",
    "cum se", "poți explica", "definește", "ce sunt",
    "cum se numește", "ce reprezintă"
}


class MultiTaskBERT(nn.Module):
    def __init__(self, model_name, num_labels, num_difficulties):
        super().__init__()
        self.bert        = AutoModel.from_pretrained(model_name)
        hidden_size      = self.bert.config.hidden_size
        self.label_head  = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(hidden_size, 256),
            nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, num_labels)
        )
        self.difficulty_head = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(hidden_size, 256),
            nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, num_difficulties)
        )

    def mean_pooling(self, outputs, attention_mask):
        token_embeddings    = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).float()
        sum_embeddings      = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask            = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled  = self.mean_pooling(outputs, attention_mask)
        return self.label_head(pooled), self.difficulty_head(pooled)


class ModelManager:
    def __init__(self):
        self.loaded_models = []
        self.rag_engine    = None
        self.device        = torch.device("cpu")

    def load_models(self):
        for cfg in MODELS_CONFIG:
            model_path   = BASE_DIR / "models" / "final" / cfg["save_name"]
            config_path  = model_path / "config.json"
            weights_path = model_path / "model_weights.pt"

            if not config_path.exists() or not weights_path.exists():
                print(f"  ⚠️  {cfg['name']} lipsește — omis")
                continue

            try:
                with open(config_path) as f:
                    config = json.load(f)

                num_labels       = config.get("num_labels", 6)
                num_difficulties = config.get("num_difficulties", 3)
                model_name       = config.get("model_name", "")

                tokenizer = AutoTokenizer.from_pretrained(str(model_path))
                model     = MultiTaskBERT(model_name, num_labels, num_difficulties)
                state     = torch.load(str(weights_path), map_location=self.device)
                model.load_state_dict(state)
                model.eval()

                self.loaded_models.append({
                    "name":      cfg["name"],
                    "model":     model,
                    "tokenizer": tokenizer,
                })
                print(f"  ✅ {cfg['name']} încărcat")

            except Exception as e:
                print(f"  ❌ Eroare {cfg['name']}: {e}")

    def load_rag(self):
        try:
            self.rag_engine = RAGEngine()
            print(f"  ✅ RAG Engine încărcat")
        except Exception as e:
            print(f"  ⚠️  RAG indisponibil: {e}")

    def classify(self, text):
        if not self.loaded_models:
            raise RuntimeError("Niciun model disponibil")

        all_label_probs = []
        all_diff_probs  = []

        for entry in self.loaded_models:
            encoding = entry["tokenizer"](
                text,
                max_length=MAX_LENGTH,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            with torch.no_grad():
                label_logits, diff_logits = entry["model"](
                    encoding["input_ids"],
                    encoding["attention_mask"]
                )
            all_label_probs.append(torch.softmax(label_logits, dim=1)[0].tolist())
            all_diff_probs.append(torch.softmax(diff_logits,   dim=1)[0].tolist())

        avg_label = np.mean(all_label_probs, axis=0)
        avg_diff  = np.mean(all_diff_probs,  axis=0)

        label_id = int(np.argmax(avg_label))
        diff_id  = int(np.argmax(avg_diff))

        return {
            "label":       ID_TO_LABEL.get(label_id, "concept_explanation"),
            "difficulty":  ID_TO_DIFF.get(diff_id, "intermediate"),
            "label_conf":  float(avg_label[label_id]),
            "diff_conf":   float(avg_diff[diff_id]),
            "models_used": len(self.loaded_models),
        }


app     = FastAPI(
    title="TinyLogic Chatbot API",
    version="3.0.0"
)
manager = ModelManager()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    print("\n🚀 TinyLogic Chatbot API v3 pornește...")

    print("\n  Încărcăm modelele ensemble...")
    manager.load_models()

    print("\n  Încărcăm RAG Engine...")
    manager.load_rag()

    print("\n  ✅ API gata pe http://localhost:8000\n")


class HistoryMessage(BaseModel):
    role:    str
    content: str


class ChatRequest(BaseModel):
    question:   str
    course:     Optional[str]                  = None
    label:      Optional[str]                  = None
    difficulty: Optional[str]                  = None
    history:    Optional[List[HistoryMessage]] = None


class ChatResponse(BaseModel):
    answer:                str
    label:                 str
    difficulty:            str
    label_confidence:      float
    difficulty_confidence: float
    models_used:           int
    used_course:           bool
    sources:               list
    question:              str
    thinking:              List[str]


class ClassifyResponse(BaseModel):
    label:                 str
    difficulty:            str
    label_confidence:      float
    difficulty_confidence: float
    models_used:           int


def detect_motivation(text, label, label_conf):
    has_keyword    = any(kw in text.lower() for kw in MOTIVATION_KEYWORDS)
    low_conf_short = label_conf < 0.60 and len(text.split()) < 6
    return has_keyword or low_conf_short


def fix_label(text, label):
    if any(kw in text.lower() for kw in CONCEPT_KEYWORDS):
        if label != "concept_explanation":
            return "concept_explanation"
    return label


@app.get("/")
async def root():
    return {
        "message":  "TinyLogic Chatbot API v3.0",
        "models":   [m["name"] for m in manager.loaded_models],
        "rag":      manager.rag_engine is not None,
        "gemini":   manager.rag_engine.gemini == "gemini" if manager.rag_engine else False,
        "features": ["ensemble", "rag", "history", "thinking", "difficulty-adaptive"],
    }


@app.get("/health")
async def health():
    return {
        "status":        "ok",
        "models_loaded": len(manager.loaded_models),
        "model_names":   [m["name"] for m in manager.loaded_models],
        "rag_loaded":    manager.rag_engine is not None,
        "gemini_ok":     manager.rag_engine.gemini == "gemini" if manager.rag_engine else False,
    }


@app.post("/classify", response_model=ClassifyResponse)
async def classify(request: ChatRequest):
    if not manager.loaded_models:
        raise HTTPException(status_code=503, detail="Niciun model disponibil")

    result = manager.classify(request.question)

    if detect_motivation(request.question, result["label"], result["label_conf"]):
        result["label"]      = "motivation_or_encouragement"
        result["difficulty"] = "beginner"
    else:
        result["label"] = fix_label(request.question, result["label"])

    return ClassifyResponse(
        label=result["label"],
        difficulty=result["difficulty"],
        label_confidence=round(result["label_conf"], 4),
        difficulty_confidence=round(result["diff_conf"], 4),
        models_used=result["models_used"],
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Întrebarea nu poate fi goală")

    if not manager.loaded_models:
        raise HTTPException(status_code=503, detail="Niciun model disponibil")

    if request.label and request.difficulty:
        label      = request.label
        difficulty = request.difficulty
        label_conf = 1.0
        diff_conf  = 1.0
        n_models   = 0
    else:
        result     = manager.classify(question)
        label      = result["label"]
        difficulty = result["difficulty"]
        label_conf = result["label_conf"]
        diff_conf  = result["diff_conf"]
        n_models   = result["models_used"]

        if detect_motivation(question, label, label_conf):
            label      = "motivation_or_encouragement"
            difficulty = "beginner"
        else:
            label = fix_label(question, label)

    if request.label:
        label = request.label
    if request.difficulty:
        difficulty = request.difficulty

    history = None
    if request.history:
        history = [
            {"role": msg.role, "content": msg.content}
            for msg in request.history
        ]

    if manager.rag_engine:
        rag_result  = manager.rag_engine.answer(
            question=question,
            label=label,
            difficulty=difficulty,
            course=request.course,
            history=history,
        )
        answer      = rag_result["answer"]
        sources     = rag_result["sources"]
        used_course = rag_result.get("used_course", False)
        thinking    = rag_result.get("thinking", [])
    else:
        answer      = "RAG Engine indisponibil."
        sources     = []
        used_course = False
        thinking    = []

    return ChatResponse(
        answer=answer,
        label=label,
        difficulty=difficulty,
        label_confidence=round(label_conf, 4),
        difficulty_confidence=round(diff_conf, 4),
        models_used=n_models,
        used_course=used_course,
        sources=sources,
        question=question,
        thinking=thinking,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.api.main:app", host="0.0.0.0", port=8000, reload=True)