
import pandas as pd
import json
import re
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

import os
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR      = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
SEED          = 42

TRAIN_RATIO   = 0.70
VAL_RATIO     = 0.15
TEST_RATIO    = 0.15

DIFFICULTY_MAP = {
    0: "beginner",
    1: "intermediate",
    2: "advanced"
}



def load_all_csvs(data_dir):
    csv_files = list(Path(data_dir).glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"Nu am găsit fișiere CSV în: {data_dir}")

    print(f"\n📂 Fișiere găsite: {len(csv_files)}")
    dfs = []
    for path in csv_files:
        df = pd.read_csv(path, encoding="utf-8")
        df["source_file"] = path.stem
        dfs.append(df)
        print(f"   ✅ {path.name} → {len(df)} rânduri | label: {df['label'].unique().tolist()}")

    combined = pd.concat(dfs, ignore_index=True)
    print(f"\n   📌 Total rânduri combinate: {len(combined)}")
    return combined




def validate(df):
    print("\n🔍 Validare date...")
    initial = len(df)

    for col in ["text", "label", "difficulty"]:
        if col not in df.columns:
            raise ValueError(f"Coloana '{col}' lipsește din CSV!")

    df = df.dropna(subset=["text", "label", "difficulty"])
    df = df[df["text"].astype(str).str.strip().str.len() >= 5]

    before = len(df)
    df = df.drop_duplicates(subset=["text"])
    print(f"   Duplicate eliminate: {before - len(df)}")

    df["difficulty"] = pd.to_numeric(df["difficulty"], errors="coerce")
    df = df.dropna(subset=["difficulty"])
    df["difficulty"] = df["difficulty"].astype(int)
    df = df[df["difficulty"].isin([0, 1, 2])]

    print(f"   ✅ Rânduri valide: {len(df)} (eliminate: {initial - len(df)})")
    return df.reset_index(drop=True)




def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s\-.,!?;:()\u00C0-\u017E]", "", text)
    text = text.strip()
    return text


def preprocess(df):
    print("\n🧹 Curățare text...")
    df = df.copy()
    df["text_clean"]      = df["text"].apply(clean_text)
    df["label"]           = df["label"].str.lower().str.strip()
    df["difficulty_name"] = df["difficulty"].map(DIFFICULTY_MAP)
    df["word_count"]      = df["text_clean"].str.split().str.len()
    print(f"   ✅ Lungime medie: {df['word_count'].mean():.1f} cuvinte/întrebare")
    return df




def encode(df):
    print("\n🏷️  Encodare label-uri...")
    le = LabelEncoder()
    df["label_id"] = le.fit_transform(df["label"])
    label_map = {int(i): cls for i, cls in enumerate(le.classes_)}
    print(f"   Labels găsite ({len(label_map)}): {list(label_map.values())}")
    return df, le, label_map




def print_stats(df):
    print("\n" + "=" * 50)
    print(" STATISTICI DATE")
    print("=" * 50)
    print(f"\n   Total exemple: {len(df)}")

    print("\n   Distribuție LABEL:")
    for label, count in df["label"].value_counts().items():
        pct = count / len(df) * 100
        bar = "█" * int(pct / 2)
        print(f"   {label:<30} {count:>5} ({pct:5.1f}%) {bar}")

    print("\n   Distribuție DIFFICULTY:")
    for diff, count in df["difficulty_name"].value_counts().sort_index().items():
        pct = count / len(df) * 100
        bar = "█" * int(pct / 2)
        print(f"   {diff:<15} {count:>5} ({pct:5.1f}%) {bar}")

    print("\n   Distribuție LABEL × DIFFICULTY:")
    print(pd.crosstab(df["label"], df["difficulty_name"]).to_string())
    print("=" * 50)




def split(df):
    print("\n Split date...")
    train_val, test = train_test_split(
        df, test_size=TEST_RATIO,
        stratify=df["label_id"], random_state=SEED
    )
    val_size = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    train, val = train_test_split(
        train_val, test_size=val_size,
        stratify=train_val["label_id"], random_state=SEED
    )
    print(f"   Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    return train, val, test



def save(train, val, test, label_map):
    Path(PROCESSED_DIR).mkdir(parents=True, exist_ok=True)
    cols = ["text_clean", "label", "label_id", "difficulty", "difficulty_name"]

    train[cols].to_csv(f"{PROCESSED_DIR}/train.csv", index=False)
    val[cols].to_csv(f"{PROCESSED_DIR}/val.csv",     index=False)
    test[cols].to_csv(f"{PROCESSED_DIR}/test.csv",   index=False)

    with open(f"{PROCESSED_DIR}/label_map.json", "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)

    print(f"\n Salvat în {PROCESSED_DIR}/")
    print(f"   train.csv  → {len(train)} rânduri")
    print(f"   val.csv    → {len(val)} rânduri")
    print(f"   test.csv   → {len(test)} rânduri")
    print(f"   label_map.json")



if __name__ == "__main__":
    print("\nTinyLogic — Data Pipeline\n")

    df = load_all_csvs(DATA_DIR)
    df = validate(df)
    df = preprocess(df)
    df, le, label_map = encode(df)

    print_stats(df)

    train, val, test = split(df)
    save(train, val, test, label_map)

    print("\n✅ Pipeline complet! Datele sunt gata pentru antrenament.\n")