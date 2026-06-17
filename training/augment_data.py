

import pandas as pd
import numpy as np
import json
import random
import re
import os
from pathlib import Path
from collections import Counter



BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR      = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
AUGMENTED_DIR = os.path.join(BASE_DIR, "data", "augmented")

SEED          = 42
random.seed(SEED)
np.random.seed(SEED)

NUM_AUG       = 2

ALPHA_SR      = 0.1
ALPHA_RI      = 0.1
ALPHA_RS      = 0.1
ALPHA_RD      = 0.1


SYNONYMS_RO = {

    "înțeleg": ["pricep", "știu", "cunosc"],
    "înțelege": ["pricepe", "ști", "cunoaște"],
    "vreau": ["doresc", "aș vrea", "îmi doresc"],
    "fac": ["realizez", "creez", "construiesc"],
    "face": ["realiza", "crea", "construi"],
    "explică": ["descrie", "prezintă", "arată"],
    "explicați": ["descrieți", "prezentați", "arătați"],
    "ajutor": ["sprijin", "asistență", "suport"],
    "ajutați": ["sprijiniți", "asistați"],
    "știu": ["înțeleg", "cunosc", "pricep"],
    "învăța": ["studia", "înțelege", "asimila"],
    "rezolva": ["soluționa", "descurca", "face"],
    "scrie": ["creez", "construiesc", "codez"],
    "scriu": ["creez", "construiesc", "codez"],
    "exemplu": ["model", "demonstrație", "ilustrație"],
    "cod": ["program", "script", "implementare"],
    "eroare": ["greșeală", "bug", "problemă"],
    "problemă": ["eroare", "dificultate", "obstacol"],
    "funcție": ["metodă", "procedură", "rutină"],
    "variabilă": ["valoare", "parametru"],
    "listă": ["vector", "colecție", "tablou"],
    "cum": ["în ce mod", "cum anume", "de ce fel"],
    "ce": ["care", "ce anume"],
    "când": ["în ce moment", "la ce moment"],
    "unde": ["în ce loc", "în care parte"],
    "corect": ["bine", "corect", "exact"],
    "greșit": ["incorect", "eronat", "greșit"],
    "simplu": ["ușor", "elementar", "facil"],
    "greu": ["dificil", "complex", "complicat"],
    "rapid": ["repede", "prompt", "imediat"],
    "bine": ["corect", "exact", "cum trebuie"],
    "început": ["start", "inițial", "prima parte"],
}

DIFFICULTY_MAP = {0: "beginner", 1: "intermediate", 2: "advanced"}


def synonym_replacement(text, n):

    words  = text.split()
    new_words = words.copy()

    eligible = [(i, w) for i, w in enumerate(words)
                if w.lower() in SYNONYMS_RO]

    random.shuffle(eligible)

    replaced = 0
    for i, word in eligible:
        if replaced >= n:
            break
        synonyms = SYNONYMS_RO[word.lower()]
        new_words[i] = random.choice(synonyms)
        replaced += 1

    return " ".join(new_words)




def random_insertion(text, n):

    words = text.split()
    for _ in range(n):
        eligible = [w for w in words if w.lower() in SYNONYMS_RO]
        if not eligible:
            break
        word    = random.choice(eligible)
        synonym = random.choice(SYNONYMS_RO[word.lower()])
        pos     = random.randint(0, len(words))
        words.insert(pos, synonym)

    return " ".join(words)



def random_swap(text, n):

    words = text.split()
    if len(words) < 2:
        return text

    for _ in range(n):
        i, j = random.sample(range(len(words)), 2)
        words[i], words[j] = words[j], words[i]

    return " ".join(words)



def random_deletion(text, p):

    words = text.split()
    if len(words) == 1:
        return text

    new_words = [w for w in words if random.random() > p]

    # Dacă am șters tot, returnăm un cuvânt aleatoriu
    if not new_words:
        return random.choice(words)

    return " ".join(new_words)




def eda(text, alpha_sr=ALPHA_SR, alpha_ri=ALPHA_RI,
        alpha_rs=ALPHA_RS, alpha_rd=ALPHA_RD, num_aug=NUM_AUG):

    words  = text.split()
    n_sr   = max(1, int(len(words) * alpha_sr))
    n_ri   = max(1, int(len(words) * alpha_ri))
    n_rs   = max(1, int(len(words) * alpha_rs))

    augmented = []
    operations = [
        lambda t: synonym_replacement(t, n_sr),
        lambda t: random_insertion(t, n_ri),
        lambda t: random_swap(t, n_rs),
        lambda t: random_deletion(t, alpha_rd),
    ]

    for _ in range(num_aug):
        op  = random.choice(operations)
        aug = op(text)

        if aug != text and aug not in augmented:
            augmented.append(aug)

    return augmented



def augment_dataset(df, target_per_class=None):

    if target_per_class is None:
        target_per_class = df["difficulty"].value_counts().max()

    print(f"\n Țintă per nivel difficulty: {target_per_class} exemple")

    diff_counts = df["difficulty"].value_counts()
    print("\n   Distribuție ÎNAINTE augmentare:")
    for diff, count in sorted(diff_counts.items()):
        name = DIFFICULTY_MAP[diff]
        bar  = "█" * int(count / 50)
        print(f"   {name:<15} {count:>5} {bar}")

    augmented_rows = []

    for difficulty in sorted(df["difficulty"].unique()):
        subset  = df[df["difficulty"] == difficulty]
        current = len(subset)
        needed  = target_per_class - current

        if needed <= 0:
            print(f"\n   ✅ {DIFFICULTY_MAP[difficulty]}: {current} exemple — nu necesită augmentare")
            continue

        print(f"\n   🔄 {DIFFICULTY_MAP[difficulty]}: {current} exemple → generăm {needed} exemple noi")

        generated = 0
        attempts  = 0
        max_attempts = needed * 10

        while generated < needed and attempts < max_attempts:

            row      = subset.sample(1).iloc[0]
            text     = row["text_clean"] if "text_clean" in row else row["text"]
            variants = eda(text, num_aug=2)

            for variant in variants:
                if generated >= needed:
                    break
                augmented_rows.append({
                    "text_clean":      variant,
                    "label":           row["label"],
                    "label_id":        row["label_id"],
                    "difficulty":      difficulty,
                    "difficulty_name": DIFFICULTY_MAP[difficulty],
                    "augmented":       True,
                    "original_text":   text,
                })
                generated += 1

            attempts += 1

        print(f"      Generat: {generated} exemple noi")

    df_orig = df.copy()
    df_orig["augmented"] = False
    df_orig["original_text"] = ""

    if augmented_rows:
        df_aug = pd.DataFrame(augmented_rows)
        df_final = pd.concat([df_orig, df_aug], ignore_index=True)
    else:
        df_final = df_orig

    diff_counts_after = df_final["difficulty"].value_counts()
    print(f"\n   Distribuție DUPĂ augmentare:")
    for diff, count in sorted(diff_counts_after.items()):
        name = DIFFICULTY_MAP[diff]
        bar  = "█" * int(count / 50)
        print(f"   {name:<15} {count:>5} {bar}")

    return df_final



def print_comparison(df_before, df_after):
    print("\n" + "=" * 55)
    print("📊 COMPARAȚIE ÎNAINTE / DUPĂ AUGMENTARE")
    print("=" * 55)

    print(f"\n   Total exemple:  {len(df_before):>6} → {len(df_after):>6} "
          f"(+{len(df_after) - len(df_before)})")

    print(f"\n   {'Difficulty':<15} {'Înainte':>10} {'După':>10} {'Diferență':>10}")
    print(f"   {'─'*47}")

    for diff in sorted(df_before["difficulty"].unique()):
        name   = DIFFICULTY_MAP[diff]
        before = len(df_before[df_before["difficulty"] == diff])
        after  = len(df_after[df_after["difficulty"] == diff])
        print(f"   {name:<15} {before:>10} {after:>10} {'+'+str(after-before):>10}")

    print(f"\n   {'Label':<30} {'Înainte':>10} {'După':>10}")
    print(f"   {'─'*52}")
    for label in sorted(df_before["label"].unique()):
        before = len(df_before[df_before["label"] == label])
        after  = len(df_after[df_after["label"] == label])
        print(f"   {label:<30} {before:>10} {after:>10}")

    if "augmented" in df_after.columns:
        aug_examples = df_after[df_after["augmented"] == True].head(3)
        if len(aug_examples) > 0:
            print(f"\n   Exemple texte generate (primele 3):")
            print(f"   {'─'*55}")
            for _, row in aug_examples.iterrows():
                print(f"   Original:  {row['original_text'][:60]}")
                print(f"   Augmentat: {row['text_clean'][:60]}")
                print(f"   Label: {row['label']} | Difficulty: {row['difficulty_name']}")
                print()

    print("=" * 55)


def save_augmented(df_final, label_map):
    Path(AUGMENTED_DIR).mkdir(parents=True, exist_ok=True)

    from sklearn.model_selection import train_test_split


    train_val, test = train_test_split(
        df_final, test_size=0.15,
        stratify=df_final["label_id"], random_state=SEED
    )
    train, val = train_test_split(
        train_val, test_size=0.15 / 0.85,
        stratify=train_val["label_id"], random_state=SEED
    )

    cols = ["text_clean", "label", "label_id", "difficulty", "difficulty_name"]

    train[cols].to_csv(f"{AUGMENTED_DIR}/train.csv", index=False)
    val[cols].to_csv(f"{AUGMENTED_DIR}/val.csv",     index=False)
    test[cols].to_csv(f"{AUGMENTED_DIR}/test.csv",   index=False)
    df_final.to_csv(f"{AUGMENTED_DIR}/full_augmented.csv", index=False)

    with open(f"{AUGMENTED_DIR}/label_map.json", "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)

    print(f"\n Salvat în {AUGMENTED_DIR}/")
    print(f"   train.csv           → {len(train)} rânduri")
    print(f"   val.csv             → {len(val)} rânduri")
    print(f"   test.csv            → {len(test)} rânduri")
    print(f"   full_augmented.csv  → {len(df_final)} rânduri")
    print(f"   label_map.json")




if __name__ == "__main__":
    print("\nTinyLogic — Data Augmentation (EDA)\n")
    print("   Referință: Wei & Zou (2019) — EMNLP")
    print("   Tehnici: Synonym Replacement, Random Insertion,")
    print("            Random Swap, Random Deletion\n")

    train_path = os.path.join(PROCESSED_DIR, "train.csv")
    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"Nu găsesc {train_path}\n"
            f"Rulează mai întâi: python training/data_pipeline.py"
        )

    with open(os.path.join(PROCESSED_DIR, "label_map.json"), encoding="utf-8") as f:
        label_map = json.load(f)

    df_train = pd.read_csv(train_path)
    print(f"📂 Train set încărcat: {len(df_train)} exemple\n")

    df_augmented = augment_dataset(df_train)

    print_comparison(df_train, df_augmented)

    save_augmented(df_augmented, label_map)

    print("\n Augmentare completa")
