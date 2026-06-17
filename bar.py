import pandas as pd

for split, path in [("train", "data/augmented/train.csv"),
                    ("test",  "data/augmented/test.csv"),
                    ("val",   "data/augmented/val.csv")]:
    df = pd.read_csv(path)
    print(f"\n=== {split.upper()} (total: {len(df)}) ===")
    print("Label counts:")
    print(df["label"].value_counts().to_string())
    print("\nDifficulty counts:")
    print(df["difficulty"].value_counts().to_string())