"""
Selects all verses labeled as 'trust' from all CSVs in data/processed/bible_kjv/,
removes duplicates, and saves them for annotation.
Columns: id, verse_id, verse
"""

import pandas as pd
from pathlib import Path

# === Params ===
INPUT_DIR = Path(__file__).parent.parent.parent / "data" / "labeled" / "bible_kjv" / "emotion"
OUTPUT_FILE = Path(__file__).parent.parent.parent / "data" / "evaluation" / "emotion_verses_trust_only.csv"

def main():
    print("🔎 Searching for CSV files...")
    all_csvs = list(INPUT_DIR.glob("*.csv"))
    print(f"Found {len(all_csvs)} files.")

    records = []
    for csv in all_csvs:
        df = pd.read_csv(csv)
        if {'verse_id', 'text', 'emotion'}.issubset(df.columns):
            temp = df[['verse_id', 'text', 'emotion']].dropna()
            temp = temp.rename(columns={'text': 'verse'})
            records.append(temp)
        else:
            print(f"Skipping {csv.name}: missing 'verse_id', 'text', or 'emotion'")

    if not records:
        print("❌ No suitable data found!")
        return

    df_all = pd.concat(records, ignore_index=True)
    print(f"Total verses loaded: {len(df_all)}")

    # Remove duplicates by verse_id
    df_all = df_all.drop_duplicates(subset=['verse_id'])
    print(f"Unique verses by verse_id: {len(df_all)}")

    # Select only 'trust' verses
    df_trust = df_all[df_all['emotion'].str.strip().str.lower() == "trust"]
    n_trust = len(df_trust)
    print(f"Verses labeled as 'trust': {n_trust}")

    # Reset id column
    df_trust = df_trust.reset_index(drop=True)
    df_trust.insert(0, 'id', range(len(df_trust)))

    # Output columns
    df_gpt = df_trust[['id', 'verse_id', 'verse']].copy()

    # Save the file
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_gpt.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Saved CSV for 'trust' verses to {OUTPUT_FILE.absolute()}")

if __name__ == "__main__":
    main()
