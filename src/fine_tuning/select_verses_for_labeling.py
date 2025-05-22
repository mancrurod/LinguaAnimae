"""
Selects 1000 random unique verses from all CSVs in data/processed/bible_kjv/
and saves them as emotion_verses_to_label.csv for annotation, including verse_id.

Columns: id, verse_id, verse
"""

import pandas as pd
from pathlib import Path

# === Paths ===
INPUT_DIR = Path(__file__).parent.parent.parent / "data" / "processed" / "bible_kjv"
OUTPUT_FILE = Path(__file__).parent.parent.parent / "data" / "evaluation" / "verses_to_label" / "emotion_verses_to_label_4.csv" # Change as needed
N_SAMPLES = 1000

def main():
    print("🔎 Searching for CSV files...")
    all_csvs = list(INPUT_DIR.glob("*.csv"))
    print(f"Found {len(all_csvs)} files.")

    records = []
    for csv in all_csvs:
        df = pd.read_csv(csv)
        if 'verse_id' in df.columns and 'text' in df.columns:
            temp = df[['verse_id', 'text']].dropna()
            temp = temp.rename(columns={'text': 'verse'})
            records.append(temp)
        else:
            print(f"Skipping {csv.name}: missing 'verse_id' or 'text'")
    
    if not records:
        print("❌ No suitable data found!")
        return
    
    df_all = pd.concat(records, ignore_index=True)
    print(f"Total verses loaded: {len(df_all)}")

    # Remove duplicates by verse_id (safer than by text only)
    df_all = df_all.drop_duplicates(subset=['verse_id'])
    print(f"Unique verses by verse_id: {len(df_all)}")

    # (Opcional) Exclude already labeled verses if file exists
    EXISTING_LABELS = Path(__file__).parent.parent.parent / "data" / "evaluation" / "emotion_verses_labeled_combined.csv"
    if EXISTING_LABELS.exists():
        df_labeled = pd.read_csv(EXISTING_LABELS)
        labeled_ids = set(df_labeled['verse_id'])
        before = len(df_all)
        df_all = df_all[~df_all['verse_id'].isin(labeled_ids)]
        print(f"Filtered out {before - len(df_all)} verses already labeled. Now {len(df_all)} left.")


    # Sample randomly
    n = min(N_SAMPLES, len(df_all))
    df_sample = df_all.sample(n=n, random_state=22).reset_index(drop=True)

    # Add 'id' as incremental integer
    df_sample.insert(0, 'id', range(n))

    # Output desired columns and order
    df_sample = df_sample[['id', 'verse_id', 'verse']]
    # Crea carpeta de salida si no existe
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_sample.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Sample of {n} verses saved to {OUTPUT_FILE.absolute()}")

if __name__ == "__main__":
    main()
