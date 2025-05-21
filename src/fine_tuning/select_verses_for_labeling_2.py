"""
Selects N random verses from all CSVs in data/processed/bible_kjv/, 
prioritizing those labeled as trust, fear, sadness, or surprise.
If insufficient, fills with joy, anger, neutral. 
Includes id, verse_id, verse, and emotion.
"""

import pandas as pd
from pathlib import Path

# === Params ===
INPUT_DIR = Path(__file__).parent.parent.parent / "data" / "labeled" / "bible_kjv" / "emotion"
OUTPUT_FILE = Path(__file__).parent.parent.parent / "data" / "evaluation" / "emotion_verses_priority_sample.csv"
N_SAMPLES = 1000

# Emotions to prioritize and deprioritize
PRIORITY_EMOTIONS = ["trust", "fear", "sadness", "surprise"]
NON_PRIORITY_EMOTIONS = ["joy", "neutral", "anger"]

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

    # Split by priority
    df_priority = df_all[df_all['emotion'].str.lower().isin(PRIORITY_EMOTIONS)]
    df_non_priority = df_all[df_all['emotion'].str.lower().isin(NON_PRIORITY_EMOTIONS)]

    # Sample as many as possible from priority emotions
    n_priority = min(len(df_priority), N_SAMPLES)
    df_sample_priority = df_priority.sample(n=n_priority, random_state=42) if n_priority > 0 else pd.DataFrame()

    # Fill the rest from non-priority
    n_remaining = N_SAMPLES - n_priority
    if n_remaining > 0:
        n_non_priority = min(len(df_non_priority), n_remaining)
        df_sample_non_priority = df_non_priority.sample(n=n_non_priority, random_state=99) if n_non_priority > 0 else pd.DataFrame()
    else:
        df_sample_non_priority = pd.DataFrame()

    # Concatenate and shuffle final sample
    df_final = pd.concat([df_sample_priority, df_sample_non_priority], ignore_index=True)
    df_final = df_final.sample(frac=1, random_state=123).reset_index(drop=True)

    # Add incremental id
    df_final.insert(0, 'id', range(len(df_final)))

    # Elimina la columna 'emotion' para la versión que vas a pasar a GPT
    df_gpt = df_final[['id', 'verse_id', 'verse']].copy()
    # Si también quieres quitar la columna 'verse' (y dejar solo id,verse_id para el prompt de GPT), haz:
    # df_gpt = df_final[['id', 'verse_id']].copy()

    # Guarda el archivo CSV listo para GPT
    GPT_FILE = Path(__file__).parent.parent.parent / "data" / "evaluation" / "emotion_verses_to_gpt.csv"
    GPT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_gpt.to_csv(GPT_FILE, index=False)
    print(f"✅ Saved CSV for GPT labeling to {GPT_FILE.absolute()}")

    # (Opcional: también puedes guardar el CSV con la columna 'emotion' para tu control)
    # df_final.to_csv(OUTPUT_FILE, index=False)

if __name__ == "__main__":
    main()
