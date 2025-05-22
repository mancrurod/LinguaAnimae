"""
Parses GPT output for emotion labeling in CSV format (id,verse_id,label)
and merges it with the sampled verses,
producing a CSV ready for fine-tuning: id, verse_id, verse, label.
"""

import pandas as pd
from pathlib import Path

# Adjust as necessary
SAMPLES_FILE = Path(__file__).parent.parent.parent / "data" / "evaluation" / "verses_to_label" / "emotion_verses_to_label_6.csv"
GPT_OUTPUT_FILE = Path(__file__).parent.parent.parent / "data" / "evaluation" / "verses_labeled_gpt" / "gpt_output_6.csv"
OUTPUT_FILE = Path(__file__).parent.parent.parent / "data" / "evaluation" / "verses_parsed" / "emotion_verses_labeled_6.csv"

def main():
    print("🔗 Loading sample verses...")
    df_samples = pd.read_csv(SAMPLES_FILE)
    print(f"Sample verses: {len(df_samples)}")
    
    print("🧑‍💻 Loading GPT output as CSV...")
    df_gpt = pd.read_csv(GPT_OUTPUT_FILE, names=["id", "verse_id", "label"])
    print(f"GPT-labeled verses: {len(df_gpt)}")

    # Convert both id and verse_id columns to str for safe merging
    df_samples["id"] = df_samples["id"].astype(str)
    df_gpt["id"] = df_gpt["id"].astype(str)
    df_samples["verse_id"] = df_samples["verse_id"].astype(str)
    df_gpt["verse_id"] = df_gpt["verse_id"].astype(str)

    # Merge on 'id' and 'verse_id' to avoid mismatches
    df_final = pd.merge(df_samples, df_gpt, on=["id", "verse_id"], how="inner")
    print(f"Verses with labels: {len(df_final)}")

    # Reorder and save
    df_final = df_final[['id', 'verse_id', 'verse', 'label']]
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Final labeled dataset saved to {OUTPUT_FILE.absolute()}")

if __name__ == "__main__":
    main()
