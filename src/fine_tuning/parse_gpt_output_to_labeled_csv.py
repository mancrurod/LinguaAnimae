"""
Parses GPT output for emotion labeling in CSV format (id,verse_id,label)
and merges it with the sampled verses,
producing a CSV ready for fine-tuning: id, verse_id, verse, label.
"""

import pandas as pd
from pathlib import Path

SAMPLES_FILE = Path(__file__).parent.parent.parent / "data" / "evaluation" / "emotion_verses_to_label_2.csv"
GPT_OUTPUT_FILE = Path(__file__).parent.parent.parent / "data" / "evaluation" / "gpt_output_2.csv"
OUTPUT_FILE = Path(__file__).parent.parent.parent / "data" / "evaluation" / "emotion_verses_labeled_2.csv"

def main():
    print("🔗 Loading sample verses...")
    df_samples = pd.read_csv(SAMPLES_FILE)
    print(f"Sample verses: {len(df_samples)}")
    
    print("🧑‍💻 Loading GPT output as CSV...")
    df_gpt = pd.read_csv(GPT_OUTPUT_FILE, names=["id", "verse_id", "label"])
    print(f"GPT-labeled verses: {len(df_gpt)}")

    # Merge on 'id' and 'verse_id' to avoid mismatches
    df_final = pd.merge(df_samples, df_gpt, on=["id", "verse_id"], how="inner")
    print(f"Verses with labels: {len(df_final)}")

    df_final = df_final[['id', 'verse_id', 'verse', 'label']]
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Final labeled dataset saved to {OUTPUT_FILE.absolute()}")

if __name__ == "__main__":
    main()
