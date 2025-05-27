"""
Parses GPT output for emotion labeling in CSV format (id,verse_id,label)
and merges it with the sampled verses,
producing a CSV ready for fine-tuning: id, verse_id, verse, label.
"""

import pandas as pd
from pathlib import Path
import logging

def setup_logger(log_path: Path, level: int = logging.INFO, log_name: str = "parse_gpt_logger"):
    logger = logging.getLogger(log_name)
    logger.setLevel(level)
    if not logger.handlers:
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(fh)
    return logger

# Adjust as necessary
SAMPLES_FILE = Path(__file__).parent.parent.parent / "data" / "evaluation" / "verses_to_label" / "emotion_verses_to_label_6.csv"
GPT_OUTPUT_FILE = Path(__file__).parent.parent.parent / "data" / "evaluation" / "verses_labeled_gpt" / "gpt_output_6.csv"
OUTPUT_FILE = Path(__file__).parent.parent.parent / "data" / "evaluation" / "verses_parsed" / "emotion_verses_labeled_6.csv"
LOG_DIR = Path(__file__).parent.parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "parse_gpt_output.log"
logger = setup_logger(LOG_FILE)

def main(
    samples_file=SAMPLES_FILE,
    gpt_output_file=GPT_OUTPUT_FILE,
    output_file=OUTPUT_FILE,
    logger=logger
):
    logger.info("🔗 Loading sample verses...")
    try:
        df_samples = pd.read_csv(samples_file)
        logger.info(f"Sample verses: {len(df_samples)}")
    except Exception as e:
        logger.error(f"Error loading sample verses file: {e}")
        return
    
    logger.info("🧑‍💻 Loading GPT output as CSV...")
    try:
        df_gpt = pd.read_csv(gpt_output_file, names=["id", "verse_id", "label"])
        logger.info(f"GPT-labeled verses: {len(df_gpt)}")
    except Exception as e:
        logger.error(f"Error loading GPT output file: {e}")
        return

    # Convert both id and verse_id columns to str for safe merging
    df_samples["id"] = df_samples["id"].astype(str)
    df_gpt["id"] = df_gpt["id"].astype(str)
    df_samples["verse_id"] = df_samples["verse_id"].astype(str)
    df_gpt["verse_id"] = df_gpt["verse_id"].astype(str)

    # Merge on 'id' and 'verse_id' to avoid mismatches
    df_final = pd.merge(df_samples, df_gpt, on=["id", "verse_id"], how="inner")
    logger.info(f"Verses with labels: {len(df_final)}")

    if len(df_final) < len(df_samples):
        logger.warning(f"{len(df_samples) - len(df_final)} sample verses had no label and were dropped.")

    # Reorder and save
    df_final = df_final[['id', 'verse_id', 'verse', 'label']]
    output_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        df_final.to_csv(output_file, index=False)
        logger.info(f"✅ Final labeled dataset saved to {output_file.absolute()}")
    except Exception as e:
        logger.error(f"Error saving output file: {e}")


if __name__ == "__main__":
    main()

