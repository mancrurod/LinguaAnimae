import pandas as pd
from transformers import pipeline
from pathlib import Path
from typing import List
from datetime import datetime
import argparse
import torch

# ========================
# === CONFIG DEFAULTS ====
# ========================

DEFAULT_MODEL = "j-hartmann/emotion-english-distilroberta-base"
DEFAULT_BIBLE = "bible_kjv"
DEFAULT_LOG_DIR = Path("data/logs")
BATCH_SIZE = 32

# ========================
# === MODEL LOADER =======
# ========================

def load_classifier(model_name: str, device: int = 0):
    device_id = device if torch.cuda.is_available() else -1
    print(f"📦 Loading model: {model_name} (device: {'cuda' if device_id >= 0 else 'cpu'})")
    return pipeline("text-classification", model=model_name, device=device_id, top_k=None)

# ========================
# === EMOTION CLASSIFY ===
# ========================

def classify_batch(texts: List[str], classifier) -> List[str]:
    results = classifier(texts)
    return [max(r, key=lambda x: x["score"])["label"] for r in results]

def process_file(file: Path, classifier, output_dir: Path, summary_lines: List[str], summarize: bool) -> None:
    print(f"🔍 Processing: {file.name}")
    df = pd.read_csv(file)
    if "text" not in df.columns:
        print(f"⚠️ Skipped (missing 'text' column): {file.name}")
        return

    texts = df["text"].astype(str).tolist()
    emotions = []

    for i in range(0, len(texts), BATCH_SIZE):
        emotions.extend(classify_batch(texts[i:i + BATCH_SIZE], classifier))

    df["emotion"] = emotions

    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / file.name.replace("_cleaned.csv", "_emotion.csv")
    df.to_csv(out_file, index=False)
    print(f"✅ Saved: {out_file.name}")

    if summarize:
        counts = df["emotion"].value_counts().to_dict()
        summary = f"{out_file.name} — " + ", ".join(f"{k}: {v}" for k, v in counts.items())
        summary_lines.append(summary)

def run(model_name: str, bible: str, dry_run: Path = None, summarize: bool = True, device: int = 0):
    classifier = load_classifier(model_name, device)
    input_dir = Path("data/processed") / bible
    output_dir = Path("data/labeled") / bible / "emotion"
    log_dir = DEFAULT_LOG_DIR
    summary_lines = []

    files = [dry_run] if dry_run else list(input_dir.glob("*.csv"))
    if not files:
        print("📁 No input files found.")
        return

    for file in files:
        process_file(file, classifier, output_dir, summary_lines, summarize)

    if summarize and summary_lines:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"emotion_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_path.write_text("\n".join(summary_lines), encoding="utf-8")
        print(f"📝 Summary saved to {log_path}")

# ========================
# === CLI ================
# ========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="📘 Emotion Labeling Script")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--bible", type=str, default=DEFAULT_BIBLE)
    parser.add_argument("--dry-run", type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--no-summary", action="store_true")
    args = parser.parse_args()

    run(
        model_name=args.model,
        bible=args.bible,
        dry_run=args.dry_run,
        summarize=not args.no_summary,
        device=args.device
    )
