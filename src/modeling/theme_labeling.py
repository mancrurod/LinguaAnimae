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

DEFAULT_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
DEFAULT_BIBLE = "bible_kjv"
DEFAULT_LABELS = ["love", "faith", "hope", "forgiveness", "fear"]
DEFAULT_THRESHOLD = 0.7
DEFAULT_LOG_DIR = Path("data/logs")
BATCH_SIZE = 16

# ========================
# === MODEL LOADER =======
# ========================

def load_classifier(model_name: str, device: int = 0):
    device_id = device if torch.cuda.is_available() else -1
    print(f"📦 Loading model: {model_name} (device: {'cuda' if device_id >= 0 else 'cpu'})")
    return pipeline("zero-shot-classification", model=model_name, device=device_id)

# ========================
# === THEME CLASSIFY =====
# ========================

def classify_batch(texts: List[str], classifier, labels: List[str], threshold: float) -> List[str]:
    results = classifier(texts, candidate_labels=labels, multi_label=True)
    if not isinstance(results, list): results = [results]
    return [
        ";".join([l for l, s in zip(r["labels"], r["scores"]) if s >= threshold])
        for r in results
    ]

def process_file(file: Path, classifier, output_dir: Path, labels: List[str],
                 threshold: float, summary_lines: List[str], summarize: bool) -> None:
    print(f"🔍 Processing: {file.name}")
    df = pd.read_csv(file)
    if "text" not in df.columns or "emotion" not in df.columns:
        print(f"⚠️ Skipped (missing 'text' or 'emotion'): {file.name}")
        return

    texts = df["text"].astype(str).tolist()
    themes = []

    for i in range(0, len(texts), BATCH_SIZE):
        themes.extend(classify_batch(texts[i:i + BATCH_SIZE], classifier, labels, threshold))

    df["theme"] = themes
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / file.name.replace("_emotion.csv", "_emotion_theme.csv")
    df.to_csv(out_file, index=False)
    print(f"✅ Saved: {out_file.name}")

    if summarize:
        all_themes = df["theme"].dropna().str.split(";").explode()
        counts = all_themes.value_counts().to_dict()
        summary = f"{out_file.name} — " + ", ".join(f"{k}: {v}" for k, v in counts.items())
        summary_lines.append(summary)

def run(model_name: str, bible: str, labels: List[str], threshold: float,
        dry_run: Path = None, summarize: bool = True, device: int = 0):
    classifier = load_classifier(model_name, device)
    input_dir = Path("data/labeled") / bible / "emotion"
    output_dir = Path("data/labeled") / bible / "emotion_theme"
    log_dir = DEFAULT_LOG_DIR
    summary_lines = []

    files = [dry_run] if dry_run else list(input_dir.glob("*_emotion.csv"))
    if not files:
        print("📁 No emotion-labeled files found.")
        return

    for file in files:
        result_path = output_dir / file.name.replace("_emotion.csv", "_emotion_theme.csv")
        if result_path.exists():
            print(f"⏩ Skipped (already labeled): {file.name}")
            continue
        process_file(file, classifier, output_dir, labels, threshold, summary_lines, summarize)

    if summarize and summary_lines:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"theme_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_path.write_text("\n".join(summary_lines), encoding="utf-8")
        print(f"📝 Summary saved to {log_path}")

# ========================
# === CLI INTERFACE ======
# ========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="📘 Theme Labeling Script")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--bible", type=str, default=DEFAULT_BIBLE)
    parser.add_argument("--labels", nargs="+", default=DEFAULT_LABELS)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--dry-run", type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--no-summary", action="store_true")
    args = parser.parse_args()

    run(
        model_name=args.model,
        bible=args.bible,
        labels=args.labels,
        threshold=args.threshold,
        dry_run=args.dry_run,
        summarize=not args.no_summary,
        device=args.device
    )
