"""
This script transfers emotion and theme labels generated from the English text
to the Spanish corpus by matching verses by (chapter, verse) and translating labels.
No inference is performed on the Spanish text.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
import logging

def setup_logger(
    log_path: Path,
    level: int = logging.INFO,
    console: bool = True,
    log_name: str = "label_transfer_logger"
) -> logging.Logger:
    """
    Set up a logger that logs to both a file and optionally the console.

    Args:
        log_path (Path): Path to the log file.
        level (int): Logging level (e.g., logging.INFO, logging.ERROR).
        console (bool): Whether to log also to the console.
        log_name (str): Name of the logger instance.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(log_name)
    logger.setLevel(level)
    if not logger.handlers:
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(level)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s")
        fh.setFormatter(file_formatter)
        logger.addHandler(fh)

        if console:
            ch = logging.StreamHandler()
            ch.setLevel(level)
            console_formatter = logging.Formatter(
                "%(levelname)s: %(message)s")
            ch.setFormatter(console_formatter)
            logger.addHandler(ch)
    return logger


# ========================
# === TRANSLATION MAPS ===
# ========================

THEME_MAP = {
    "love": "amor",
    "faith": "fe",
    "hope": "esperanza",
    "forgiveness": "perdón",
    "fear": "miedo"
}

EMOTION_MAP = {
    "joy": "Alegría",
    "sadness": "Tristeza",
    "anger": "Ira",
    "fear": "Miedo",
    "surprise": "Sorpresa",
    "neutral": "Neutral",
    "disgust": "Asco"
}

# ==========================
# === TRANSLATION HELPERS ==
# ==========================

def translate_theme(theme: str) -> str:
    if pd.isna(theme):
        return ""
    return ";".join([THEME_MAP.get(t.strip().lower(), t.strip()) for t in theme.split(";")])

def translate_emotion(emotion: str) -> str:
    if pd.isna(emotion):
        return ""
    return EMOTION_MAP.get(emotion.strip().lower(), emotion.strip())

# ========================
# === MAIN SCRIPT ========
# ========================

def main(logger: logging.Logger):
    base = Path("data")
    english_dir = base / "labeled" / "bible_kjv" / "emotion_theme"
    spanish_dir = base / "processed" / "bible_rv60"
    output_dir = base / "labeled" / "bible_rv60" / "emotion_theme"
    log_dir = Path("logs/labeling_logs")

    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    translated_count = 0
    mismatch_count = 0
    total_unmatched_verses = 0
    log_lines = []

    for file in sorted(english_dir.glob("*_emotion_theme.csv")):
        book_id = file.name.split("_")[0]
        matches = list(spanish_dir.glob(f"{book_id}_*_cleaned.csv"))

        if not matches:
            logger.warning(f"No Spanish file for: {file.name}")
            log_lines.append(f"⚠️ No Spanish file for: {file.name}")
            continue
        if len(matches) > 1:
            logger.warning(f"Multiple Spanish files found for {book_id}: {[m.name for m in matches]}")
            log_lines.append(f"⚠️ Multiple Spanish files found for {book_id}: {[m.name for m in matches]}")

        try:
            df_en = pd.read_csv(file)
        except Exception as e:
            logger.error(f"Error reading English file {file.name}: {e}")
            log_lines.append(f"❌ Error reading English file {file.name}: {e}")
            continue
        try:
            df_es = pd.read_csv(matches[0])
        except Exception as e:
            logger.error(f"Error reading Spanish file {matches[0].name}: {e}")
            log_lines.append(f"❌ Error reading Spanish file {matches[0].name}: {e}")
            continue

        # Build index for lookup on (chapter, verse)
        en_lookup = {}
        duplicate_keys = set()
        for _, row in df_en.iterrows():
            key = (int(row["chapter"]), int(row["verse"]))
            if key in en_lookup:
                duplicate_keys.add(key)
            en_lookup[key] = (
                translate_theme(row.get("theme")),
                translate_emotion(row.get("emotion"))
            )
        if duplicate_keys:
            logger.warning(f"Duplicate (chapter, verse) keys found in {file.name}: {duplicate_keys}")
            log_lines.append(f"⚠️ Duplicate keys in {file.name}: {duplicate_keys}")

        themes, emotions = [], []
        unmatched_verses = 0
        for _, row in df_es.iterrows():
            try:
                key = (int(row["chapter"]), int(row["verse"]))
            except Exception as e:
                logger.warning(f"Invalid chapter/verse in Spanish file {matches[0].name}: {e}")
                key = None
            if key and key in en_lookup:
                theme, emotion = en_lookup[key]
            else:
                theme, emotion = "", ""
                unmatched_verses += 1
            themes.append(theme)
            emotions.append(emotion)

        df_es["theme"] = themes
        df_es["emotion"] = emotions

        out_name = matches[0].name.replace("_cleaned.csv", "_emotion_theme.csv")
        try:
            df_es.to_csv(output_dir / out_name, index=False, encoding="utf-8", lineterminator="\n")
            logger.info(f"Translated: {out_name} (unmatched verses: {unmatched_verses})")
            log_lines.append(f"✅ Translated: {out_name} (unmatched verses: {unmatched_verses})")
            translated_count += 1
            total_unmatched_verses += unmatched_verses
        except Exception as e:
            logger.error(f"Error saving file {out_name}: {e}")
            log_lines.append(f"❌ Error saving file {out_name}: {e}")

    # Save log
    summary = (
        f"\n--- SUMMARY ---\n"
        f"Translated files: {translated_count}\n"
        f"Total unmatched verses: {total_unmatched_verses}\n"
        f"Log lines: {len(log_lines)}\n"
    )
    log_lines.append(summary)
    log_path = log_dir / f"translation_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_path.write_text("\n".join(log_lines), encoding="utf-8")
    logger.info(summary)


if __name__ == "__main__":
    LOG_DIR = Path("logs/labeling_logs")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"label_transfer_{timestamp}.log"
    logger = setup_logger(log_path, level=logging.INFO, console=True)
    main(logger)

