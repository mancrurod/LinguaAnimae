"""
This script transfers emotion and theme labels generated from the English text
to the Spanish corpus by matching verses by (chapter, verse) and translating labels.
No inference is performed on the Spanish text.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime

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
    "trust": "Confianza",
    "surprise": "Sorpresa",
    "neutral": "Neutral"
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

def main():
    base = Path("data")
    english_dir = base / "labeled" / "bible_kjv" / "emotion_theme"
    spanish_dir = base / "processed" / "bible_rv60"
    output_dir = base / "labeled" / "bible_rv60" / "emotion_theme"
    log_dir = Path("logs/labeling_logs")

    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    translated_count = 0
    mismatch_count = 0
    log_lines = []

    for file in sorted(english_dir.glob("*_emotion_theme.csv")):
        book_id = file.name.split("_")[0]
        matches = list(spanish_dir.glob(f"{book_id}_*_cleaned.csv"))
        if not matches:
            log_lines.append(f"⚠️ No Spanish file for: {file.name}")
            continue

        df_en = pd.read_csv(file)
        df_es = pd.read_csv(matches[0])

        # Build index for lookup on (chapter, verse)
        en_lookup = {
            (int(row["chapter"]), int(row["verse"])): (translate_theme(row.get("theme")), translate_emotion(row.get("emotion")))
            for _, row in df_en.iterrows()
        }

        themes, emotions = [], []
        for _, row in df_es.iterrows():
            key = (int(row["chapter"]), int(row["verse"]))
            theme, emotion = en_lookup.get(key, ("", ""))
            themes.append(theme)
            emotions.append(emotion)

        df_es["theme"] = themes
        df_es["emotion"] = emotions

        out_name = matches[0].name.replace("_cleaned.csv", "_emotion_theme.csv")
        df_es.to_csv(output_dir / out_name, index=False)
        translated_count += 1
        log_lines.append(f"✅ Translated: {out_name}")

    # Save log
    summary = f"\n--- SUMMARY ---\nTranslated: {translated_count}\nLogged: {len(log_lines)}\n"
    log_lines.append(summary)
    log_path = log_dir / f"translation_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_path.write_text("\n".join(log_lines), encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
