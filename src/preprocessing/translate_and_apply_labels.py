import pandas as pd
from pathlib import Path

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
    "surprise": "Sorpresa"
}

# ========================
# === TRANSLATION LOGIC ==
# ========================

def translate_themes(theme_str: str) -> str:
    if pd.isna(theme_str):
        return ""
    return ";".join(THEME_MAP.get(label.strip(), label.strip()) for label in theme_str.split(";"))

def translate_emotion(emotion: str) -> str:
    return EMOTION_MAP.get(emotion.strip().lower(), emotion)

# ========================
# === MAIN PROCESS =========
# ========================

def translate_and_merge(bible_en: str = "bible_kjv", bible_es: str = "bible_rv60"):
    base = Path("data")
    source_dir = base / "labeled" / bible_en / "emotion_theme"
    spanish_dir = base / "processed" / bible_es
    output_dir = base / "labeled" / bible_es / "emotion_theme"
    output_dir.mkdir(parents=True, exist_ok=True)

    files = list(source_dir.glob("*.csv"))
    if not files:
        print("❌ No files found in:", source_dir)
        return

    for file in files:
        print(f"🔁 Processing {file.name}")
        df_en = pd.read_csv(file)
        df_es_path = spanish_dir / file.name.replace("_emotion_theme.csv", "_cleaned.csv")

        if not df_es_path.exists():
            print(f"⚠️ Skipped (missing Spanish file): {df_es_path.name}")
            continue

        df_es = pd.read_csv(df_es_path)

        if len(df_en) != len(df_es):
            print(f"❌ Mismatch in verse count: {file.name}")
            continue

        # Translate labels
        df_translated = df_es.copy()
        df_translated["emotion"] = df_en["emotion"].apply(translate_emotion)
        df_translated["theme"] = df_en["theme"].apply(translate_themes)

        # Save
        out_path = output_dir / file.name
        df_translated.to_csv(out_path, index=False)
        print(f"✅ Saved: {out_path.name}")

# ========================
# === CLI RUNNER ==========
# ========================

if __name__ == "__main__":
    translate_and_merge()
