import pandas as pd
from pathlib import Path

# ========================
# === TRANSLATION MAPS ===
# ========================

# Mapping of English themes to their Spanish translations
THEME_MAP = {
    "love": "amor",
    "faith": "fe",
    "hope": "esperanza",
    "forgiveness": "perdón",
    "fear": "miedo"
}

# Mapping of English emotions to their Spanish translations
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

# Function to translate themes from English to Spanish
def translate_themes(theme_str: str) -> str:
    if pd.isna(theme_str):  # Handle NaN values
        return ""
    # Translate each theme in a semicolon-separated list
    return ";".join(THEME_MAP.get(label.strip(), label.strip()) for label in theme_str.split(";"))

# Function to translate a single emotion from English to Spanish
def translate_emotion(emotion: str) -> str:
    # Use the mapping, defaulting to the original if no match is found
    return EMOTION_MAP.get(emotion.strip().lower(), emotion)

# ========================
# === MAIN PROCESS =========
# ========================

# Main function to translate and merge English and Spanish Bible data
def translate_and_merge(bible_en: str = "bible_kjv", bible_es: str = "bible_rv60"):
    # Define base directory paths
    base = Path("data")
    source_dir = base / "labeled" / bible_en / "emotion_theme"  # Source directory for English data
    spanish_dir = base / "processed" / bible_es  # Directory for Spanish data
    output_dir = base / "labeled" / bible_es / "emotion_theme"  # Output directory for translated data
    output_dir.mkdir(parents=True, exist_ok=True)  # Create output directory if it doesn't exist

    # Get all CSV files in the source directory
    files = list(source_dir.glob("*.csv"))
    if not files:  # Check if no files are found
        print("❌ No files found in:", source_dir)
        return

    # Process each file
    for file in files:
        print(f"🔁 Processing {file.name}")
        df_en = pd.read_csv(file)  # Read the English CSV file
        df_es_path = spanish_dir / file.name.replace("_emotion_theme.csv", "_cleaned.csv")  # Corresponding Spanish file path

        if not df_es_path.exists():  # Skip if the Spanish file is missing
            print(f"⚠️ Skipped (missing Spanish file): {df_es_path.name}")
            continue

        df_es = pd.read_csv(df_es_path)  # Read the Spanish CSV file

        if len(df_en) != len(df_es):  # Check for mismatched row counts
            print(f"❌ Mismatch in verse count: {file.name}")
            continue

        # Translate labels
        df_translated = df_es.copy()  # Create a copy of the Spanish DataFrame
        df_translated["emotion"] = df_en["emotion"].apply(translate_emotion)  # Translate emotions
        df_translated["theme"] = df_en["theme"].apply(translate_themes)  # Translate themes

        # Save the translated DataFrame to the output directory
        out_path = output_dir / file.name
        df_translated.to_csv(out_path, index=False)
        print(f"✅ Saved: {out_path.name}")

# ========================
# === CLI RUNNER ==========
# ========================

# Run the script if executed directly
if __name__ == "__main__":
    translate_and_merge()
