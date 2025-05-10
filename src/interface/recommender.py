from pathlib import Path
import pandas as pd
from typing import List, Literal
from src.utils.translation_maps import EMOTION_MAP, THEME_MAP


# ==============================
# === CONFIGURATION DEFAULTS ===
# ==============================

CORPUS_PATH = Path("data/labeled/bible_kjv/emotion_theme")
MAX_RESULTS = 5


# ==============================
# === RECOMMENDATION LOGIC ==== 
# ==============================

def load_entire_corpus(lang: str = "en") -> pd.DataFrame:
    """
    Load the entire labeled corpus for the selected language.

    Args:
        lang (str): Language code ("en" or "es")

    Returns:
        pd.DataFrame: Combined dataframe with all labeled verses
    """
    base_path = Path("data/labeled")
    corpus_dir = base_path / ("bible_kjv" if lang == "en" else "bible_rv60") / "emotion_theme"

    all_files = list(corpus_dir.glob("*_emotion_theme.csv"))
    if not all_files:
        return pd.DataFrame()

    dfs = []
    for file in all_files:
        try:
            df = pd.read_csv(file)
            df["source_file"] = file.stem  # Optional: to keep track of origin
            dfs.append(df)
        except Exception as e:
            print(f"❌ Error loading {file.name}: {e}")

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()



def recommend_verses(
    df: pd.DataFrame,
    emotion: str,
    theme: str,
    lang: str = "en",
    max_results: int = MAX_RESULTS
) -> pd.DataFrame:
    """
    Recommend random verses from the entire corpus based on emotion and theme.

    Args:
        df (pd.DataFrame): Dataframe with all annotated verses.
        emotion (str): Detected emotion in English.
        theme (str): Detected theme in English.
        lang (str): Language of the verse corpus ("en" or "es").
        max_results (int): Max number of verses to return.

    Returns:
        pd.DataFrame: Random sample of matching verses.
    """
    if lang == "es":
        emotion = EMOTION_MAP.get(emotion.lower(), emotion)
        theme = THEME_MAP.get(theme.lower(), theme)

    df_filtered = df[
        (df["emotion"].str.lower() == emotion.lower()) &
        (df["theme"].str.lower().str.contains(theme.lower()))
    ]

    return (
        df_filtered.sample(n=min(max_results, len(df_filtered)), random_state=42)
        if not df_filtered.empty else pd.DataFrame()
    )