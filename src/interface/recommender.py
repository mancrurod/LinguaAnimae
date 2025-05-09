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

def load_corpus(book: str, lang: str = "en") -> pd.DataFrame:
    """
    Load a verse dataset for the given book and language.

    Args:
        book (str): File prefix (e.g., "1_genesis")
        lang (str): Language code ("en" or "es")

    Returns:
        pd.DataFrame: Loaded verse dataset with emotion and theme labels
    """
    base_path = Path("data/labeled")
    corpus_dir = base_path / ("bible_kjv" if lang == "en" else "bible_rv60") / "emotion_theme"
    file_path = corpus_dir / f"{book}_emotion_theme.csv"

    if not file_path.exists():
        return pd.DataFrame()

    return pd.read_csv(file_path)


def recommend_verses(
    df: pd.DataFrame,
    emotion: str,
    theme: str,
    lang: str = "en",
    max_results: int = MAX_RESULTS
) -> pd.DataFrame:
    """
    Filter a dataframe of verses by emotion and theme, considering the UI language.

    Args:
        df (pd.DataFrame): Dataframe with annotated verses.
        emotion (str): Detected emotion in English.
        theme (str): Detected theme in English.
        lang (str): Language of the verse corpus ("en" or "es").
        max_results (int): Max number of verses to return.

    Returns:
        pd.DataFrame: Filtered and sampled dataframe with matching verses.
    """
    if lang == "es":
        emotion = EMOTION_MAP.get(emotion.lower(), emotion)
        theme = THEME_MAP.get(theme.lower(), theme)

    df_filtered = df[
        (df["emotion"].str.lower() == emotion.lower()) &
        (df["theme"].str.lower().str.contains(theme.lower()))
    ]

    return df_filtered.sample(n=min(max_results, len(df_filtered)), random_state=42) if not df_filtered.empty else pd.DataFrame()

# ==============================
# === OPTIONAL: BOOK LISTING ===
# ==============================

def list_available_books() -> List[str]:
    """
    List all book prefixes available in the emotion_theme folder.

    Returns:
        List[str]: List of book name prefixes (e.g., ["1_genesis", "2_exodus"])
    """
    return [p.stem.replace("_emotion_theme", "") for p in CORPUS_PATH.glob("*_emotion_theme.csv")]