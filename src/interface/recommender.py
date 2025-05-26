"""
Recommender module for Bible verses labeled by emotion and theme.

- Loads the labeled corpus for English or Spanish.
- Recommends verses by emotion and theme, with options for random sampling or by biblical sections.
- Includes robust normalization and mapping between English and Spanish labels.
- Logging is included for traceability and debugging.

Designed for integration with Streamlit and use in production environments.
"""

from pathlib import Path
import pandas as pd
from typing import List, Literal
from src.utils.translation_maps import EMOTION_MAP, THEME_MAP
import streamlit as st
import unidecode
import logging

# ==============================
# === LOGGER SETUP ============
# ==============================

def setup_logger(
    log_path: Path,
    level: int = logging.INFO,
    console: bool = False,
    log_name: str = "recommender_logger"
) -> logging.Logger:
    """
    Set up a logger that logs to both a file and optionally the console.
    """
    logger = logging.getLogger(log_name)
    logger.setLevel(level)
    if not logger.handlers:
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(fh)
        if console:
            ch = logging.StreamHandler()
            ch.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
            logger.addHandler(ch)
    return logger

# Create log directory and logger instance (adjust path as needed)
LOG_DIR = Path("logs/recommender_logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOGGER = setup_logger(LOG_DIR / "recommender.log", level=logging.INFO)

# ==============================
# === CONFIGURATION DEFAULTS ===
# ==============================

CORPUS_PATH = Path("data/labeled/bible_kjv/emotion_theme")
MAX_RESULTS = 5

# ==============================
# === NORMALIZATION UTILS ======
# ==============================

def normalize(s: str, logger: logging.Logger = None) -> str:
    """
    Normalize a string for robust matching (lowercase, no diacritics, trimmed).
    Args:
        s (str): Input string.
        logger (logging.Logger, optional): Logger for any normalization issue.
    Returns:
        str: Normalized string.
    """
    try:
        return unidecode.unidecode(str(s).strip().lower())
    except Exception as e:
        if logger:
            logger.error(f"Normalization error for input: {s} — {e}")
        return ""

# ==============================
# === LOAD CORPUS ==============
# ==============================

@st.cache_data
def load_entire_corpus(lang: str = "en", logger: logging.Logger = LOGGER) -> pd.DataFrame:
    """
    Load the entire labeled corpus for the selected language.

    Args:
        lang (str): Language code ("en" or "es")
        logger (logging.Logger): Logger for error reporting.

    Returns:
        pd.DataFrame: Combined dataframe with all labeled verses
    """
    base_path = Path("data/labeled")
    corpus_dir = base_path / ("bible_kjv" if lang == "en" else "bible_rv60") / "emotion_theme"

    all_files = list(corpus_dir.glob("*_emotion_theme.csv"))
    if not all_files:
        logger.warning(f"No files found in {corpus_dir}")
        st.warning(f"No files found for language '{lang}' in {corpus_dir}")
        return pd.DataFrame()

    dfs = []
    for file in all_files:
        try:
            df = pd.read_csv(file)
            df["source_file"] = file.stem  # Optional: to keep track of origin
            dfs.append(df)
        except Exception as e:
            logger.error(f"Error loading {file.name}: {e}")
            st.warning(f"Error loading file {file.name}: {e}")

    if not dfs:
        logger.error("No valid files loaded for corpus.")
        st.warning("No valid corpus files could be loaded.")
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"Corpus loaded for language '{lang}': {len(combined)} verses from {len(dfs)} files.")
    return combined


# ==============================
# === RECOMMENDATION LOGIC ==== 
# ==============================

def recommend_verses(
    df: pd.DataFrame,
    emotion: str,
    theme: str,
    lang: str = "en",
    max_results: int = MAX_RESULTS,
    logger: logging.Logger = LOGGER
) -> pd.DataFrame:
    """
    Recommend random verses from the entire corpus based on emotion and theme.

    Args:
        df (pd.DataFrame): Dataframe with all annotated verses.
        emotion (str): Detected emotion in English or Spanish.
        theme (str): Detected theme in English or Spanish.
        lang (str): Language of the verse corpus ("en" or "es").
        max_results (int): Max number of verses to return.
        logger (logging.Logger): Logger for reporting.

    Returns:
        pd.DataFrame: Random sample of matching verses.
    """
    if df.empty:
        logger.warning("Attempted to recommend verses from an empty DataFrame.")
        st.warning("No verses to recommend: corpus is empty.")
        return pd.DataFrame()

    # Map emotion and theme if Spanish
    if lang == "es":
        emotion = EMOTION_MAP.get(emotion.lower(), emotion)
        theme = THEME_MAP.get(theme.lower(), theme)

    emotion_norm = normalize(emotion, logger=logger)
    theme_norm = normalize(theme, logger=logger)

    df_filtered = df[
        (df["emotion"].apply(lambda x: normalize(x, logger=logger)) == emotion_norm) &
        (df["theme"].apply(lambda x: normalize(x, logger=logger)).str.contains(theme_norm))
    ]

    if df_filtered.empty:
        logger.info(f"No verses found for emotion '{emotion}' and theme '{theme}' in language '{lang}'.")
        st.info("No verses found for your selected emotion and theme.")
        return pd.DataFrame()

    try:
        sample = df_filtered.sample(n=min(max_results, len(df_filtered)))
        logger.info(
            f"Recommended {len(sample)} verse(s) for emotion='{emotion}', theme='{theme}', lang='{lang}'."
        )
        return sample
    except Exception as e:
        logger.error(f"Error sampling verses: {e}")
        st.warning("Error generating recommendations. Please try again.")
        return pd.DataFrame()

def recommend_verses_by_sections(
    df: pd.DataFrame,
    emotion: str,
    theme: str,
    lang: str = "en",
    logger: logging.Logger = LOGGER
) -> pd.DataFrame:
    """
    Recommend 2 verses from the Gospels, 2 from the rest of the NT, and 2 from the OT,
    matching the emotion and theme robustly (with normalization and logging).

    Args:
        df (pd.DataFrame): DataFrame with all annotated verses.
        emotion (str): Detected emotion.
        theme (str): Detected theme.
        lang (str): Language ("en" or "es").
        logger (logging.Logger): Logger for reporting.

    Returns:
        pd.DataFrame: 6 recommended verses (up to 2 from each section).
    """
    if df.empty:
        logger.warning("Attempted to recommend verses by sections from an empty DataFrame.")
        st.warning("No verses to recommend by sections: corpus is empty.")
        return pd.DataFrame()

    # Define book groups using normalized names
    if lang == "en":
        GOSPELS = ['matthew', 'mark', 'luke', 'john']
        NT_REST = [
            'acts', 'romans', '1_corinthians', '2_corinthians', 'galatians', 'ephesians', 'philippians',
            'colossians', '1_thessalonians', '2_thessalonians', '1_timothy', '2_timothy', 'titus',
            'philemon', 'hebrews', 'james', '1_peter', '2_peter', '1_john', '2_john', '3_john', 'jude', 'revelation'
        ]
    else:  # lang == "es"
        GOSPELS = ['mateo', 'marcos', 'lucas', 'juan']
        NT_REST = [
            'hechos', 'romanos', '1_corintios', '2_corintios', 'galatas', 'efesios', 'filipenses',
            'colosenses', '1_tesalonicenses', '2_tesalonicenses', '1_timoteo', '2_timoteo', 'tito',
            'filemon', 'hebreos', 'santiago', '1_pedro', '2_pedro', '1_juan', '2_juan', '3_juan', 'judas', 'apocalipsis'
        ]

    ALL_NT = GOSPELS + NT_REST

    # Map emotion and theme if Spanish
    if lang == "es":
        emotion = EMOTION_MAP.get(emotion.lower(), emotion)
        theme = THEME_MAP.get(theme.lower(), theme)

    emotion_norm = normalize(emotion, logger=logger)
    theme_norm = normalize(theme, logger=logger)

    df = df.copy()
    df['book_norm'] = df['book'].apply(lambda x: normalize(x, logger=logger))

    # Filter verses by emotion and theme (normalized)
    df_filtered = df[
        (df["emotion"].apply(lambda x: normalize(x, logger=logger)) == emotion_norm) &
        (df["theme"].apply(lambda x: normalize(x, logger=logger)).str.contains(theme_norm))
    ]

    # Sections
    gospels = df_filtered[df_filtered["book_norm"].isin(GOSPELS)]
    nt_rest = df_filtered[df_filtered["book_norm"].isin(NT_REST)]
    ot = df_filtered[~df_filtered["book_norm"].isin(ALL_NT)]

    # Helper for safe sampling with logging
    def safe_sample(section_df, n, section_name):
        if section_df.empty:
            logger.info(f"No verses found in section '{section_name}'.")
            return section_df
        size = min(n, len(section_df))
        try:
            sample = section_df.sample(n=size)
            if size < n:
                logger.warning(f"Requested {n} from '{section_name}', only {size} available.")
            return sample
        except Exception as e:
            logger.error(f"Error sampling from '{section_name}': {e}")
            return section_df.head(size)

    sample_gospels = safe_sample(gospels, 2, "Gospels")
    sample_nt_rest = safe_sample(nt_rest, 2, "NT Rest")
    sample_ot = safe_sample(ot, 2, "OT")

    result = pd.concat([sample_gospels, sample_nt_rest, sample_ot], ignore_index=True)
    if not result.empty:
        result = result.sample(frac=1, random_state=42).reset_index(drop=True)
        logger.info(f"Recommended {len(result)} verse(s) by section for emotion='{emotion}', theme='{theme}', lang='{lang}'.")
    else:
        logger.info(f"No verses found in any section for emotion='{emotion}' and theme='{theme}'.")

    return result