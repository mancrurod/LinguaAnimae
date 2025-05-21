from pathlib import Path
import pandas as pd
from typing import List, Literal
from src.utils.translation_maps import EMOTION_MAP, THEME_MAP
import streamlit as st
import unidecode

# ==============================
# === CONFIGURATION DEFAULTS ===
# ==============================

CORPUS_PATH = Path("data/labeled/bible_kjv/emotion_theme")
MAX_RESULTS = 5

# ==============================
# === NORMALIZATION UTILS ======
# ==============================

def normalize(s: str) -> str:
    """
    Normalize a string for robust matching (lowercase, no diacritics, trimmed).
    Args:
        s (str): Input string.
    Returns:
        str: Normalized string.
    """
    return unidecode.unidecode(str(s).strip().lower())

# ==============================
# === LOAD CORPUS ==============
# ==============================

@st.cache_data
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


# ==============================
# === RECOMMENDATION LOGIC ==== 
# ==============================

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
        emotion (str): Detected emotion in English or Spanish.
        theme (str): Detected theme in English or Spanish.
        lang (str): Language of the verse corpus ("en" or "es").
        max_results (int): Max number of verses to return.

    Returns:
        pd.DataFrame: Random sample of matching verses.
    """
    # Map emotion and theme if Spanish
    if lang == "es":
        emotion = EMOTION_MAP.get(emotion.lower(), emotion)
        theme = THEME_MAP.get(theme.lower(), theme)

    emotion_norm = normalize(emotion)
    theme_norm = normalize(theme)

    df_filtered = df[
        (df["emotion"].apply(normalize) == emotion_norm) &
        (df["theme"].apply(normalize).str.contains(theme_norm))
    ]

    return (
        df_filtered.sample(n=min(max_results, len(df_filtered)), random_state=42)
        if not df_filtered.empty else pd.DataFrame()
    )

def recommend_verses_by_sections(
    df: pd.DataFrame,
    emotion: str,
    theme: str,
    lang: str = "en"
) -> pd.DataFrame:
    """
    Recommend 2 verses from the Gospels, 2 from the rest of the NT, and 2 from the OT,
    matching the emotion and theme robustly (with normalization).

    Args:
        df (pd.DataFrame): DataFrame with all annotated verses.
        emotion (str): Detected emotion.
        theme (str): Detected theme.
        lang (str): Language ("en" or "es").

    Returns:
        pd.DataFrame: 6 recommended verses (up to 2 from each section).
    """
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

    emotion_norm = normalize(emotion)
    theme_norm = normalize(theme)

    # Normalize 'book' column just once for performance
    df = df.copy()
    df['book_norm'] = df['book'].apply(normalize)

    # Filter verses by emotion and theme (normalized)
    df_filtered = df[
        (df["emotion"].apply(normalize) == emotion_norm) &
        (df["theme"].apply(normalize).str.contains(theme_norm))
    ]

    # Sections
    gospels = df_filtered[df_filtered["book_norm"].isin(GOSPELS)]
    nt_rest = df_filtered[df_filtered["book_norm"].isin(NT_REST)]
    ot = df_filtered[~df_filtered["book_norm"].isin(ALL_NT)]

    # Random selection per section
    sample_gospels = gospels.sample(n=min(2, len(gospels)), random_state=42)
    sample_nt_rest = nt_rest.sample(n=min(2, len(nt_rest)), random_state=42)
    sample_ot = ot.sample(n=min(2, len(ot)), random_state=42)

    # Concatenate and shuffle
    result = pd.concat([sample_gospels, sample_nt_rest, sample_ot], ignore_index=True)
    result = result.sample(frac=1, random_state=42).reset_index(drop=True)

    return result
