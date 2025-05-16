import sys
from pathlib import Path

# Ensure src/ is in the path for module resolution
sys.path.append(str(Path(__file__).resolve().parent.parent))

import streamlit as st
import base64
import pandas as pd
from deep_translator import GoogleTranslator
from transformers import pipeline

from texts import TEXTS
from components.render_emotion import render_emotion_block
from components.render_theme import render_theme_block
from components.render_feedback import render_feedback_section
from src.interface.recommender import load_entire_corpus, recommend_verses
from src.utils.save_feedback_to_gsheet import save_feedback_to_gsheet
from src.utils.translation_maps import BOOK_NAME_MAP_ES


# === Streamlit config ===
st.set_page_config(page_title="Lingua Animae", page_icon="📖")

def get_base64_bg(image_path: str) -> str:
    """
    Encode an image in base64 to be used as background.

    Args:
        image_path (str): Path to the image file.

    Returns:
        str: Base64-encoded string.
    """
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def set_background(image_path: str) -> None:
    """
    Set a background image for the app and apply global CSS styling.

    Args:
        image_path (str): Path to the image file.
    """
    try:
        bg_base64 = get_base64_bg(image_path)
        st.markdown(f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;600&family=Merriweather:wght@300;400&display=swap');

            html, body, .stApp, [class^="css"], [class*="st-"] {{
                font-family: 'Cormorant Garamond', serif !important;
                font-size: 22px;
                color: #5d4037;
                background-color: transparent;
                text-shadow: 1px 1px 1px rgba(0,0,0,0.1);
            }}

            .stApp {{
                background-image: url("data:image/jpg;base64,{bg_base64}");
                background-size: cover;
                background-attachment: fixed;
                background-position: center;
            }}

            h1, h2, h3 {{
                color: #5d4037;
                text-shadow: 0 1px 1px #ffffffaa;
            }}

            .stTextInput > div > div > input {{
                background-color: #fdf6e3cc;
                border: 1px solid #a1887f;
                border-radius: 10px;
                padding: 0.6rem;
                color: #4e342e;
                transition: all 0.3s ease;
            }}

            footer {{ visibility: hidden; }}

            header[data-testid="stHeader"] {{
                display: none;
            }}
        """, unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("⚠️ Background image not found. Please check the path.")

def inject_custom_styles() -> None:
    """
    Inject additional custom CSS styles for UI elements.
    """
    st.markdown("""
    <style>
        div[data-testid="column"] div:nth-child(1) button {
            background-color: rgba(76, 175, 80, 0.2);
            border: 1px solid #4CAF50;
            color: #2e7d32;
            font-weight: 600;
            border-radius: 10px;
        }

        div[data-testid="column"] div:nth-child(1) + div button {
            background-color: rgba(244, 67, 54, 0.2);
            border: 1px solid #f44336;
            color: #c62828;
            font-weight: 600;
            border-radius: 10px;
        }

        button:hover {
            opacity: 0.85;
        }

        button[kind="secondary"] {
            background-color: #fdf6e3cc;
            border: 1px solid #a1887f;
            border-radius: 12px;
            padding: 0.6rem 1rem;
            font-size: 1.1rem;
            font-weight: 500;
            color: #4e342e;
            box-shadow: 2px 2px 6px rgba(0,0,0,0.1);
            transition: all 0.2s ease-in-out;
        }

        button[kind="secondary"]:hover {
            transform: scale(1.05);
            background-color: #fbe9e7;
        }

        .feedback-button {
            display: inline-block;
            padding: 0.75rem 1.5rem;
            background-color: #fbe9e7;
            border: 1px solid #a1887f;
            border-radius: 12px;
            font-size: 1.05rem;
            font-weight: 500;
            color: #4e342e;
            text-decoration: none;
            box-shadow: 2px 2px 6px rgba(0,0,0,0.12);
            transition: all 0.25s ease-in-out;
        }

        .feedback-button:hover {
            background-color: #f8d9ce;
            transform: scale(1.03);
            box-shadow: 3px 3px 8px rgba(0,0,0,0.2);
        }
    </style>
    """, unsafe_allow_html=True)

def translate_to_english(text: str) -> str:
    """
    Translate any input text to English.

    Args:
        text (str): Input text in any language.

    Returns:
        str: Translated text in English, or original text if translation fails.
    """
    try:
        return GoogleTranslator(source="auto", target="en").translate(text)
    except Exception:
        return text

@st.cache_resource
def load_emotion_model():
    """
    Load the emotion classifier pipeline from HuggingFace.

    Returns:
        Pipeline: Text classification pipeline.
    """
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=None
    )

@st.cache_resource
def load_theme_model():
    """
    Load the zero-shot classifier for theme detection.

    Returns:
        Pipeline: Zero-shot classification pipeline.
    """
    return pipeline(
        "zero-shot-classification",
        model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    )

def get_top_theme(text: str, lang: str) -> dict:
    """
    Detect the top theme from a given text and return it in the appropriate language.

    Args:
        text (str): Input text.
        lang (str): Language code ('en' or 'es').

    Returns:
        dict: Dictionary with top theme and score.
    """
    classifier = load_theme_model()
    themes_en = ["Love", "Faith", "Hope", "Forgiveness", "Fear"]
    themes_es = ["Amor", "Fe", "Esperanza", "Perdón", "Miedo"]
    result = classifier(text, candidate_labels=themes_en)
    label = result["labels"][0]
    score = result["scores"][0]
    if lang == "es":
        label = dict(zip(themes_en, themes_es)).get(label, label)
    return {"label": label, "score": score}

def render_language_selector() -> tuple[str, dict]:
    """
    Render the language selection UI and return selected language and translation dictionary.

    Returns:
        tuple[str, dict]: Selected language code and translation dictionary.
    """
    lang_options = {"ES": "es", "EN": "en"}
    col1, col2 = st.columns([7, 3])
    with col2:
        default_lang = st.session_state.get("lang_selector", "ES")
        label = "Elegir idioma" if default_lang == "ES" else "Choose language"
        st.markdown(f"**{label}**")
        selected_key = st.radio(
            "Language selector",
            options=list(lang_options.keys()),
            horizontal=True,
            key="lang_selector",
            label_visibility="collapsed"
        )
    language = lang_options[selected_key]
    return language, TEXTS[language]

def render_user_inputs(T: dict) -> tuple[str, str]:
    """
    Render the main title, subtitle, and input fields for user name and user text.

    Args:
        T (dict): Translation dictionary for the selected language.

    Returns:
        tuple[str, str]: User name and user input text.
    """
    with st.container():
        # Title
        st.markdown("""
            <h1 style='font-family: Cormorant Garamond, serif; font-size: 2.5rem; font-weight: 600;
            color: #5d4037; text-align: center; margin-top: 0.5rem; text-shadow: 1px 1px 2px rgba(0,0,0,0.2);'>
            📖 Lingua Animae 📖</h1>
        """, unsafe_allow_html=True)

        # Subtitle
        st.markdown(
            f"<p style='font-family: Merriweather, serif; font-size: 1.15rem; line-height: 1.6; font-weight: 300; text-shadow: 0.5px 0.5px 1px rgba(0,0,0,0.15); text-align: center; margin-top: 0.5rem;'>"
            f"{T['subtitle']}</p>", unsafe_allow_html=True
        )

        # Mental health note
        st.markdown(
            f"<p style='font-family: Merriweather, serif; font-size: 0.65rem; opacity: 0.7; text-align: center; margin-top: -0.5rem;'>"
            f"{T['mental_health_note']}</p>", unsafe_allow_html=True
        )

        # Input: user name
        st.markdown(
            f"<p style='font-size: 1.1rem; font-weight: 500; margin-top: 1rem;'>💬 {T['name_label']}</p>",
            unsafe_allow_html=True
        )
        usuario = st.text_input("Nombre", key="nombre_usuario", label_visibility="collapsed")

        # Input: user text
        st.markdown(
            f"<p style='font-size: 1.1rem; font-weight: 500; margin-top: 1rem;'>{T['input_label']}</p>",
            unsafe_allow_html=True
        )
        user_input = st.text_input("Texto", key="user_input", label_visibility="collapsed")

    return usuario, user_input

def validate_user_input(text: str, lang: str) -> bool:
    """
    Validate user input to ensure it has enough content.

    Args:
        text (str): User input text.
        lang (str): Language code ('en' or 'es').

    Returns:
        bool: True if input is valid, False otherwise. Shows a warning if invalid.
    """
    if text and len(text.strip()) < 3:
        warning_msg = "⚠️ Por favor, escribe un poco más de contexto." if lang == "es" \
                      else "⚠️ Please write a bit more context."
        st.warning(warning_msg)
        return False
    return True

def analyze_user_input(text: str, lang: str) -> tuple[dict, dict, str, pd.DataFrame]:
    """
    Perform translation, emotion classification, theme classification, and verse recommendation.

    Args:
        text (str): The user input text.
        lang (str): The language code ('en' or 'es').

    Returns:
        tuple[dict, dict, str, pd.DataFrame]:
            - top_emotion: dict with emotion label and score.
            - theme_result: dict with theme label and score.
            - translated: translated input text (English).
            - recommendations: DataFrame of recommended verses.
    """
    try:
        with st.spinner("Analizando..." if lang == "es" else "Analyzing..."):
            translated = translate_to_english(text)

            # Emotion classification
            try:
                emotion_result = load_emotion_model()(translated)
                top_emotion = max(emotion_result[0], key=lambda x: x["score"])
            except Exception:
                st.error("❌ No se pudo analizar la emoción del texto." if lang == "es"
                         else "❌ Failed to analyze emotion from text.")
                st.stop()

            # Theme classification
            try:
                theme_result = get_top_theme(translated, lang=lang)
            except Exception:
                st.error("❌ No se pudo detectar el tema principal." if lang == "es"
                         else "❌ Failed to detect the main theme.")
                st.stop()

            # Load verse corpus
            df_verses = load_entire_corpus(lang=lang)
            if df_verses.empty:
                st.warning("⚠️ No se pudo cargar el corpus de versículos." if lang == "es"
                           else "⚠️ Verse corpus could not be loaded.")
                st.stop()

            # Generate recommendations
            recommendations = recommend_verses(
                df_verses,
                top_emotion["label"],
                theme_result["label"],
                lang=lang
            )

    except Exception:
        st.error("❌ Ha ocurrido un error inesperado durante el análisis." if lang == "es"
                 else "❌ An unexpected error occurred during analysis.")
        st.stop()

    return top_emotion, theme_result, translated, recommendations

from src.utils.translation_maps import BOOK_NAME_MAP_ES

def render_analysis_results(
    T: dict,
    user_input: str,
    translated: str,
    emotion: dict,
    theme: dict,
    recommendations: pd.DataFrame,
    lang: str
) -> None:
    """
    Render the results of the analysis: detected emotion, theme, translation, and verse recommendations.

    Args:
        T (dict): Translation dictionary.
        user_input (str): Original user input text.
        translated (str): Translated version of the user input.
        emotion (dict): Detected emotion label and score.
        theme (dict): Detected theme label and score.
        recommendations (pd.DataFrame): Recommended verses.
        lang (str): Selected language code.
    """
    st.markdown("""
        <div style='text-align: center; margin-top: 2.5rem; margin-bottom: 0.5rem;'>
            <span style='font-size: 1.5rem;'>✶</span>
        </div>
        <hr style='border: none; border-top: 1.2px solid #5d4037; margin: 0 auto 1.5rem auto; width: 60%;'>
    """, unsafe_allow_html=True)

    st.markdown(T["detected"], unsafe_allow_html=True)
    render_emotion_block(st, emotion["label"], emotion["score"], lang=lang)

    st.markdown(T["theme_detected"], unsafe_allow_html=True)
    render_theme_block(st, theme["label"], theme["score"], lang=lang)

    st.markdown(
        f"<p style='font-size: 0.95rem; color: #4e342e; text-shadow: 0.5px 0.5px 1px rgba(0,0,0,0.2); margin-top: 1rem; text-align: center;'>"
        f"{T['translated_as']} <i>{translated}</i></p>",
        unsafe_allow_html=True
    )

    if not recommendations.empty:
        st.markdown(
            f"<h3 style='color: #4e342e;'>📖 {'Versículos recomendados' if lang == 'es' else 'Recommended verses'}:</h3>",
            unsafe_allow_html=True
        )

        for _, row in recommendations.iterrows():
            # Normalize book name (especially for Spanish)
            book_raw = row["book"].lower().replace("_", "-")
            book_display = (
                BOOK_NAME_MAP_ES.get(book_raw, book_raw.title())
                if lang == "es"
                else book_raw.replace("-", " ").title()
            )

            st.markdown(
                f"""
                <div style='margin-bottom: 1.2rem; padding: 1rem; border-left: 4px solid #5d4037;
                            background-color: #fefefeaa; border-radius: 8px;
                            box-shadow: 0 1px 3px rgba(0,0,0,0.08);'>
                    <p style='font-size: 1.05rem; line-height: 1.6; margin-bottom: 0.5rem;'>{row['text']}</p>
                    <p style='font-size: 0.9rem; color: #6d4c41; font-style: italic;'>({book_display} {row['chapter']}:{row['verse']})</p>
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.markdown(
            f"<p style='color: #6d4c41; font-style: italic;'>"
            f"{'No se encontraron coincidencias.' if lang == 'es' else 'No matches found.'}</p>",
            unsafe_allow_html=True
        )


def render_feedback_section_final(user_name: str, user_input: str, recommendations: pd.DataFrame, emotion: dict, theme: dict, lang: str) -> None:
    """
    Render the feedback block and the external feedback form link.

    Args:
        user_name (str): Name of the user.
        user_input (str): Original input text.
        recommendations (pd.DataFrame): Recommended verses.
        emotion (dict): Detected emotion and score.
        theme (dict): Detected theme and score.
        lang (str): Language code.
    """
    render_feedback_section(user_name, user_input, recommendations, emotion, theme, lang)

    form_url = "https://forms.gle/y61oV5xXLqew22K4A"
    label = "📋 Sería de gran ayuda conocer tu opinión:" if lang == "es" else "📋 Would you like to give us more detailed feedback?"
    button_text = "Ir al formulario del feedback" if lang == "es" else "Open feedback form"

    st.markdown(
        f"""
        <div style='text-align: center; margin-top: 2.5rem;'>
            <p style='font-size: 1.15rem;'>{label}</p>
            <a href="{form_url}" target="_blank" class="feedback-button">{button_text}</a>
        </div>
        """,
        unsafe_allow_html=True
    )

def main() -> None:
    """
    Main Streamlit app logic.
    Handles setup, language selection, user input, analysis, and output rendering.
    """
    # === Visual setup ===
    image_path = Path(__file__).parent / "assets" / "old-wrinkled-paper.jpg"
    set_background(str(image_path))
    inject_custom_styles()

    # === Language selector ===
    lang, T = render_language_selector()

    # === Inputs ===
    user_name, user_input = render_user_inputs(T)

    # === Input validation ===
    if not validate_user_input(user_input, lang):
        st.stop()

    # === Analysis and results ===
    if user_input:
        emotion, theme, translated, recommendations = analyze_user_input(user_input, lang)
        render_analysis_results(T, user_input, translated, emotion, theme, recommendations, lang)
        render_feedback_section_final(user_name, user_input, recommendations, emotion, theme, lang)

if __name__ == "__main__":
    main()