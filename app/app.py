import sys
from pathlib import Path

# Ensure src/ is in the path for module resolution
sys.path.append(str(Path(__file__).resolve().parent.parent))

import streamlit as st
import base64
import pandas as pd
from pathlib import Path
from deep_translator import GoogleTranslator
from transformers import pipeline
from texts import TEXTS
from components.render_emotion import render_emotion_block
from components.render_theme import render_theme_block
from components.render_feedback import render_feedback_section
from src.interface.recommender import load_entire_corpus, recommend_verses
from src.utils.save_feedback_to_gsheet import save_feedback_to_gsheet


# === Streamlit config ===
st.set_page_config(page_title="Lingua Animae", page_icon="📖")

def get_base64_bg(image_path: str) -> str:
    """
    Reads an image from the given path and encodes it as a base64 string.
    Used for embedding images in CSS backgrounds.
    """
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(image_path: str):
    """
    Sets the Streamlit app background using a base64-encoded image and injects custom CSS styles.
    """
    try:
        bg_base64 = get_base64_bg(image_path)
        st.markdown(
            f"""
            <style>
                @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;600&family=Merriweather:wght@300;400&display=swap');

                html, body, .stApp, [class^="css"], [class*="st-"] {{
                    font-family: 'Cormorant Garamond', serif !important;
                    font-size: 22px;
                    color: #4e342e;
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

                footer {{
                    visibility: hidden;
                }}

                .st-emotion-cache-1u2mwt6 eu6p4el3 {{
                    max-width: 800px;
                    margin: 3rem auto;
                    padding: 2rem;
                    background-color: rgba(255, 255, 255, 0.5);
                    border-radius: 12px;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.05);
                }}

                /* === Feedback buttons === */
                div[data-testid="column"] div:nth-child(1) button {{
                    background-color: rgba(76, 175, 80, 0.2); /* translucent green */
                    border: 1px solid #4CAF50;
                    color: #2e7d32;
                    font-weight: 600;
                    border-radius: 10px;
                }}

                div[data-testid="column"] div:nth-child(1) + div button {{
                    background-color: rgba(244, 67, 54, 0.2); /* translucent red */
                    border: 1px solid #f44336;
                    color: #c62828;
                    font-weight: 600;
                    border-radius: 10px;
                }}

                button:hover {{
                    opacity: 0.85;
                }}
                div[class*="emotion-block"], div[class*="theme-block"], .emotion-card, .theme-card {{
                font-family: 'Merriweather', serif !important;
                }}
                button[kind="secondary"] {{
                    background-color: #fdf6e3cc;
                    border: 1px solid #a1887f;
                    border-radius: 12px;
                    padding: 0.6rem 1rem;
                    font-size: 1.1rem;
                    font-weight: 500;
                    color: #4e342e;
                    box-shadow: 2px 2px 6px rgba(0,0,0,0.1);
                    transition: all 0.2s ease-in-out;
                }}
                button[kind="secondary"]:hover {{
                    transform: scale(1.05);
                    background-color: #fbe9e7;
                }}

            </style>
            """,
            unsafe_allow_html=True
        )
    except FileNotFoundError:
        # Show a warning if the background image is not found
        st.warning("⚠️ Background image not found. Please check the path.")

# CSS
st.markdown("""
<style>
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
    Translates the input text to English using GoogleTranslator.
    If translation fails, returns the original text.
    """
    try:
        return GoogleTranslator(source="auto", target="en").translate(text)
    except Exception:
        return text

@st.cache_resource
def load_emotion_model():
    """
    Loads and returns the HuggingFace emotion classification pipeline.
    Cached for performance.
    """
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=None
    )

@st.cache_resource
def load_theme_model():
    """
    Loads and returns the HuggingFace zero-shot classification pipeline for themes.
    Cached for performance.
    """
    return pipeline(
        "zero-shot-classification",
        model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    )

def get_top_theme(text: str, lang: str) -> dict:
    """
    Classifies the input text into predefined themes and returns the top theme and its score.
    Maps theme label to Spanish if the language is Spanish.
    """
    classifier = load_theme_model()
    themes_en = ["Love", "Faith", "Hope", "Forgiveness", "Fear"]
    themes_es = ["Amor", "Fe", "Esperanza", "Perdón", "Miedo"]
    result = classifier(text, candidate_labels=themes_en)
    top_label_en = result["labels"][0]
    score = result["scores"][0]
    if lang == "es":
        # Map English themes to Spanish if needed
        label_map = dict(zip(themes_en, themes_es))
        top_label = label_map.get(top_label_en, top_label_en)
    else:
        top_label = top_label_en
    return {"label": top_label, "score": score}

def main():
    """
    Main function for the Streamlit app.
    Handles UI, user input, emotion/theme analysis, recommendations, and feedback.
    """
    # Set the background image
    image_path = Path(__file__).parent / "assets" / "old-wrinkled-paper.jpg"
    set_background(str(image_path))
    
    # Language selection
    lang_options = {"ES": "es", "EN": "en"}
    col1, col2 = st.columns([7, 3])
    with col2:
        label = "Elegir idioma" if st.session_state.get("lang_selector", "ES") == "ES" else "Choose language"
        st.markdown(f"**{label}**")
        selected_key = st.radio(
            "Language selector",
            options=list(lang_options.keys()),
            horizontal=True,
            key="lang_selector",
            label_visibility="collapsed"
        )
    language = lang_options[selected_key]
    T = TEXTS[language]

    with st.container():
        # Main title
        st.markdown("""<h1 style='font-family: Cormorant Garamond, serif; font-size: 2.5rem; font-weight: 600;
            color: #5d4037; text-align: center; margin-top: 0.5rem; text-shadow: 1px 1px 2px rgba(0,0,0,0.2);'>
            📖 Lingua Animae 📖</h1>""", unsafe_allow_html=True)

        # Subtitle
        st.markdown(
            f"<p style='font-family: Merriweather, serif; font-size: 1.15rem; line-height: 1.6; font-weight: 300; text-shadow: 0.5px 0.5px 1px rgba(0,0,0,0.15); text-align: center; margin-top: 0.5rem;'>"
            f"{T['subtitle']}</p>", unsafe_allow_html=True)

        # Input: User name
        st.markdown(
            f"<p style='font-size: 1.1rem; font-weight: 500; margin-top: 1rem;'>💬 {T['name_label']}</p>",
            unsafe_allow_html=True
        )
        usuario = st.text_input("Nombre", key="nombre_usuario", label_visibility="collapsed")

        # Input: User text
        st.markdown(
            f"<p style='font-size: 1.1rem; font-weight: 500; margin-top: 1rem;'>{T['input_label']}</p>",
            unsafe_allow_html=True
        )
        user_input = st.text_input("Texto", key="user_input", label_visibility="collapsed")

        # Validate input length
        if user_input and len(user_input.strip()) < 3:
            # Warn if input is too short
            st.warning("⚠️ Por favor, escribe un poco más de contexto." if language == "es"
                    else "⚠️ Please write a bit more context.")
            st.stop()

        # === Handle user input ===
        if user_input:
            # === Validate input length ===
            if len(user_input.strip()) < 3:
                st.warning("⚠️ Por favor, escribe un poco más de contexto." if language == "es"
                        else "⚠️ Please write a bit more context.")
                st.stop()

            # === Phase 1: Analysis ===
            try:
                with st.spinner("Analizando..." if language == "es" else "Analyzing..."):
                    # Translation
                    translated = translate_to_english(user_input)

                    # Emotion classification
                    try:
                        emotion_result = load_emotion_model()(translated)
                        top_emotion = max(emotion_result[0], key=lambda x: x["score"])
                    except Exception:
                        st.error("❌ No se pudo analizar la emoción del texto." if language == "es"
                                else "❌ Failed to analyze emotion from text.")
                        st.stop()

                    # Theme classification
                    try:
                        theme_result = get_top_theme(translated, lang=language)
                    except Exception:
                        st.error("❌ No se pudo detectar el tema principal." if language == "es"
                                else "❌ Failed to detect the main theme.")
                        st.stop()

                    # Load verse corpus
                    df_verses = load_entire_corpus(lang=language)
                    if df_verses.empty:
                        st.warning("⚠️ No se pudo cargar el corpus de versículos." if language == "es"
                                else "⚠️ Verse corpus could not be loaded.")
                        st.stop()

                    # Generate recommendations
                    recommendations = recommend_verses(
                        df_verses,
                        top_emotion["label"],
                        theme_result["label"],
                        lang=language
                    )

            except Exception:
                st.error("❌ Ha ocurrido un error inesperado durante el análisis." if language == "es"
                        else "❌ An unexpected error occurred during analysis.")
                st.stop()

            # === Phase 2: Display results ===
            st.markdown("""<div style='text-align: center; margin-top: 2.5rem; margin-bottom: 0.5rem;'>
                    <span style='font-size: 1.5rem;'>✶</span></div>
                <hr style='border: none; border-top: 1.2px solid #5d4037; margin: 0 auto 1.5rem auto; width: 60%;'>""",
                unsafe_allow_html=True)

            st.markdown(T["detected"], unsafe_allow_html=True)
            render_emotion_block(st, top_emotion["label"], top_emotion["score"], lang=language)

            st.markdown(T["theme_detected"], unsafe_allow_html=True)
            render_theme_block(st, theme_result["label"], theme_result["score"], lang=language)

            st.markdown(
                f"<p style='font-size: 0.95rem; color: #4e342e; text-shadow: 0.5px 0.5px 1px rgba(0,0,0,0.2); margin-top: 1rem; text-align: center;'>"
                f"{T['translated_as']} <i>{translated}</i></p>",
                unsafe_allow_html=True
            )

            # === Display recommendations ===
            if not recommendations.empty:
                st.markdown(
                    f"<h3 style='color: #4e342e;'>📖 {'Versículos recomendados' if language == 'es' else 'Recommended verses'}:</h3>",
                    unsafe_allow_html=True
                )

                for _, row in recommendations.iterrows():
                    st.markdown(
                        f"""
                        <div style='margin-bottom: 1.2rem; padding: 1rem; border-left: 4px solid #5d4037;
                                    background-color: #fefefeaa; border-radius: 8px;
                                    box-shadow: 0 1px 3px rgba(0,0,0,0.08);'>
                            <p style='font-size: 1.05rem; line-height: 1.6; margin-bottom: 0.5rem;'>{row['text']}</p>
                            <p style='font-size: 0.9rem; color: #6d4c41; font-style: italic;'>({row['book'].capitalize()} {row['chapter']}:{row['verse']})</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            else:
                st.markdown(
                    f"<p style='color: #6d4c41; font-style: italic;'>"
                    f"{'No se encontraron coincidencias.' if language == 'es' else 'No matches found.'}</p>",
                    unsafe_allow_html=True
                )
            # === Render feedback section ===
            render_feedback_section(
                usuario,
                user_input,
                recommendations,
                top_emotion,
                theme_result,
                language
            )

            # === Optional: Google Forms link for detailed feedback ===
            form_url = "https://forms.gle/y61oV5xXLqew22K4A" 

            label = "📋 Sería de gran ayuda conocer tu opinión:" if language == "es" else "📋 Would you like to give us more detailed feedback?"
            button_text = "Ir al formulario del feedback" if language == "es" else "Open feedback form"

            st.markdown(
                f"""
                <div style='text-align: center; margin-top: 2.5rem;'>
                    <p style='font-size: 1.15rem;'>{label}</p>
                    <a href="{form_url}" target="_blank" class="feedback-button">{button_text}</a>
                </div>
                """,
                unsafe_allow_html=True
            )

if __name__ == "__main__":
    main()
