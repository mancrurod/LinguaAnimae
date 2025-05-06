# app/app.py

import streamlit as st
import base64
from pathlib import Path
from deep_translator import GoogleTranslator
from transformers import pipeline
from texts import TEXTS
from components.render_emotion import render_emotion_block

# === Streamlit config ===
st.set_page_config(page_title="Lingua Animae", page_icon="📖")

# === Convert local image to base64 ===
def get_base64_bg(image_path: str) -> str:
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# === Inject CSS with base64 background and visual theme ===
def set_background(image_path: str):
    try:
        bg_base64 = get_base64_bg(image_path)
        st.markdown(
            f"""
            <style>
                @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;600&family=Merriweather:wght@300;400&display=swap');

                html, body, .stApp, [class^="css"], [class*="st-"] {{
                    font-family: 'Cormorant Garamond', serif !important;
                    font-size: 20px;
                    color: #4e342e;
                    background-color: transparent;
                    text-shadow: 0.4px 0.4px 0.6px rgba(0,0,0,0.08);
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

            </style>
            """,
            unsafe_allow_html=True
        )
    except FileNotFoundError:
        st.warning("⚠️ Background image not found. Please check the path.")


# === Translate Spanish input to English ===
def translate_to_english(text: str) -> str:
    try:
        return GoogleTranslator(source="auto", target="en").translate(text)
    except Exception:
        return text

# === Load the HuggingFace emotion model once ===
@st.cache_resource
def load_emotion_model():
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=None
    )

# === Load the HuggingFace theme classification model ===
@st.cache_resource
def load_theme_model():
    return pipeline(
        "zero-shot-classification",
        model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    )

# === Detect thematic label based on input text and language ===
def get_top_theme(text: str, lang: str) -> dict:
    classifier = load_theme_model()

    # Candidate labels based on UI language
    themes_en = ["Love", "Faith", "Hope", "Forgiveness", "Fear"]
    themes_es = ["Amor", "Fe", "Esperanza", "Perdón", "Miedo"]

    # Always classify in English (text is already translated)
    result = classifier(text, candidate_labels=themes_en)

    top_label_en = result["labels"][0]
    score = result["scores"][0]

    if lang == "es":
        label_map = dict(zip(themes_en, themes_es))
        top_label = label_map.get(top_label_en, top_label_en)
    else:
        top_label = top_label_en

    return {"label": top_label, "score": score}

# === Main app logic ===
def main():
    # Set background
    image_path = Path(__file__).parent / "assets" / "old-wrinkled-paper.jpg"
    set_background(str(image_path))

    # Language options
    lang_options = {"ES": "es", "EN": "en"}

    # Create two columns, right one for selector
    col1, col2 = st.columns([7, 3])

    with col2:
        label = "Elegir idioma" if st.session_state.get("lang_selector", "ES") == "ES" else "Choose language"
        st.markdown(f"**{label}**")
        selected_key = st.radio(
            label="",
            options=list(lang_options.keys()),
            horizontal=True,
            key="lang_selector"
        )

    # Apply language choice
    language = lang_options[selected_key]
    T = TEXTS[language]

    with st.container():
        # Aquí va TODO el contenido principal:
        # título, subtítulo, input, resultado, etc.

        # === Title and subtitle ===
        st.markdown(
            """
            <h1 style='
                font-family: Cormorant Garamond, serif;
                font-size: 2.5rem;
                font-weight: 600;
                color: #5d4037;
                text-align: center;
                margin-top: 0.5rem;
                text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
            '>📖 Lingua Animae 📖</h1>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"<p style='font-family: Merriweather, serif; font-size: 1.15rem; "
            f"line-height: 1.6; font-weight: 300; "
            f"text-shadow: 0.5px 0.5px 1px rgba(0,0,0,0.15);'>"
            f"{T['subtitle']}</p>",
            unsafe_allow_html=True
        )



        # === User input and emotion detection ===
        st.markdown(
            f"<p style='font-size: 1.3rem; font-weight: 500; margin-bottom: 0.2rem; "
            f"text-shadow: 0.5px 0.5px 1px rgba(0,0,0,0.2);'>"
            f"{T['input_label']}</p>",
            unsafe_allow_html=True
        )


        user_input = st.text_input(label="")

        if user_input:
            with st.spinner("Analizando..." if language == "es" else "Analyzing..."):
                translated = translate_to_english(user_input)
                emotion_result = load_emotion_model()(translated)
                top_emotion = max(emotion_result[0], key=lambda x: x["score"])

                # === Thematic classification ===
                theme_result = get_top_theme(translated, lang=language)

            # Ornamental separator
            st.markdown(
                """
                <div style='text-align: center; margin-top: 2.5rem; margin-bottom: 0.5rem;'>
                    <span style='font-size: 1.5rem;'>✶</span>
                </div>
                <hr style='border: none; border-top: 1.2px solid #5d4037; margin: 0 auto 1.5rem auto; width: 60%;'>
                """,
                unsafe_allow_html=True
            )

            # Show emotion
            st.markdown(f"<h3 style='color: #4e342e;'>🧠 {T['detected'].replace('### ', '')}</h3>", unsafe_allow_html=True)
            render_emotion_block(st, top_emotion["label"], top_emotion["score"], lang=language)

            # Show theme
            st.markdown(
                f"<h3 style='color: #4e342e;'>🏷️ {'Tema detectado' if language == 'es' else 'Detected theme'}: "
                f"{theme_result['label']} — {theme_result['score']*100:.2f}%</h3>",
                unsafe_allow_html=True
            )

            # Translated text
            st.markdown(
                f"<p style='font-size: 0.95rem; font-style: italic; color: #4e342e; "
                f"text-shadow: 0.5px 0.5px 1px rgba(0,0,0,0.2); margin-top: 1rem;'>"
                f"{T['translated_as']} <i>{translated}</i></p>",
                unsafe_allow_html=True
            )


if __name__ == "__main__":
    main()
