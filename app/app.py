import sys
from pathlib import Path

# Ensure src/ is in the path for module resolution
sys.path.append(str(Path(__file__).resolve().parent.parent))

import streamlit as st
import base64
from deep_translator import GoogleTranslator
from transformers import pipeline
from texts import TEXTS
from components.render_emotion import render_emotion_block
from components.render_theme import render_theme_block
from src.interface.recommender import load_corpus, recommend_verses

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
    themes_en = ["Love", "Faith", "Hope", "Forgiveness", "Fear"]
    themes_es = ["Amor", "Fe", "Esperanza", "Perdón", "Miedo"]
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
    image_path = Path(__file__).parent / "assets" / "old-wrinkled-paper.jpg"
    set_background(str(image_path))
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
        st.markdown("""
            <h1 style='font-family: Cormorant Garamond, serif; font-size: 2.5rem; font-weight: 600;
            color: #5d4037; text-align: center; margin-top: 0.5rem; text-shadow: 1px 1px 2px rgba(0,0,0,0.2);'>
            📖 Lingua Animae 📖</h1>
        """, unsafe_allow_html=True)

        st.markdown(
            f"<p style='font-family: Merriweather, serif; font-size: 1.15rem; line-height: 1.6; font-weight: 300; text-shadow: 0.5px 0.5px 1px rgba(0,0,0,0.15);'>"
            f"{T['subtitle']}</p>", unsafe_allow_html=True)

        st.markdown(
            f"<p style='font-size: 1.3rem; font-weight: 500; margin-bottom: 0.2rem; text-shadow: 0.5px 0.5px 1px rgba(0,0,0,0.2);'>"
            f"{T['input_label']}</p>", unsafe_allow_html=True)

        user_input = st.text_input("User input", label_visibility="collapsed")

        if user_input:
            with st.spinner("Analizando..." if language == "es" else "Analyzing..."):
                translated = translate_to_english(user_input)
                emotion_result = load_emotion_model()(translated)
                top_emotion = max(emotion_result[0], key=lambda x: x["score"])
                theme_result = get_top_theme(translated, lang=language)

            st.markdown("""
                <div style='text-align: center; margin-top: 2.5rem; margin-bottom: 0.5rem;'>
                    <span style='font-size: 1.5rem;'>✶</span>
                </div>
                <hr style='border: none; border-top: 1.2px solid #5d4037; margin: 0 auto 1.5rem auto; width: 60%;'>
            """, unsafe_allow_html=True)

            st.markdown(T["detected"], unsafe_allow_html=True)
            render_emotion_block(st, top_emotion["label"], top_emotion["score"], lang=language)

            st.markdown(T["theme_detected"], unsafe_allow_html=True)
            render_theme_block(st, theme_result["label"], theme_result["score"], lang=language)

            st.markdown(
                f"<p style='font-size: 0.95rem; font-style: italic; color: #4e342e; text-shadow: 0.5px 0.5px 1px rgba(0,0,0,0.2); margin-top: 1rem;'>"
                f"{T['translated_as']} <i>{translated}</i></p>", unsafe_allow_html=True)

            book = "1_genesis"
            df_verses = load_corpus(book, lang=language)
            recommendations = recommend_verses(
                df_verses,
                top_emotion["label"],
                theme_result["label"],
                lang=language
            )

            if not recommendations.empty:
                st.markdown(
                    f"<h3 style='color: #4e342e;'>📖 {'Versículos recomendados' if language == 'es' else 'Recommended verses'}:</h3>",
                    unsafe_allow_html=True)
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

if __name__ == "__main__":
    main()
