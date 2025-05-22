import sys
from pathlib import Path

# Ensure src/ is in the path for module resolution
sys.path.append(str(Path(__file__).resolve().parent.parent))

import streamlit as st
import base64
import pandas as pd
import deepl
import os
from dotenv import load_dotenv
from transformers import pipeline
import requests

import psutil
import os
import streamlit as st

from texts import TEXTS
from components.render_emotion import render_emotion_block
from components.render_theme import render_theme_block
from components.render_feedback import render_feedback_section
from src.interface.recommender import load_entire_corpus, recommend_verses_by_sections
from src.utils.save_feedback_to_gsheet import save_feedback_to_gsheet
from src.utils.translation_maps import BOOK_NAME_MAP_ES
from src.utils.translation_maps import GO_EMOTIONS_TO_EKMAN

load_dotenv()

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
            @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;600;700;800;900&family=Merriweather:wght@300;400&display=swap');

            html, body, .stApp, [class^="css"], [class*="st-"] {{
                font-family: 'Cormorant Garamond', serif !important;
                font-size: 18px;
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

            .stApp > header, .stApp [data-testid="stHeader"], .block-container {{
                margin-top: 0 !important;
                padding-top: 0 !important;
            }}
            .block-container {{
                margin-top: 0 !important;
                padding-top: 1rem !important;
            }}

            .stTextInput > div > div > input {{
                background-color: #fdf6e3cc;
                border: 1px solid #5d4037;
                border-radius: 10px;
                padding: 0.6rem;
                color: #4e342e;
                transition: all 0.3s ease;
            }}

            header[data-testid="stHeader"] {{
                display: none;
            }}

            span[data-testid="stFormSubmitHelper"] {{
                display: none !important;
            }}

            div[data-testid="InputInstructions"] {{
                display: none !important;
            }}

        </style>
        """, unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("⚠️ Background image not found. Please check the path.")

def inject_custom_styles() -> None:
    """
    Inject additional custom CSS styles for UI elements.
    """
    st.markdown("""
    <style>
                
        div[data-testid="stSpinner"], .stSpinner {
            display: none !important;
        }
                
        div[data-testid="column"] div:nth-child(1) button {
            background-color: #fbe9e7;
            border: 1px solid #5d4037;
            color: #2e7d32;
            font-weight: 600;
            border-radius: 10px;
        }

        div[data-testid="column"] div:nth-child(1) + div button {
            background-color: #fbe9e7;
            border: 1px solid #5d4037;
            color: #c62828;
            font-weight: 600;
            border-radius: 10px;
        }
                
        button:hover {
            opacity: 0.85;
        }

        button[kind="secondary"] {
            background-color: #fdf6e3cc;
            border: 1px solid #5d4037;
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

        button[data-testid="stBaseButton-secondary"] {
            background-color: #f3e5d1;
            border: 1px solid #5d4037;
            border-radius: 12px;
            font-size: 1.05rem;
            font-weight: 500;
            color: #4e342e;
            text-decoration: none;
            box-shadow: 2px 2px 6px rgba(0,0,0,0.6);
            transition: all 0.25s ease-in-out;
            padding: 0.75rem 1.5rem;
        }

        button[data-testid="stBaseButton-secondary"]:hover {
            background-color: #fff1c9;
            transform: scale(1.05);
            box-shadow: 3px 3px 8px rgba(0,0,0,0.2);
        }
                
        button[data-testid="stBaseButton-secondaryFormSubmit"] {
            font-family: 'Cormorant Garamond', serif !important;
            background-color: #f3e5d1;
            border: 1px solid #5d4037;
            border-radius: 12px;
            padding: 0.75rem 1.5rem;
            font-size: 1.2rem;
            font-weight: 800 !important;
            color: #4e342e;
            box-shadow: 2px 2px 12px rgba(0,0,0,0.6);
            display: block;
            margin: 1rem auto 0 auto;
            transition: all 0.2s ease-in-out;
        }
                
        button[data-testid="stBaseButton-secondaryFormSubmit"] p {
            font-family: 'Cormorant Garamond', serif !important;
            font-weight: 800 !important;
        }

        button[data-testid="stBaseButton-secondaryFormSubmit"]:hover {
            background-color: #fff1c9;
            transform: scale(1.05);
        }
                
        div[data-testid="stForm"] {
            background-color: #f3e5d1;
            border: 1px solid #5d4037; 
            border-radius: 16px;
            padding: 2rem 2.5rem;
            margin-top: 1rem;
            box-shadow: 2px 2px 12px rgba(0, 0, 0, 0.6);
        }
                
        div[data-testid="stRadio"] label {
            background: none !important;
            border: none !important;
            font-size: 1.6rem;
            transition: transform 0.25s ease, filter 0.25s ease;
            cursor: pointer;
            transform-origin: center;
        }

        div[data-testid="stRadio"] label:hover {
            transform: scale(1.3);
            filter: brightness(1.3);
        }

        div[data-testid="stRadio"] input[type="radio"]:checked + div p {
            transform: scale(1.3);
            text-decoration: underline;
        }

        div[data-testid="stRadio"] {
            display: flex !important;
            justify-content: flex-end;
        }
                
        .custom-spinner-box {
            max-width: 400px;
            margin: 2rem auto;
            padding: 1rem 1.5rem;
            background-color: rgba(243, 229, 209, 0.85);
            border: 1px solid #5d4037;
            border-radius: 12px;
            box-shadow: 2px 2px 6px rgba(0, 0, 0, 0.15);
            text-align: center;
            font-family: 'Merriweather', serif;
            font-size: 1.05rem;
            color: #4e342e;
        }

        .custom-spinner-box p::after {
            content: ' ⏳';
            animation: pulseDots 1.2s infinite;
        }

        @keyframes pulseDots {
            0%   { opacity: 0.2; }
            50%  { opacity: 1; }
            100% { opacity: 0.2; }
        }
                
        /* Style for the feedback link inside a centered Markdown block */
        div[data-testid="stMarkdownContainer"] div[style*="text-align: center"] > a.feedback-button {
            display: inline-block;
            background-color: #f3e5d1;
            border: 1px solid #5d4037;
            border-radius: 12px;
            padding: 0.6rem 1.2rem;
            font-size: 1rem;
            font-weight: 500;
            font-family: 'Merriweather', serif;
            color: #4e342e;
            text-decoration: none;
            box-shadow: 2px 2px 6px rgba(0, 0, 0, 0.1);
            transition: all 0.25s ease-in-out;
            cursor: pointer;
            margin-top: 1rem;
        }

        /* Hover */
        div[data-testid="stMarkdownContainer"] div[style*="text-align: center"] > a.feedback-button:hover {
            background-color: #fff1c9;
            transform: scale(1.03);
            box-shadow: 3px 3px 8px rgba(0, 0, 0, 0.2);
        }

                                     
    </style>
    """, unsafe_allow_html=True)

def get_deepl_key():
    # Try st.secrets first, then fallback to env variable
    if "DEEPL_API_KEY" in st.secrets:
        return st.secrets["DEEPL_API_KEY"]
    elif "DEEPL_API_KEY" in os.environ:
        return os.environ["DEEPL_API_KEY"]
    else:
        raise ValueError("DEEPL_API_KEY not set!")


def translate_to_english(text: str) -> str:
    """
    Translate any input text to English using DeepL API.

    Args:
        text (str): Input text in any language.

    Returns:
        str: Translated text in English, or original if translation fails.
    """
    try:
        api_key = get_deepl_key()
        translator = deepl.Translator(api_key)
        result = translator.translate_text(text, target_lang="EN-US")
        return result.text
    except Exception as e:
        # Cambia a st.error si usas Streamlit en la interfaz
        st.error(f"❌ DeepL translation failed: {e}")
        return text

@st.cache_resource
def load_emotion_model():
    """
    Load the emotion classifier pipeline from HuggingFace (SamLowe GoEmotions).

    Returns:
        Pipeline: Text classification pipeline.
    """
    return pipeline(
        "text-classification",
        model="SamLowe/roberta-base-go_emotions",
        top_k=None
    )

def classify_ekman_emotion(text, emotion_model):
    """
    Classify the Ekman emotion using the SamLowe GoEmotions model and the mapping.

    Args:
        text (str): Input text in English.
        emotion_model (Pipeline): HuggingFace pipeline for GoEmotions.

    Returns:
        dict: Dict with 'ekman_label', 'go_label', and 'score'.
    """
    preds = emotion_model(text)[0]
    top_pred = max(preds, key=lambda x: x["score"])
    go_label = top_pred["label"]
    ekman_label = GO_EMOTIONS_TO_EKMAN.get(go_label, "neutral")
    return {"ekman_label": ekman_label, "go_label": go_label, "score": top_pred["score"]}


import os
import requests
import streamlit as st
from dotenv import load_dotenv

# Cargar variables de entorno desde .env
load_dotenv()

def load_theme_model():
    """
    Fetch the zero-shot classifier for theme detection from HuggingFace using their API.
    Ensure the model is private, authenticate with a HuggingFace token.

    Returns:
        function: Function to query HuggingFace API for theme classification.
    """
    api_url = "https://api-inference.huggingface.co/models/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    hf_token = os.getenv("HF_API_TOKEN")

    if not hf_token:
        st.error("❌ No se encontró el token de Hugging Face. Asegúrate de definir HF_API_TOKEN en tu archivo .env.")
        raise EnvironmentError("Token de Hugging Face no encontrado.")

    headers = {"Authorization": f"Bearer {hf_token}"}

    def query(payload):
        try:
            response = requests.post(api_url, headers=headers, json=payload)
            response.raise_for_status()  # Si la respuesta no es 200, lanzará un error
            return response.json()  # Devuelve la respuesta como JSON
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Error en la solicitud a la API de HuggingFace: {e}")
            st.error(f"Error response: {response.text if response else 'No response'}")
            raise e

    return query

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

    payload = {
        "inputs": text,  # Texto de entrada
        "parameters": {
            "candidate_labels": themes_en  # Etiquetas candidatas
        }
    }
    
    try:
        result = classifier(payload)  # Realizamos la llamada al modelo

        if "labels" in result and "scores" in result:
            label = result["labels"][0]
            score = result["scores"][0]
        else:
            st.error(f"❌ Respuesta inesperada de la API: {result}")
            raise ValueError("Respuesta inesperada de la API.")

        if lang == "es":
            label = dict(zip(themes_en, themes_es)).get(label, label)

        return {"label": label, "score": score}

    except Exception as e:
        st.error(f"❌ Error al procesar el análisis de tema: {e}")
        raise e

def render_language_selector() -> tuple[str, dict]:
    lang_options = {"ES": "es", "EN": "en"}

    # Validate previous key
    stored_key = st.session_state.get("lang_selector", "ES")
    default_lang_key = stored_key if stored_key in lang_options else "ES"
    default_lang_code = lang_options[default_lang_key]
    T = TEXTS[default_lang_code]

    with st.container():
        st.markdown(f"""
        <div style='text-align: right; width: 100%;'>
            <p style="font-weight: 600; font-size: 1rem; font-family: 'Cormorant Garamond', serif; margin-bottom: 0.3rem;">
                {T["language_label"]}
            </p>
        </div>
        """, unsafe_allow_html=True)

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
            <h1 style='font-family: Cormorant Garamond, serif; font-size: 2rem; font-weight: 600;
            color: #5d4037; text-align: center; margin-top: -1.5rem; text-shadow: 1px 1px 2px rgba(0,0,0,0.2);'>
            📖 Lingua Animae 📖</h1>
        """, unsafe_allow_html=True)

        # Subtitle
        st.markdown(
            f"<p style='font-family: Merriweather, serif; font-size: 1rem; line-height: 1.6; font-weight: 300; text-shadow: 0.5px 0.5px 1px rgba(0,0,0,0.15); text-align: center; margin-top: 0.5rem;'>"
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
            spinner_placeholder = st.empty()  # create a temporary container

            spinner_placeholder.markdown(f"""
            <div class="custom-spinner-box">
                <p>{'Analizando...' if lang == 'es' else 'Analyzing...'}</p>
            </div>
            """, unsafe_allow_html=True)

            # The actual analysis happens here
            translated = translate_to_english(text)

            # Emotion classification
            try:
                emotion_model = load_emotion_model()
                log_memory_usage("After loading emotion model")
                emotion_result = classify_ekman_emotion(translated, emotion_model)
                top_emotion = {
                    "label": emotion_result["ekman_label"],   # Esta es la emoción Ekman mapeada
                    "go_label": emotion_result["go_label"],   # (Opcional, por si quieres mostrar la etiqueta original)
                    "score": emotion_result["score"]
                }
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
            log_memory_usage("After loading corpus")
            if df_verses.empty:
                st.warning("⚠️ No se pudo cargar el corpus de versículos." if lang == "es"
                           else "⚠️ Verse corpus could not be loaded.")
                st.stop()

            # Generate recommendations
            recommendations = recommend_verses_by_sections(
                df_verses,
                top_emotion["label"],
                theme_result["label"],
                lang=lang
            )
            log_memory_usage("After generating recommendations")

    except Exception:
        st.error("❌ Ha ocurrido un error inesperado durante el análisis." if lang == "es"
                 else "❌ An unexpected error occurred during analysis.")
        st.stop()

    # Remove the box once finished
    spinner_placeholder.empty()
    
    log_memory_usage("End of analyze_user_input")
    return top_emotion, theme_result, translated, recommendations

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
        f"<p style='font-size: 0.75rem; color: #4e342e; text-shadow: 0.5px 0.5px 1px rgba(0,0,0,0.2); margin-top: 1rem; text-align: center; opacity: 0.7;'>"
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
                            box-shadow: 0 1px 3px rgba(0,0,0,0.6);'>
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

    form_url = "https://forms.gle/ATXVWXTaoCDR19rf9"
    label = "📋 Sería de gran ayuda conocer su opinión:" if lang == "es" else "📋 Would you like to give us more detailed feedback?"
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



def log_memory_usage(tag=""):
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / 1024 ** 2
    st.write(f"🔍 [{tag}] Memory usage: {mem_mb:.2f} MB")
    # Si no quieres que salga en la interfaz, usa logging:
    # import logging; logging.info(f"Memory usage: {mem_mb:.2f} MB")

def main():
    log_memory_usage("Start of main")
    # === Visual setup ===
    image_path = Path(__file__).parent / "assets" / "old-wrinkled-paper.jpg"
    set_background(str(image_path))
    inject_custom_styles()

    # === Language selector ===
    lang, T = render_language_selector()

    # === Input form ===
    submit_text = T["submit_button"]
    with st.form(key="input_form"):
        user_name, user_input = render_user_inputs(T)
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            submit = st.form_submit_button(submit_text)

    # === On submit: run analysis and store results ===
    if submit:
        if not validate_user_input(user_input, lang):
            st.stop()

        emotion, theme, translated, recommendations = analyze_user_input(user_input, lang)
        render_analysis_results(T, user_input, translated, emotion, theme, recommendations, lang)
        log_memory_usage("After render_analysis_results")

        # Store everything needed for feedback in session state
        st.session_state.analysis_ready = True
        st.session_state.user_name_text = user_name
        st.session_state.user_input_text = user_input
        st.session_state.emotion = emotion
        st.session_state.theme = theme
        st.session_state.recommendations = recommendations
        st.session_state.lang = lang
        log_memory_usage("After saving to session_state")


    # === Feedback section (persists after submit) ===
    if st.session_state.get("analysis_ready"):
        render_feedback_section_final(
            st.session_state.user_name_text,
            st.session_state.user_input_text,
            st.session_state.recommendations,
            st.session_state.emotion,
            st.session_state.theme,
            lang  # <--- solo esto, no pases T ni session_state.lang
        )

    # === Footer ===
    st.markdown("""
        <style>
        .custom-footer {
            position: fixed;
            bottom: 0;
            width: 28%;
            background-color: rgba(243, 229, 209, 0.85);
            color: #4e342e;
            text-align: center;
            font-size: 0.65rem;
            padding: 0.6rem 1rem;
            font-family: 'Merriweather', serif;
            border-top: 1px solid #5d4037;
            border-radius: 10px 10px 0 0;
            box-shadow: 0px -2px 12px rgba(0, 0, 0, 0.5);
            z-index: 9999;
        }
        .custom-footer a {
            color: #6d4c41;
            text-decoration: none;
            font-weight: 500;
            display: inline-block;
            position: relative; 
            transition: all 0.25s ease
        }
                
        .custom-footer a:hover {
            transform: scale(1.5);
            filter: drop-shadow(0 0 4px rgba(255,255,255,0.4)) brightness(1.2);
            text-shadow: 0 0 2px rgba(255, 255, 255, 0.2);
        }
        </style>

        <div class="custom-footer">
            Hecho con ❤️ por Manuel Cruz Rodríguez · <a href="https://www.linkedin.com/in/mancrurod/" target="_blank">🌐</a>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()