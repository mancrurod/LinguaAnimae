# app/components/render_emotion.py

def render_emotion_block(st, label: str, score: float, lang: str = "en"):
    EMOTION_COLORS = {
        "joy": "#d4af37",
        "sadness": "#90caf9",
        "anger": "#ef5350",
        "fear": "#e57373",
        "trust": "#a5d6a7",
        "surprise": "#ce93d8",
        "neutral": "#e0e0e0",
        "love": "#f48fb1",
        "disgust": "#c5e1a5"
    }

    EMOTION_ICONS = {
        "joy": "🌟",
        "sadness": "😔",
        "anger": "😡",
        "fear": "😱",
        "trust": "💕",
        "surprise": "😮",
        "neutral": "⚪",
        "love": "💖",
        "disgust": "🤮"
    }

    EMOTION_TRANSLATIONS = {
        "joy": "Alegría",
        "sadness": "Tristeza",
        "anger": "Ira",
        "fear": "Miedo",
        "trust": "Confianza",
        "surprise": "Sorpresa",
        "neutral": "Neutral",
        "love": "Amor",
        "disgust": "Asco"
    }

    label_lc = label.lower()
    color = EMOTION_COLORS.get(label_lc, "#eeeeee")
    icon = EMOTION_ICONS.get(label_lc, "")
    label_translated = EMOTION_TRANSLATIONS[label_lc] if lang == "es" else label.capitalize()

    porcentaje = score * 100
    if lang == "es":
        porcentaje_str = f"{porcentaje:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    else:
        porcentaje_str = f"{porcentaje:.2f}"

    st.markdown(
        f"""
        <style>
            @keyframes fadeInUp {{
                from {{
                    opacity: 0;
                    transform: translateY(20px);
                }}
                to {{
                    opacity: 1;
                    transform: translateY(0);
                }}
            }}
            .emotion-block {{
                animation: fadeInUp 0.8s ease-out;
            }}
        </style>

        <div class='emotion-block' style='
            background-color: {color};
            padding: 1rem;
            border-radius: 12px;
            border: 1px solid #5d4037;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
            margin-top: 1rem;
            text-align: center;
        '>
            <h3 style='margin: 0; color: #4e342e; font-weight: 600;'>
                {icon} {label_translated}: {porcentaje_str}%
            </h3>
        </div>
        """,
        unsafe_allow_html=True
    )
