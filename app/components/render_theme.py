# app/components/render_theme.py

def render_theme_block(st, label: str, score: float, lang: str = "en"):
    THEME_COLORS = {
        "love": "#f48fb1",
        "faith": "#a5d6a7",
        "hope": "#90caf9",
        "forgiveness": "#ffcc80",
        "fear": "#e57373"
    }

    THEME_ICONS = {
        "love": "💗",
        "faith": "🙏",
        "hope": "🌈",
        "forgiveness": "🕊️",
        "fear": "😨"
    }

    THEME_TRANSLATIONS = {
        "love": "Amor",
        "faith": "Fe",
        "hope": "Esperanza",
        "forgiveness": "Perdón",
        "fear": "Miedo"
    }

    # Inverse translation in case label is already in Spanish
    INV_THEME_TRANSLATIONS = {v.lower(): k for k, v in THEME_TRANSLATIONS.items()}

    label_lc = label.lower().strip()

    # Ensure we're always working with English internally
    label_en = INV_THEME_TRANSLATIONS.get(label_lc, label_lc)

    color = THEME_COLORS.get(label_en, "#eeeeee")
    icon = THEME_ICONS.get(label_en, "")
    label_translated = THEME_TRANSLATIONS[label_en] if lang == "es" else label_en.capitalize()

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
            .theme-block {{
                animation: fadeInUp 0.8s ease-out;
            }}
        </style>

        <div class='theme-block' style='
            background-color: {color};
            padding: 1rem;
            border-radius: 12px;
            border: 1px solid #5d4037;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
            margin-top: 1rem;
            text-align: center;
        '>
            <h3 style='margin: 0; color: #4e342e; font-weight: 600;'>
                {icon} {label_translated} — {score*100:.2f}%
            </h3>
        </div>
        """,
        unsafe_allow_html=True
    )

