import streamlit as st
from src.utils.save_feedback_to_gsheet import save_feedback_to_gsheet
from texts import TEXTS

def render_feedback_section(usuario, user_input, recommendations, top_emotion, theme_result, language):
    """
    Displays a one-time feedback section with clean state management.
    """
    T = TEXTS[language]

    # Initialize session state variables if not already present
    if "feedback_sent" not in st.session_state:
        st.session_state.feedback_sent = False
    if "feedback_submitted" not in st.session_state:
        st.session_state.feedback_submitted = False
    if "feedback_value" not in st.session_state:
        st.session_state.feedback_value = None

    # Feedback submission block (runs only once)
    if st.session_state.feedback_value and usuario and not st.session_state.feedback_submitted:
        # Create a list of formatted verse strings from recommendations DataFrame
        verses_list = [
            f"{row.get('book', '').capitalize()} {row.get('chapter', '')}:{row.get('verse', '')} - {row.get('text', '').strip()}"
            for _, row in recommendations.iterrows()
        ]

        # Prepare feedback data dictionary to be saved
        feedback_data = {
            "usuario": usuario,
            "texto": user_input,
            "emocion": top_emotion["label"],
            "emocion_pct": round(top_emotion["score"] * 100, 2),
            "tema": theme_result["label"],
            "tema_pct": round(theme_result["score"] * 100, 2),
            "feedback": st.session_state.feedback_value
        }

        # Add each recommended verse to the feedback data
        for i, verse in enumerate(verses_list):
            feedback_data[f"versiculo_{i+1}"] = verse

        # Save feedback data to Google Sheet
        save_feedback_to_gsheet(feedback_data)

        # Update session state to indicate feedback has been sent and submitted
        st.session_state.feedback_sent = True
        st.session_state.feedback_submitted = True

    # Show thank you message if feedback has already been sent
    if st.session_state.feedback_sent:
        feedback_message = T["feedback_thanks"]
        st.markdown("""
            <div style="text-align: center; margin-top: 1.5rem;">
                <p style="font-size: 1.3rem; animation: fadein 1s ease-in;">{}</p>
            </div>
            <style>
                @keyframes fadein {{
                    from {{ opacity: 0; }}
                    to   {{ opacity: 1; }}
                }}
            </style>
        """.format(feedback_message), unsafe_allow_html=True)
        return

    # Show feedback buttons if feedback has not been sent yet
    st.markdown(f"<h4 style='text-align: center;'>{T['feedback_question']}</h4>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns([2, 1, 1, 2])
    with col2:
        if st.button("👍"):
            st.session_state.feedback_value = "like"
            st.rerun()
    with col3:
        if st.button("👎"):
            st.session_state.feedback_value = "dislike"
            st.rerun()

