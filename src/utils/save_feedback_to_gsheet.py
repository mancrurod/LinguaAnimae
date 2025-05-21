from datetime import datetime
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import streamlit as st

def save_feedback_to_gsheet(feedback_data: dict):
    """
    Safely saves user feedback data to a Google Sheet, never deleting previous rows.
    If the header changes, missing columns are appended at the end; old data is preserved.

    Args:
        feedback_data (dict): Dictionary containing feedback fields.
            Must include:
                - usuario, texto, emocion, emocion_pct, tema, tema_pct, feedback
            Optionally:
                - versiculo_1, versiculo_2, ..., versiculo_n
    """
    try:
        # === Google Sheets setup ===
        scope = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive"
        ]

        creds_dict = {
            "type": st.secrets["google"]["type"],
            "project_id": st.secrets["google"]["project_id"],
            "private_key_id": st.secrets["google"]["private_key_id"],
            "private_key": st.secrets["google"]["private_key"].replace("\\n", "\n"),
            "client_email": st.secrets["google"]["client_email"],
            "client_id": st.secrets["google"]["client_id"],
            "token_uri": st.secrets["google"]["token_uri"],
        }

        credentials = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(credentials)
        sheet = client.open_by_url(st.secrets["spreadsheet"]["url"])
        worksheet = sheet.get_worksheet(0)

        # === Define base fields and dynamic versiculo_* fields ===
        fixed_fields = ["usuario", "texto", "emocion", "emocion_pct", "tema", "tema_pct", "feedback"]
        dynamic_fields = sorted(
            [k for k in feedback_data if k.startswith("versiculo_")],
            key=lambda x: int(x.split("_")[1])
        )
        all_fields = fixed_fields + dynamic_fields

        # === Prepare header (timestamp always first) ===
        desired_header = ["timestamp"] + all_fields
        existing_values = worksheet.get_all_values()
        if existing_values:
            current_header = existing_values[0]
        else:
            current_header = []

        # === If header is empty, write it ===
        if not current_header:
            worksheet.update("A1", [desired_header])
            current_header = desired_header

        # === If header exists, add missing columns at the end ===
        if current_header != desired_header:
            # Find missing columns (excluding timestamp, always first)
            missing = [col for col in desired_header[1:] if col not in current_header]
            # Append missing columns to current header
            new_header = current_header + missing
            # Update header only if different
            if new_header != current_header:
                worksheet.update("A1", [new_header])
                current_header = new_header

        # === Prepare row, using header order ===
        timestamp = datetime.now().isoformat(timespec="seconds")
        row = [timestamp]  # Always start with timestamp
        for field in current_header[1:]:  # Exclude timestamp (already added)
            row.append(feedback_data.get(field, ""))

        # === Append new feedback ===
        worksheet.append_row(row, value_input_option="USER_ENTERED")

    except Exception as e:
        st.error("❌ Failed to save feedback to Google Sheets.")
        st.exception(e)
