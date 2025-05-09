from datetime import datetime
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import streamlit as st

def save_feedback_to_gsheet(feedback_data: dict):
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

    # Cabecera esperada
    header = ["timestamp", "usuario", "texto", "emoción", "emoción_pct", "tema", "tema_pct", "feedback"]

    # Obtener la primera fila (si existe)
    existing_values = worksheet.get_all_values()
    if not existing_values or existing_values[0] != header:
        worksheet.insert_row(header, index=1)

    # Nueva fila de datos
    row = [
        datetime.now().isoformat(timespec="seconds"),
        feedback_data["usuario"],
        feedback_data["texto"],
        feedback_data["emocion"],
        feedback_data["emocion_pct"],
        feedback_data["tema"],
        feedback_data["tema_pct"],
        feedback_data["feedback"]
    ]

    worksheet.append_row(row, value_input_option="USER_ENTERED")
