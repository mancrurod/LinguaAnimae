"""
Compare emotion labels from multiple models on Bible verses.

Models: Hartmann, GoEmotions (SamLowe), Bhadresh Savani, EmoRoBERTa, Joeddav.
All results saved in a CSV for downstream analysis.
"""

import pandas as pd
from transformers import pipeline
from pathlib import Path
from tqdm import tqdm
import torch

# =============================
# === CONFIGURACIÓN GENERAL ===
# =============================

EMOTION_SCORE_THRESHOLD = 0.35  # Para GoEmotions, EmoRoBERTa y Joeddav
PATH_INPUT = Path(__file__).parent.parent / "data" / "processed" / "bible_kjv" / "1_genesis_cleaned.csv"
PATH_OUTPUT = Path(__file__).parent.parent / "data" / "processed" / "bible_kjv" / "1_genesis_emotion_comparison.csv"

MODEL_HARTMANN = "j-hartmann/emotion-english-distilroberta-base"
MODEL_SAMLOWE = "SamLowe/roberta-base-go_emotions"
MODEL_BSAVANI = "bhadresh-savani/distilbert-base-uncased-emotion"
MODEL_EMOROBERTA = "arpanghoshal/EmoRoBERTa"
MODEL_JOEDDAV = "joeddav/distilbert-base-uncased-go-emotions-student"
BATCH_SIZE = 32

# === Mapeo de etiquetas GoEmotions a Ekman ===
GOEMOTIONS_TO_EKMAN = {
    "admiration": "joy",
    "amusement": "joy",
    "anger": "anger",
    "annoyance": "anger",
    "approval": "joy",
    "caring": "joy",
    "confusion": "surprise",
    "curiosity": "surprise",
    "desire": "joy",
    "disappointment": "sadness",
    "disapproval": "disgust",
    "disgust": "disgust",
    "embarrassment": "sadness",
    "excitement": "joy",
    "fear": "fear",
    "gratitude": "joy",
    "grief": "sadness",
    "joy": "joy",
    "love": "joy",
    "nervousness": "fear",
    "optimism": "joy",
    "pride": "joy",
    "realization": "surprise",
    "relief": "joy",
    "remorse": "sadness",
    "sadness": "sadness",
    "surprise": "surprise",
    "neutral": "neutral"
}

# ========================
# === FUNCIONES AUX ===
# ========================

def load_csv(path: Path) -> pd.DataFrame:
    """Load cleaned verses CSV."""
    return pd.read_csv(path)

def label_with_hartmann(texts, pipe):
    """Label batch with Hartmann model. Returns top label and score."""
    results = pipe(texts)
    labels, scores = [], []
    for res in results:
        if isinstance(res, list): res = res[0]
        labels.append(res["label"])
        scores.append(res["score"])
    return labels, scores

def label_with_bsadistilbert(texts, pipe):
    """
    Label batch with Bhadresh Savani DistilBERT emotion model.
    Returns top label and score.
    """
    results = pipe(texts)
    labels, scores = [], []
    for res in results:
        if isinstance(res, list): res = res[0]
        labels.append(res["label"])
        scores.append(res["score"])
    return labels, scores

def label_with_multilabel_model(texts, pipe, model_name):
    """
    Generic function for multilabel models (EmoRoBERTa, Joeddav, GoEmotions).
    If top-1 emotion is 'neutral' and score < threshold, assign the next highest non-neutral.
    Returns top label, mapped Ekman label, and score.
    """
    results = pipe(texts)
    labels, ekman_labels, scores = [], [], []
    for res in results:
        preds = res if isinstance(res, list) else [res]
        preds_sorted = sorted(preds, key=lambda x: x["score"], reverse=True)
        top = preds_sorted[0]
        label = top["label"]
        ekman_label = GOEMOTIONS_TO_EKMAN.get(label, "neutral")
        score = top["score"]
        # Solo modelos que incluyen 'neutral' necesitan buscar alternativa
        has_neutral = model_name in ["goemotions", "joeddav"]
        if ekman_label == "neutral" and score < EMOTION_SCORE_THRESHOLD and has_neutral:
            for pred in preds_sorted[1:]:
                l = pred["label"]
                mapped = GOEMOTIONS_TO_EKMAN.get(l, "neutral")
                if mapped != "neutral":
                    label = l
                    ekman_label = mapped
                    score = pred["score"]
                    break
        labels.append(label)
        ekman_labels.append(ekman_label)
        scores.append(score)
    return labels, ekman_labels, scores

# ========================
# === SCRIPT PRINCIPAL ===
# ========================

def main():
    print("🚀 Loading verses...")
    df = load_csv(PATH_INPUT)
    textos = df["text"].tolist()

    print("✅ Loading models (may take a while)...")
    pipe_hartmann = pipeline(
        "text-classification",
        model=MODEL_HARTMANN,
        top_k=1,
        device=0 if torch.cuda.is_available() else -1
    )
    pipe_goemotions = pipeline(
        "text-classification",
        model=MODEL_SAMLOWE,
        top_k=None,
        device=0 if torch.cuda.is_available() else -1
    )
    pipe_bsadistilbert = pipeline(
        "text-classification",
        model=MODEL_BSAVANI,
        top_k=1,
        device=0 if torch.cuda.is_available() else -1
    )
    pipe_joeddav = pipeline(
        "text-classification",
        model=MODEL_JOEDDAV,
        top_k=None,
        device=0 if torch.cuda.is_available() else -1
    )

    # Resultados
    results = []

    print("🏷️  Labeling verses...")
    for start in tqdm(range(0, len(textos), BATCH_SIZE)):
        batch = textos[start:start+BATCH_SIZE]
        
        # Hartmann
        hart_labels, hart_scores = label_with_hartmann(batch, pipe_hartmann)
        # GoEmotions (SamLowe)
        go_labels, go_ekman_labels, go_scores = label_with_multilabel_model(batch, pipe_goemotions, "goemotions")
        # Bhadresh Savani
        bsavani_labels, bsavani_scores = label_with_bsadistilbert(batch, pipe_bsadistilbert)
        # Joeddav GoEmotions Student
        joeddav_labels, joeddav_ekman, joeddav_scores = label_with_multilabel_model(batch, pipe_joeddav, "joeddav")
        
        for i in range(len(batch)):
            results.append({
                "text": batch[i],
                "hartmann_label": hart_labels[i],
                "hartmann_score": hart_scores[i],
                "goemotions_label": go_labels[i],
                "goemotions_ekman_label": go_ekman_labels[i],
                "goemotions_score": go_scores[i],
                "bsavani_label": bsavani_labels[i],
                "bsavani_score": bsavani_scores[i],
                "joeddav_label": joeddav_labels[i],
                "joeddav_ekman_label": joeddav_ekman[i],
                "joeddav_score": joeddav_scores[i]
            })

    # DataFrame final
    out_df = pd.DataFrame(results)
    out_df.to_csv(PATH_OUTPUT, index=False)
    print(f"✅ Done! Results saved to {PATH_OUTPUT.absolute()}")

if __name__ == "__main__":
    main()
