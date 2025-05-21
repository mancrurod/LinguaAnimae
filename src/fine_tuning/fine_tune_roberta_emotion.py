import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import numpy as np
from sklearn.metrics import classification_report
from pathlib import Path

# === 1. Parámetros y emociones válidas ===
EMOTION_MAP = {
    "joy": "Alegría",
    "sadness": "Tristeza",
    "anger": "Ira",
    "fear": "Miedo",
    "trust": "Confianza",
    "surprise": "Sorpresa",
    "neutral": "Neutral"
}
label_list = list(EMOTION_MAP.keys())
label2id = {l: i for i, l in enumerate(label_list)}
id2label = {i: l for i, l in enumerate(label_list)}

print("Labels esperados:", label_list)
print("label2id:", label2id)

# === 2. Carga y validación del CSV ===
DATA_FILE = Path(__file__).parent.parent.parent / "data" / "evaluation" / "emotion_verses_labeled_combined.csv"
df = pd.read_csv(DATA_FILE)

# Limpia y valida
df['label'] = df['label'].str.strip().str.lower()
invalid_labels = set(df['label'].unique()) - set(label_list)
if invalid_labels:
    print(f"❌ ERROR: Detected invalid labels: {invalid_labels}")
    raise ValueError("Your CSV contains labels not present in the expected set.")

# Mapea a índices enteros y asegura tipo int
df['label'] = df['label'].map(label2id).astype(int)
print("Labels tras mapeo:", sorted(df['label'].unique()))

# === 3. Split en train/test ===
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
split = int(len(df) * 0.9)
df_train = df.iloc[:split].copy()
df_test = df.iloc[split:].copy()

# Asegura tipo int en train/test
df_train['label'] = df_train['label'].astype(int)
df_test['label'] = df_test['label'].astype(int)

# === 4. Datasets HuggingFace ===
ds = DatasetDict({
    "train": Dataset.from_pandas(df_train[['verse', 'label']].rename(columns={'verse':'text'}), preserve_index=False),
    "test": Dataset.from_pandas(df_test[['verse', 'label']].rename(columns={'verse':'text'}), preserve_index=False),
})

# === 5. Tokenizer y modelo ===
tokenizer = AutoTokenizer.from_pretrained("SamLowe/roberta-base-go_emotions")
model = AutoModelForSequenceClassification.from_pretrained(
    "SamLowe/roberta-base-go_emotions",
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)
model.config.problem_type = "single_label_classification"

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding=True)

tokenized_datasets = ds.map(preprocess_function, batched=True)

# === 6. TrainingArguments ===
training_args = TrainingArguments(
    output_dir="./results_finetuned_bible",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=4,
    weight_decay=0.01,
    push_to_hub=False,
    logging_steps=10,
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1"
)

# === 7. Métricas ===
def compute_metrics(pred):
    preds = np.argmax(pred.predictions, axis=1)
    labels = pred.label_ids
    report = classification_report(labels, preds, target_names=[id2label[i] for i in range(len(label_list))], output_dict=True, zero_division=0)
    macro_f1 = report["macro avg"]["f1-score"]
    return {"macro_f1": macro_f1}

print(ds['train'][0])
print("Tipo de label:", type(ds['train'][0]['label']))

# === 8. Trainer ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)

trainer.train()
trainer.save_model("finetuned-goemotions-bible")

print("✅ Fine-tuning completed and model saved as 'finetuned-goemotions-bible/'")
