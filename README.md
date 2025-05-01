# 📖 LinguaAnimae

**LinguaAnimae** is a multilingual NLP pipeline that classifies and explores sacred texts through the lens of **themes** and **emotions**, culminating in a **Streamlit-based chatbot** that retrieves Bible verses aligned with natural language prompts.

---

## 🔍 Project Goals

- Extract and normalize full Bible corpora (English + Spanish)
- Annotate every verse with emotion and theme labels
- Translate annotations for multilingual consistency
- Power a semantic chatbot that suggests aligned verses in real time
- Support additional domains like poetry or music lyrics (planned)

---

## 🧠 Core Technologies

- **Python 3.10+**
- `transformers`, `torch`, `sentence-transformers`
- `pandas`, `scikit-learn`, `numpy`, `regex`
- `beautifulsoup4`, `requests`
- `streamlit` – interactive frontend (planned)
- `sqlite` or `duckdb` – fast local verse retrieval

---

## 📁 Project Structure

```
LinguaAnimae/
├── data/
│   ├── raw/                    # Unprocessed texts
│   ├── processed/              # Cleaned verse-by-verse CSVs
│   └── labeled/                # Emotion & theme-labeled output
│       └── <bible_name>/
│           ├── emotion/
│           └── emotion_theme/
├── logs/                       # Labeling summaries and timers
│   └── labeling_logs/
├── notebooks/
│   ├── 01_scraping_exploration.ipynb
│   ├── 03_label_emotions_and_themes.ipynb
│   ├── 04_translate_labels.ipynb
│   └── 05_evaluation.ipynb
├── src/
│   ├── scraping/               # HTML + OSIS parsing
│   ├── preprocessing/          # Cleaning, translation
│   ├── interface/              # Labeling pipeline CLI
│   └── modeling/               # (future) classifier training
├── tests/
├── requirements.txt
├── environment.yml
└── README.md
```

---

## 🚀 Getting Started

You can set up the environment using either `conda` (recommended) or `pip`.

### 🧪 Option 1: Using Conda (recommended)

```bash
conda env create -f environment.yml
conda activate linguaanimae
```

### 💡 Option 2: Using pip

1. Clone the repository
```bash
git clone https://github.com/your-username/LinguaAnimae.git
cd LinguaAnimae
```

2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Run the Bible scraper to download all books
```bash
python src/scraping/bible_scraper.py
```

---

## 🧰 Usage

### 1. Scrape the Bible (RV60)

Use the scraping script to extract the full Reina-Valera 1960 Bible and save it as structured CSVs:

```bash
python src/scraping/bible_scraper.py
```

### 2. Label Verses with Emotions + Themes

Use the labeling pipeline to classify English Bible verses (bible_kjv) using pretrained HuggingFace models:

```bash
python src/interface/labeling_pipeline.py --bible bible_kjv
```

Optional flags:

- --skip-emotion to skip emotion classification
- --skip-theme to skip theme labeling
- --device -1 to force CPU mode (default is --device 0 for GPU)
- --dry-run path/to/file.csv to test a single file

### 3. Translate Labels into Spanish

Align the English emotion/theme annotations with their Spanish verse equivalents in bible_rv60:

```bash
python src/preprocessing/translate_and_apply_labels.py
```

This creates a labeled Spanish version under:

```bash
data/labeled/bible_rv60/emotion_theme/
```

---

## 💬 Coming Soon: Streamlit Chatbot

### Example Interaction

> Type something like:  
> *"I'm feeling anxious about the future..."*  
>  
> And receive:  
> 📖 **Jeremías 29:11** — *"Porque yo sé los planes que tengo para ti…"*

The chatbot will:
- Analyze user prompts
- Match them to verses based on emotion, theme, and embeddings
- Return the most semantically aligned results
- Support both English and Spanish lookups

---

## 📊 Outputs

Labeled files are saved to:

- *_emotion.csv: Emotion column using 6 Plutchik labels
- *_emotion_theme.csv: Adds multilabel theme column from 5 canonical themes
- Logs are saved to: logs/labeling_logs/ with per-file runtime and pipeline summary

---

## 📌 Roadmap

Planned features include:

- ✅ Streamlit chatbot interface
- ✅ Cross-lingual label alignment
- 🔍 Embedding-based verse retrieval
- 💽 DuckDB or SQLite integration for fast local querying
- 📊 Annotation statistics and disagreement visualization
- 📦 Export format: JSONL / Parquet for training or downstream use

[See CHANGELOG.md](CHANGELOG.md) for complete history.

---

## 📖 License

For academic and research use only. Sources are derived from public domain Bibles (e.g., RV60, KJV) and open ML models. License will be finalized before v1.0.

---

## ✨ Acknowledgements

Developed by [Manuel Cruz Rodríguez](https://github.com/mancrurod) as part of an NLP and Data Science learning journey.


