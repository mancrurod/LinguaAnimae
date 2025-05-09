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
- `streamlit` – multilingual app for emotion/theme-based verse recommendation
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
├── logs/                       
│   └── labeling_logs/
│   └── cleaning_logs/
├── notebooks/
│   ├── 01_scraping_exploration.ipynb
│   ├── 02_cleaning.ipynb
│   ├── 03_label_emotions_and_themes.ipynb
│   ├── 04_translate_labels.ipynb
│   └── 05_evaluation.ipynb
├── src/
│   ├── scraping/               
│   ├── preprocessing/          
│   ├── interface/              
│   └── modeling/               
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

## 💬 Streamlit Interface

The interactive app is now live!

You can enter a short prompt like:

> *"I'm feeling anxious about the future..."*  
> or  
> *"Tengo miedo y necesito consuelo..."*

And receive:

📖 **Genesis 40:7** — *"Wherefore look ye so sadly today?"*  

📖 **Génesis 40:7** — *"¿Por qué parecen hoy mal vuestros semblantes?"*

The system:
- Translates input if needed
- Detects main emotion and theme using Hugging Face models
- Loads the appropriate corpus (`bible_kjv` or `bible_rv60`)
- Matches verses labeled with those same emotion + theme
- Returns the most relevant matches as stylized cards

---

## 📊 Outputs

Labeled files are saved to:

- *_emotion.csv: Emotion column using 6 Plutchik labels
- *_emotion_theme.csv: Adds multilabel theme column from 5 canonical themes
- Logs are saved to: logs/labeling_logs/ with per-file runtime and pipeline summary

---

## 📌 Roadmap

### ✅ Completed (Weeks 1–3)
- Full Bible scraping (KJV + RV60)
- Corpus cleaning and normalization
- Emotion and theme labeling using pretrained HuggingFace models
- Cross-lingual label transfer and alignment
- Manual evaluation with accuracy and F1 metrics
- Streamlit interface: emotion + theme detection, stylized results
- Multilingual support: automatic input translation and corpus selection
- Recommendation system based on emotion + theme match

### 🔄 Week 4: Model + Interface Integration and User Testing
- [ ] Connect model inference to real-time recommendations in the interface
- [ ] Run test sessions with 5–10 users
- [ ] Deploy and collect feedback via form (Google Forms or equivalent)

### 🔄 Week 5: Iteration Based on Feedback
- [ ] Refine model behavior and recommendation logic
- [ ] Improve clarity of explanations and label rendering
- [ ] Implement user-suggested improvements

### 🏁 Week 6: Final Demo and Documentation
- [ ] Consolidate the MVP into a cohesive narrative
- [ ] Write technical and functional report
- [ ] Prepare public demo with real examples
- [ ] (Optional) Add export features (PDF), voice synthesis, or word cloud summaries

[See CHANGELOG.md](CHANGELOG.md) for complete history.

---

## 📖 License

For academic and research use only. Sources are derived from public domain Bibles (e.g., RV60, KJV) and open ML models from HugginFace. License will be finalized before v1.0.

---

## ✨ Acknowledgements

Developed by [Manuel Cruz Rodríguez](https://github.com/mancrurod) as part of an NLP and Data Science learning journey.