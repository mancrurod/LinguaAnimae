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
- `pandas`, `scikit-learn`, `regex`
- `beautifulsoup4`, `requests`
- `streamlit` – multilingual app for emotion/theme-based verse recommendation

---

## 📁 Project Structure

```
LinguaAnimae/
├── .streamlit/                        # Streamlit secrets and config
│   └── secrets.toml
├── app/                               # Streamlit app frontend
│   ├── assets/                        # Visual assets (background image)
│   │   └── old-wrinkled-paper.jpg
│   ├── components/                    # UI rendering components
│   │   ├── render_emotion.py
│   │   └── render_theme.py
│   ├── app.py                         # Main Streamlit entry point
│   └── texts.py                       # Multilingual UI dictionary
├── data/
│   ├── raw/                           # Original scraped texts
│   ├── processed/                     # Cleaned and merged verse data
│   └── labeled/                       # Emotion and theme-labeled corpora
│       └── <bible_name>/
│           ├── emotion/
│           └── emotion_theme/
├── logs/
│   ├── labeling_logs/                 # Logs from the labeling pipeline
│   └── cleaning_logs/                 # Logs from cleaning steps
├── notebooks/                         # Data exploration and validation
│   ├── 01_scraping_exploration.ipynb
│   ├── 02_cleaning.ipynb
│   ├── 03_label_emotions_and_themes.ipynb
│   ├── 04_translate_labels.ipynb
│   └── 05_evaluation.ipynb
├── src/
│   ├── interface/
│   │   ├── recommender.py
│   │   └── labeling_pipeline.py
│   ├── modeling/
│   │   ├── emotion_theme_labeling.py
│   │   ├── theme_labeling.py
│   │   └── labeling_pipeline.py
│   ├── preprocessing/
│   │   ├── cleaning.py
│   │   ├── merge.py
│   │   └── translate_and_apply_labels.py
│   ├── scraping/
│   │   ├── bible_scraper.py
│   │   └── parse_osis_kjv.py
│   └── utils/
│       ├── save_feedback_to_gsheet.py
│       └── translation_maps.py
├── tests/                             # Future test coverage
├── .gitignore
├── requirements.txt
├── environment.yml
├── README.md
├── CHANGELOG.md
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

The interactive Streamlit app allows users to input a free-form emotional message and receive recommended Bible verses matching its **emotion** and **theme**.

### Features

- 🔄 **Automatic translation** of input (EN/ES)
- 🧠 **Emotion detection** (6 Plutchik categories)
- 🏷️ **Theme classification** (5 canonical themes)
- 📖 **Context-aware verse matching** from KJV or RV60
- 🎨 **Stylized cards** with emotion/theme color, emoji, and verse metadata
- ✅ **User feedback collection** via like/dislike buttons (stored in Google Sheets)

### Example

Input:

> *Tengo miedo y necesito consuelo...*

Returns:

📖 *Génesis 40:7* — *"¿Por qué parecen hoy mal vuestros semblantes?"*

---

## 📤 Feedback System

Users can now rate the relevance of the emotion/theme detection with a 👍 / 👎 system.  
Feedback is saved to a **Google Sheet** along with:

- Original input
- Detected emotion and score
- Detected theme and score
- User name (optional)
- Feedback value (`like` / `dislike`)

This enables future model refinement and analytics.

---

## ✨ UI Enhancements

- Feedback buttons styled with semantic colors and **hover animation**
- Subtitles, emotion/theme blocks, and translation notices are now **centered and consistently styled**
- Merriweather font applied to all key UI blocks for elegance and readability


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