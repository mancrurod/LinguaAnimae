# 📖 LinguaAnimae

**LinguaAnimae** is a multilingual NLP pipeline that classifies and explores sacred texts through the lens of **themes** and **emotions**, culminating in a **Streamlit-based chatbot** that retrieves Bible verses aligned with natural language prompts.

---

## 🔍 Project Goals

* Extract and normalize full Bible corpora (English + Spanish)
* Annotate every verse with emotion and theme labels
* Translate annotations for multilingual consistency
* Power a semantic chatbot that suggests aligned verses in real time
* Support additional domains like poetry or music lyrics (planned)

---

## 🧠 Core Technologies

* **Python 3.10+**
* `transformers`, `torch`, `sentence-transformers`
* `pandas`, `scikit-learn`, `regex`
* `beautifulsoup4`, `requests`
* `streamlit` – multilingual app for emotion/theme-based verse recommendation

---

## 📁 Project Structure

```
LinguaAnimae/
├── .streamlit/
│   └── secrets.toml
├── app/
│   ├── assets/
│   ├── components/
│   │   ├── render_emotion.py
│   │   ├── render_feedback.py
│   │   ├── render_theme.py
│   ├── app.py
│   └── texts.py
├── data/
│   ├── evaluation/
│   │   ├── verses_labeled_gpt/
│   │   ├── verses_parsed/
│   │   ├── verses_to_label/
│   │   ├── eval_examples.csv
│   │   └── eval_results.csv
│   ├── labeled/
│   ├── processed/
│   └── raw/
├── logs/
├── notebooks/
│   ├── finetuned-goemotions-bible/
│   ├── results_finetuned_bible/
│   ├── 01_scraping_exploration.ipynb
│   ├── 02_cleaning.ipynb
│   ├── 03_label_emotions_and_themes.ipynb
│   ├── 04_translate_labels.ipynb
│   ├── 05_evaluation.ipynb
│   ├── 06_emotion_finetuning_pipeline.ipynb
│   └── viz_models.ipynb
├── src/
│   ├── fine_tuning/
│   │   ├── finetuned-goemotions-bible/
│   │   ├── fine_tune_roberta_emotion.py
│   │   ├── parse_gpt_output_to_labeled_csv.py
│   │   └── select_verses_for_labeling.py
│   ├── interface/
│   │   └── recommender.py
│   ├── modeling/
│   │   ├── emotion_theme_labeling.py
│   │   ├── labeling_pipeline.py
│   │   └── theme_labeling.py
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
├── tests/
├── .gitignore
├── requirements.txt
├── requirements_local.txt
├── environment.yml
├── README.md
├── CHANGELOG.md
```

---

## 🆕 Data Selection, Annotation & Versioning

**Sampling, annotation, and batch tracking workflow:**

- Automated random verse selection script for new annotation rounds, guaranteeing no duplication of already labeled verses.
- Supports multiple annotation rounds with batch/version tracking (`emotion_verses_to_label_X.csv`).
- New annotation batches can be labeled via GPT or other models, then easily merged with existing datasets.
- Utility scripts included for remapping, cleaning, and validating emotion labels prior to model training.
- Each annotation batch and its integration is versioned for reproducibility and experiment traceability.

---

## 📝 Label Mapping and Cleaning

- **Robust label mapping:** All scripts and model pipelines use unified dictionaries for emotion and theme mapping (`EMOTION_MAP`, `THEME_MAP`), ensuring compatibility between annotation, translation, and modeling.
- **Label cleaning utilities:** Automated routines for handling strange/ambiguous emotions and mapping them to the canonical set. Out-of-vocabulary or inconsistent labels are filtered out before training.

---

## 🚦 Model Training & Evaluation

The project now supports full training and evaluation workflows for emotion classification models, including:

- Fine-tuning with Hugging Face Transformers on the annotated Bible corpus.
- Optional oversampling for class balancing during training.
- Comprehensive cross-validation pipeline using StratifiedKFold and HuggingFace Trainer, reporting mean and std of macro F1 across folds.
- Export of classification reports and confusion matrices after each experiment for documentation and analysis.
- Early stopping to prevent overfitting in all model workflows.

See `notebooks/05_evaluation.ipynb` and `src/fine_tuning/` for code examples and experiment tracking.

---

## 🧪 Example: Cross-validation Training

```python
from sklearn.model_selection import StratifiedKFold
from transformers import Trainer

# Use the provided notebook or scripts to perform k-fold cross-validation
# Reports macro F1 per fold and mean ± std for robust model evaluation
````

---

## 🆕 File & Script Changes

* `src/fine_tuning/` — All fine-tuning and evaluation scripts/notebooks (cross-validation, standard train/test, report generation).
* `src/utils/select_random_verses.py` — Script for random batch selection for annotation.
* `src/utils/parse_and_merge_batches.py` — Script to merge new annotation batches and ensure no duplicate verse\_id.
* All selection scripts now ensure that each new annotation batch contains only unique, unlabeled verses.

---

## 📈 Model Versioning & Experiment Tracking

* Each training/fine-tuning run is versioned by date and experiment.
* All metrics, reports, and confusion matrices are saved for each run (see `/results_finetuned_bible/` and related directories).
* Final models for deployment are saved under `/src/fine_tuning/` after evaluation on the full train/test split.

---

## 🚀 Getting Started

You can set up the environment using either `conda` (recommended) or `pip`.

### 🧪 Option 1: Using Conda (recommended)

```bash
conda env create -f environment_local.yml
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

Use the labeling pipeline to classify English Bible verses (bible\_kjv) using pretrained HuggingFace models:

```bash
python src/interface/labeling_pipeline.py --bible bible_kjv
```

Optional flags:

* \--skip-emotion to skip emotion classification
* \--skip-theme to skip theme labeling
* \--device -1 to force CPU mode (default is --device 0 for GPU)
* \--dry-run path/to/file.csv to test a single file

### 3. Translate Labels into Spanish

Align the English emotion/theme annotations with their Spanish verse equivalents in bible\_rv60:

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

* 🔄 **Automatic translation** of input (EN/ES)
* 🧠 **Emotion detection** (6 Ekman categories)
* 🏷️ **Theme classification** (5 canonical themes)
* 📖 **Context-aware verse matching** from KJV or RV60
* 🎨 **Stylized cards** with emotion/theme color, emoji, and verse metadata
* ✅ **User feedback collection** via like/dislike buttons (stored in Google Sheets)

### Example

Input:

> *Tengo miedo y necesito consuelo...*

Returns:

📖 *Génesis 40:7* — *"¿Por qué parecen hoy mal vuestros semblantes?"*

---

## 📤 Feedback System

Users can now rate the relevance of the emotion/theme detection with a 👍 / 👎 system.
Feedback is saved to a **Google Sheet** along with:

* Original input
* Detected emotion and score
* Detected theme and score
* User name (optional)
* Feedback value (`like` / `dislike`)

This enables future model refinement and analytics.

---

## ✨ UI Enhancements

* Feedback buttons styled with semantic colors and **hover animation**
* Subtitles, emotion/theme blocks, and translation notices are now **centered and consistently styled**
* Merriweather font applied to all key UI blocks for elegance and readability
* Book names in verse references are now normalized: numbers are preserved (e.g. `1 Pedro`, `2 Timoteo`) and accents are applied where appropriate (e.g. `Isaías`, `Jeremías`) for Spanish; English names are capitalized and spaced (`1 John`, `2 Timothy`)
* Fully refactored `app.py` into reusable components

---

## 📊 Outputs

Labeled files are saved to:

* \*\_emotion.csv: Emotion column using 6 Plutchik labels
* \*\_emotion\_theme.csv: Adds multilabel theme column from 5 canonical themes
* Logs are saved to: logs/labeling\_logs/ with per-file runtime and pipeline summary

---

## 📈 Model Versioning & Experiment Tracking

* Each training/fine-tuning run is versioned by date and experiment.
* All metrics, reports, and confusion matrices are saved for each run (see `/results_finetuned_bible/` and related directories).
* Final models for deployment are saved under `/src/fine_tuning/` after evaluation on the full train/test split.

---

## 📌 Roadmap

### ✅ Completed (Weeks 1–4)
- Full Bible scraping (KJV + RV60) and corpus organization
- Data cleaning and normalization
- Emotion and theme labeling using pretrained HuggingFace models
- Cross-lingual label transfer and Spanish label alignment
- Robust manual evaluation with accuracy, macro F1, and confusion matrix reporting
- Streamlit interface: emotion + theme detection, stylized results, and interactive recommendations
- Multilingual support: automatic input translation and dynamic corpus selection (EN/ES)
- Recommendation system matching user queries by emotion and theme
- Feedback system: like/dislike buttons with logging to Google Sheets
- Model fine-tuning workflow: train/test split, metrics, early stopping, and artifact saving
- Batch random sampling, annotation pipeline, and batch version tracking
- Cross-validation pipeline (StratifiedKFold + HuggingFace Trainer) for robust evaluation
- Automated report and confusion matrix export for each experiment

### 🔄 Week 5: Iteration Based on Feedback
- [ ] Refine model behavior and recommendation logic
- [ ] Improve clarity of explanations and label rendering
- [ ] Implement user-suggested improvements
- [ ] Integrate feedback from user testing sessions

### 🏁 Week 6: Final Demo and Documentation
- [ ] Consolidate the MVP into a cohesive narrative
- [ ] Write technical and functional report
- [ ] Prepare public demo with real examples
- [ ] (Optional) Add export features (PDF), voice synthesis, or word cloud summaries

[See CHANGELOG.md](CHANGELOG.md) for complete history.


---

## 🆕 Recent Highlights

* Cross-validation for robust metric estimation
* Automated verse sampling and annotation pipeline
* Improved label consistency and data cleaning
* Modular, reproducible scripts for all workflow stages
* Full pipeline documented in notebooks and CHANGELOG

---

## 📖 License

For academic and research use only. Sources are derived from public domain Bibles (e.g., RV60, KJV) and open ML models from HugginFace. License will be finalized before v1.0.

---

## ✨ Acknowledgements

Developed by [Manuel Cruz Rodríguez](https://github.com/mancrurod) as part of an NLP and Data Science learning journey.
