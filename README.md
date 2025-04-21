# 🕊️ LinguaAnimae

**LinguaAnimae** is an NLP-based project designed to classify and explore sacred texts, classical literature, and lyrical content by their underlying **themes** and **emotional resonance**.

The project begins with the Gospel of John (Reina-Valera 1960 translation), aiming to build a pipeline that can scale to multiple textual domains: spiritual writings, poetry, and even music lyrics.

---

## 🔍 Project Goals

- Extract and structure textual corpora from sacred or literary sources.
- Annotate verses with **thematic** and **emotional** labels (manually or automatically).
- Train classifiers to predict these categories.
- Build an interface for **interactive search**, **recommendation**, and **interpretation**.
- Support multiple input/output formats (text, audio, HTML).

---

## 🧠 Technologies

- **Python 3.10+**
- `pandas`, `seaborn`, `matplotlib`
- `beautifulsoup4`, `requests`
- `scikit-learn`, `transformers` (planned)
- `streamlit` (for the frontend)
- `PostgreSQL` or `SQLite` (for structured storage)

---

## 📁 Project Structure

```
LinguaAnimae/
├── data/                 # Raw, processed, and annotated corpora
│   ├── raw/
│   ├── processed/
│   └── annotated/
├── notebooks/            # Exploratory analysis and visualizations
│   └── 01_scraping_exploration.ipynb
├── src/                  # Modular pipeline code
│   ├── scraping/
│   ├── preprocessing/
│   ├── modeling/
│   ├── interface/
│   └── utils/
├── tests/                # Unit tests (to be implemented)
├── .env                  # Environment variables (ignored)
├── .gitignore
├── requirements.txt
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

4. Run the scraper (optional first step)
```bash
python src/scraping/bible_scraper.py
```


### 1. Clone the repository
```bash
git clone https://github.com/your-username/LinguaAnimae.git
cd LinguaAnimae
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the scraper (optional first step)
```bash
python src/scraping/bible_scraper.py
```

---

## 📊 Example Output

- 📘 `john_gospel_rv60.csv`: 879 verses, structured with chapter, verse, subtitle, text and source URL.
- 📈 Visualizations: verse count per chapter, average verse length, subtitle distribution.
- 🧾 Planned outputs: semantic embeddings, predicted categories, recommendations.

---

## 📌 Future Features

- Thematic & emotional annotation interface
- HuggingFace models for multi-label classification
- Streamlit app for real-time verse suggestion
- Expansion to poetry and song lyrics
- Semantic search with vector embeddings

---

## 📖 License

This project is for academic and educational purposes. It uses public domain content and follows fair use guidelines. A formal license will be added upon stabilization.

---

## ✨ Acknowledgements

Developed by [Manuel Cruz Rodríguez](https://github.com/mancrurod) as part of an NLP and Data Science learning journey.
