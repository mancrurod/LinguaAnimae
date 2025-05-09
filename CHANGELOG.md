# 📜 CHANGELOG

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [0.4.0] - 2025-05-09

### Added
- `render_theme.py`: Visual component matching the emotion renderer, with color, icon and multilingual support.
- Theme and emotion block rendering fully integrated into `app.py` with consistent style and accessibility.
- `recommender.py`: Support for corpus selection based on interface language (`bible_kjv` or `bible_rv60`).
- Label translation logic in recommendation filtering via `EMOTION_MAP` and `THEME_MAP` to ensure Spanish corpus compatibility.

### Changed
- Streamlit accessibility warnings resolved by setting `label_visibility="collapsed"` on empty labels.
- Improved `load_corpus()` to load from the appropriate directory depending on the current language.
- `recommend_verses()` now supports dynamic filtering by language and auto-translation of query labels.
- Updated `05_evaluation.ipynb` with a final summary and refined conclusions based on model performance and cross-lingual alignment.

### Fixed
- Prevented `KeyError` in `render_theme_block()` when receiving already-translated labels.
- Ensured consistent theme rendering regardless of input language.
- Fixed mismatch between label language and corpus filtering logic that prevented Spanish recommendations from working.

---

## [0.3.0] - 2025-05-01

### Added
- `labeling_pipeline.py`: Unified emotion + theme classification runner with CLI support, GPU detection, logging, and skip flags.
- `emotion_theme_labeling.py` and `theme_labeling.py`: Modular labelers supporting HuggingFace pipelines and batch GPU processing.
- `translate_and_apply_labels.py`: Maps English-labeled files into Spanish using custom theme/emotion translations and merges with `bible_rv60`.
- `04_translate_labels.ipynb`: Notebook demonstrating cross-language label alignment.
- `05_evaluation.ipynb`: Validates Spanish label quality against English annotations using exact match and theme overlap metrics.
- Per-file runtime logging to `logs/labeling_logs/` with timestamps and total summaries.

### Changed
- Folder structure aligned: `data/labeled/<bible_name>/emotion/` and `emotion_theme/` created per Bible version.
- Replaced deprecated `annotated/` folder with `labeled/`.
- All labeling scripts updated to write `_emotion.csv` and `_emotion_theme.csv` outputs.
- Git commit logic standardized using conventional commits + emoji tags.
- `.gitignore` updated to exclude `__init__.py` files by default.

### Documentation
- Docstrings and inline comments added to all labeling and translation scripts for clarity.
- Expanded README instructions for GPU setup and script execution.

---

## [0.2.0] - 2025-04-25
### Changed
- `bible_scraper.py` now automatically extracts all books and chapters from the Reina-Valera 1960 Bible.
- Outputs structured CSV files named `<index>_<book>.csv`.
- Implemented error logging system into `logs/` with timestamped error logs.
- Improved scraping logic for robustness against missing or broken chapters.

---

## [0.1.0] - 2025-04-21
### Added
- Initial project structure: `data/`, `src/`, `notebooks/`, `tests/`.
- `bible_scraper.py` to extract the Gospel of John (RV60) from biblia.es.
- CSV output format including chapter, verse, subtitle, text, and source_url.
- `01_scraping_exploration.ipynb` for data validation and visualization.
- Custom README with Conda and pip instructions.
- Basic project metadata for GitHub setup.

---

## [Unreleased]
### Planned
- Streamlit app with a chatbot interface to retrieve Bible verses aligned with user input.
  - Input: free-form user prompt
  - Output: verses ranked by emotion, theme, and semantic similarity
- Multilingual support: English and Spanish queries using labeled corpora (`bible_kjv`, `bible_rv60`)
- Vector search integration using embeddings from sentence-transformers or HuggingFace models
- DuckDB or SQLite integration for fast, local verse retrieval
- Session-based logging to capture feedback and suggested improvements

