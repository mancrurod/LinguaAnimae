# 📜 CHANGELOG

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

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


## [0.2.0] - 2025-04-25
### Changed
- `bible_scraper.py` now automatically extracts all books and chapters from the Reina-Valera 1960 Bible.
- Outputs structured CSV files named `<index>_<book>.csv`.
- Implemented error logging system into `logs/` with timestamped error logs.
- Improved scraping logic for robustness against missing or broken chapters.


## [Unreleased]
### Planned
- `02_preprocessing_pipeline.ipynb` for cleaning and normalization.
- Manual or semi-automatic annotation tools for themes/emotions.
- Classifier training scripts using Transformers.
- Streamlit interface for real-time exploration.
- PostgreSQL integration and dashboard connectivity.
