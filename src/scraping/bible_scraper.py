import requests
from bs4 import BeautifulSoup
import pandas as pd
from time import sleep
from random import uniform
from pathlib import Path
from datetime import datetime
import logging

# Constants
BASE_FORM_URL = "https://www.biblia.es/biblia-buscar-libros.php"
BASE_SCRAPE_URL = "https://www.biblia.es/biblia-buscar-libros-1.php?libro={}&capitulo={}&version=rv60"
OUTPUT_DIR = Path("data/raw/bible_rv60")
LOG_DIR = Path("logs")

def setup_logger() -> logging.Logger:
    """
    Configure the logger to write error logs into logs/errors_<date>.log.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_filename = LOG_DIR / f"errors_{datetime.now().strftime('%Y-%m-%d')}.log"

    logger = logging.getLogger("BibleScraper")
    logger.setLevel(logging.ERROR)

    file_handler = logging.FileHandler(log_filename, encoding="utf-8")
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(file_handler)

    return logger

def get_all_books() -> dict[int, tuple[str, str]]:
    """
    Extracts the ordered list of books from the <select name="libro"> element.
    Returns a dictionary like {1: ("genesis", "Génesis"), 2: ("exodo", "Éxodo"), ...}
    """
    response = requests.get(BASE_FORM_URL)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    select = soup.find("select", {"name": "libro"})

    book_options = select.find_all("option")
    books = {
        idx: (option["value"], option.text.strip())
        for idx, option in enumerate(book_options, start=0)
        if option["value"] != "0"
    }
    return books

def scrape_chapter(book_id: str, chapter_number: int) -> list[dict]:
    """
    Scrapes a single chapter of a given book.
    Returns a list of dictionaries with verse data.
    """
    url = BASE_SCRAPE_URL.format(book_id, chapter_number)
    response = requests.get(url)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    content_div = soup.find("div", class_="col_i_1_1_inf")
    if not content_div:
        raise ValueError(f"No content found for {book_id} chapter {chapter_number}")

    verses = []
    current_subtitle = ""
    current_verse = None

    for element in content_div.find_all(["h2", "span", "br"], recursive=True):
        if element.name == "h2" and "estudio" in element.get("class", []):
            current_subtitle = element.get_text(strip=True)
        elif element.name == "span" and "versiculo" in element.get("class", []):
            current_verse = {
                "book": book_id,
                "chapter": chapter_number,
                "verse": int(element.get_text(strip=True)),
                "subtitle": current_subtitle,
                "text": "",
                "source_url": url
            }
        elif element.name == "span" and "texto" in element.get("class", []) and current_verse is not None:
            current_verse["text"] += element.get_text(strip=True) + " "
        elif element.name == "br" and current_verse:
            verses.append(current_verse)
            current_verse = None

    return verses

def scrape_entire_bible():
    """
    Scrapes all books and their chapters. Each book is saved to a file named <index>_<book>.csv.
    Logs errors during the process.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logger()

    try:
        books = get_all_books()
    except Exception as e:
        logger.error(f"Failed to retrieve book list: {e}")
        return

    for idx, (book_id, book_name) in books.items():
        print(f"\nScraping book: {book_name} ({book_id})")
        all_verses = []
        chapter = 1

        while True:
            try:
                print(f"  Chapter {chapter}")
                verses = scrape_chapter(book_id, chapter)
                if not verses:
                    break
                all_verses.extend(verses)
                chapter += 1
                sleep(uniform(1.5, 3.0))
            except Exception as e:
                logger.error(f"{book_id} chapter {chapter}: {e}")
                print(f"  ✖ Error in {book_id} chapter {chapter}. Moving to next book.")
                break

        if all_verses:
            filename = OUTPUT_DIR / f"{idx}_{book_id}.csv"
            df = pd.DataFrame(all_verses)
            df.to_csv(filename, index=False, encoding="utf-8")
            print(f"  ✔ Saved {filename.name} with {len(df)} verses")

if __name__ == "__main__":
    scrape_entire_bible()
