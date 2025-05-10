import requests
import pandas as pd
from bs4 import BeautifulSoup
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
import logging
from time import sleep
from random import uniform
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import argparse
import sys

# ==========================
# === CONSTANTS ============
# ==========================

# URL for the form to fetch all books
BASE_FORM_URL = "https://www.biblia.es/biblia-buscar-libros.php"
# URL template for scraping a specific chapter
BASE_SCRAPE_URL = "https://www.biblia.es/biblia-buscar-libros-1.php?libro={}&capitulo={}&version=rv60"
# Directory to save the scraped data
OUTPUT_DIR = Path("data/raw/bible_rv60")
# Directory to save logs
LOG_DIR = Path("logs")

# ==========================
# === LOGGER SETUP =========
# ==========================

def setup_logger() -> logging.Logger:
    # Ensure the log directory exists
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    # Create a log file with the current date
    log_filename = LOG_DIR / f"errors_{datetime.now().strftime('%Y-%m-%d')}.log"
    logger = logging.getLogger("BibleScraper")
    logger.setLevel(logging.ERROR)
    if not logger.handlers:
        # Set up a file handler for logging errors
        handler = logging.FileHandler(log_filename, encoding="utf-8")
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

# ==========================
# === HTTP WRAPPER =========
# ==========================

def fetch_url(url: str, retries: int = 3, delay_range=(1.5, 3.0)) -> Optional[str]:
    # Attempt to fetch the URL with retries
    for attempt in range(retries):
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise an exception for HTTP errors
            return response.text
        except Exception as e:
            # Wait for a random delay before retrying
            sleep(uniform(*delay_range))
            if attempt == retries - 1:
                raise e  # Raise the exception if all retries fail
    return None

# ==========================
# === SCRAPING UTILS =======
# ==========================

def get_all_books() -> Dict[int, tuple[str, str]]:
    # Fetch the HTML of the form page
    html = fetch_url(BASE_FORM_URL)
    soup = BeautifulSoup(html, "html.parser")
    # Find the dropdown containing the list of books
    select = soup.find("select", {"name": "libro"})
    # Parse the books into a dictionary
    books = {
        idx: (option["value"], option.text.strip())
        for idx, option in enumerate(select.find_all("option"), start=0)
        if option["value"] != "0"  # Exclude invalid options
    }
    return books

def scrape_chapter(book_id: str, chapter_number: int) -> List[Dict[str, Any]]:
    # Construct the URL for the specific chapter
    url = BASE_SCRAPE_URL.format(book_id, chapter_number)
    html = fetch_url(url)
    soup = BeautifulSoup(html, "html.parser")
    # Find the content division containing the verses
    content_div = soup.find("div", class_="col_i_1_1_inf")
    if not content_div:
        raise ValueError(f"No content found for {book_id} chapter {chapter_number}")

    verses, subtitle, current_verse = [], "", None
    # Iterate through the elements in the content division
    for el in content_div.find_all(["h2", "span", "br"], recursive=True):
        if el.name == "h2" and "estudio" in el.get("class", []):
            subtitle = el.get_text(strip=True)
        elif el.name == "span" and "versiculo" in el.get("class", []):
            current_verse = {
                "book": book_id,
                "chapter": chapter_number,
                "verse": int(el.get_text(strip=True)),
                "subtitle": subtitle,
                "text": "",
                "source_url": url
            }
        elif el.name == "span" and "texto" in el.get("class", []) and current_verse:
            text_part = el.get_text().strip()
            if text_part:
                current_verse["text"] += text_part + "\n"
        elif el.name == "br" and current_verse:
            current_verse["text"] = current_verse["text"].strip()
            verses.append(current_verse)
            current_verse = None

    # If the last verse wasn't closed with <br>, make sure we still include it
    if current_verse:
        current_verse["text"] = current_verse["text"].strip()
        verses.append(current_verse)

    return verses

# ==========================
# === MAIN LOGIC ===========
# ==========================

def scrape_book(idx: int, book_id: str, book_name: str, logger: logging.Logger, max_chapters: Optional[int] = None) -> None:
    # Ensure the output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Define the output file path
    filepath = OUTPUT_DIR / f"{idx}_{book_id}.csv"
    if filepath.exists():
        # Skip if the book has already been scraped
        print(f"⏭️ Skipping {book_name} (already scraped)")
        return

    all_verses = []

    def chapter_task(chapter):
        # Task to scrape a single chapter
        try:
            verses = scrape_chapter(book_id, chapter)
            return verses
        except Exception as e:
            # Log errors for failed chapters
            logger.error(f"{book_id} chapter {chapter}: {e}")
            return []

    chapter = 1
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        while True:
            if max_chapters and chapter > max_chapters:
                break  # Stop if the max chapter limit is reached
            futures.append(executor.submit(chapter_task, chapter))
            chapter += 1
            sleep(uniform(0.1, 0.3))  # Throttle tasks

            # Break when no future returns content
            if chapter > 50:  # Soft cap to avoid infinite loops
                break

        # Process completed futures
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"📖 {book_name}"):
            verses = future.result()
            if verses:
                all_verses.extend(verses)

    if all_verses:
        # Save the scraped verses to a CSV file
        df = pd.DataFrame(all_verses)
        df.sort_values(by=["chapter", "verse"], inplace=True)
        df.to_csv(filepath, index=False, encoding="utf-8")
        print(f"✅ Saved {filepath.name} with {len(df)} verses")

def main(selected_books: Optional[List[int]] = None, max_chapters: Optional[int] = None):
    # Set up the logger
    logger = setup_logger()

    try:
        # Fetch the list of books
        books = get_all_books()
    except Exception as e:
        # Log and print an error if book retrieval fails
        logger.error(f"Book list retrieval failed: {e}")
        print("❌ Could not fetch book list.")
        return

    # Iterate through the books and scrape the selected ones
    for idx, (book_id, book_name) in books.items():
        if selected_books and idx not in selected_books:
            continue  # Skip books not in the selected list
        print(f"\n🔍 Processing book {idx}: {book_name}")
        scrape_book(idx, book_id, book_name, logger, max_chapters)

# ==========================
# === CLI WRAPPER ==========
# ==========================

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="📘 Spanish Bible Scraper (RV60)")
    parser.add_argument("--books", type=str, help="Comma-separated list of book indices to scrape")
    parser.add_argument("--chapters", type=int, help="Max chapters per book to scrape (for testing)")
    args = parser.parse_args()

    # Parse the selected books and chapters
    selected = [int(i.strip()) for i in args.books.split(",")] if args.books else None
    main(selected_books=selected, max_chapters=args.chapters)
