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
    """
    Set up a logger for error tracking and create the log directory if needed.

    Returns:
        logging.Logger: Configured logger for the scraper.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_filename = LOG_DIR / f"errors_{datetime.now().strftime('%Y-%m-%d')}.log"
    logger = logging.getLogger("BibleScraper")
    logger.setLevel(logging.ERROR)
    if not logger.handlers:
        handler = logging.FileHandler(log_filename, encoding="utf-8")
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

# ==========================
# === HTTP WRAPPER =========
# ==========================

def fetch_url(url: str, retries: int = 3, delay_range=(1.5, 3.0), logger: Optional[logging.Logger] = None) -> Optional[str]:
    """
    Fetch a URL with retry logic and random delay between attempts.

    Args:
        url (str): The URL to fetch.
        retries (int): Number of attempts before failing.
        delay_range (tuple): Range of delay between retries.
        logger (Optional[logging.Logger]): Logger for error reporting.

    Returns:
        Optional[str]: HTML content if successful, None otherwise.
    """
    for attempt in range(retries):
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.text
        except Exception as e:
            # Log the error on the last attempt
            if logger and attempt == retries - 1:
                logger.error(f"Failed to fetch URL: {url}. Error: {e}")
            sleep(uniform(*delay_range))
    return None

# ==========================
# === SCRAPING UTILS =======
# ==========================

def get_all_books(logger: Optional[logging.Logger] = None) -> Dict[int, tuple[str, str]]:
    """
    Retrieve all Bible books from the dropdown in the base form page.

    Args:
        logger (Optional[logging.Logger]): Logger for error reporting.

    Returns:
        Dict[int, tuple[str, str]]: Mapping from index to (book_id, book_name).

    Raises:
        ValueError: If the expected dropdown is not found.
    """
    html = fetch_url(BASE_FORM_URL, logger=logger)
    if not html:
        msg = "Failed to fetch the main form page. No HTML returned."
        if logger:
            logger.error(msg)
        raise ValueError(msg)
    soup = BeautifulSoup(html, "html.parser")
    select = soup.find("select", {"name": "libro"})
    if not select:
        msg = "<select name='libro'> not found in the HTML page."
        if logger:
            logger.error(msg)
        raise ValueError(msg)
    books = {
        idx: (option["value"], option.text.strip())
        for idx, option in enumerate(select.find_all("option"), start=0)
        if option["value"] != "0"
    }
    if not books:
        msg = "No books found in the dropdown. Possible site change."
        if logger:
            logger.error(msg)
        raise ValueError(msg)
    return books

def scrape_chapter(book_id: str, chapter_number: int, logger: Optional[logging.Logger] = None) -> List[Dict[str, Any]]:
    """
    Scrape all verses from a specific chapter of a book.

    Args:
        book_id (str): ID of the Bible book.
        chapter_number (int): Chapter number to scrape.
        logger (Optional[logging.Logger]): Logger for error reporting.

    Returns:
        List[Dict[str, Any]]: List of verse dictionaries.

    Raises:
        ValueError: If content is missing for the chapter.
    """
    url = BASE_SCRAPE_URL.format(book_id, chapter_number)
    html = fetch_url(url, logger=logger)
    if not html:
        msg = f"Failed to fetch chapter URL: {url}"
        if logger:
            logger.error(msg)
        raise ValueError(msg)
    soup = BeautifulSoup(html, "html.parser")
    content_div = soup.find("div", class_="col_i_1_1_inf")
    if not content_div:
        msg = f"No content found for {book_id} chapter {chapter_number}. Site structure may have changed."
        if logger:
            logger.error(msg)
        raise ValueError(msg)

    verses, subtitle, current_verse = [], "", None
    for el in content_div.find_all(["h2", "span", "br"], recursive=True):
        if el.name == "h2" and "estudio" in el.get("class", []):
            subtitle = el.get_text(strip=True)
        elif el.name == "span" and "versiculo" in el.get("class", []):
            try:
                verse_num = int(el.get_text(strip=True))
            except ValueError:
                # Skip if not a valid integer, but log the occurrence
                if logger:
                    logger.error(f"Non-integer verse number in {book_id} chapter {chapter_number}: {el.get_text(strip=True)}")
                continue
            current_verse = {
                "book": book_id,
                "chapter": chapter_number,
                "verse": verse_num,
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
            if current_verse["text"]:  # Only append if verse text is not empty
                verses.append(current_verse)
            else:
                if logger:
                    logger.error(f"Empty verse in {book_id} chapter {chapter_number} verse {current_verse['verse']}")
            current_verse = None

    if current_verse:
        current_verse["text"] = current_verse["text"].strip()
        if current_verse["text"]:
            verses.append(current_verse)
        else:
            if logger:
                logger.error(f"Last verse is empty in {book_id} chapter {chapter_number} verse {current_verse['verse']}")

    return verses

def scrape_book(idx: int, book_id: str, book_name: str, logger: logging.Logger, max_chapters: Optional[int] = None) -> None:
    """
    Scrape all chapters for a given book, handling errors robustly.

    Args:
        idx (int): Index of the book.
        book_id (str): ID of the book.
        book_name (str): Name of the book.
        logger (logging.Logger): Logger for error reporting.
        max_chapters (Optional[int]): Maximum chapters to scrape (for testing).
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    filepath = OUTPUT_DIR / f"{idx}_{book_id}.csv"
    if filepath.exists():
        print(f"⏭️ Skipping {book_name} (already scraped)")
        return

    all_verses = []
    def chapter_task(chapter):
        try:
            return scrape_chapter(book_id, chapter, logger=logger)
        except Exception as e:
            logger.error(f"{book_id} chapter {chapter}: {e}")
            return []

    chapter = 1
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        failures = 0
        while True:
            if max_chapters and chapter > max_chapters:
                break
            futures.append(executor.submit(chapter_task, chapter))
            chapter += 1
            sleep(uniform(0.1, 0.3))
            # Break when soft cap is reached or after several consecutive failures (site may have ended chapters)
            if chapter > 50:
                break

        for future in tqdm(as_completed(futures), total=len(futures), desc=f"📖 {book_name}"):
            try:
                verses = future.result()
                if verses:
                    all_verses.extend(verses)
                else:
                    failures += 1
                    if failures > 3:  # If several consecutive chapters fail, likely reached end
                        break
            except Exception as e:
                logger.error(f"Error processing future: {e}")

    if all_verses:
        df = pd.DataFrame(all_verses)
        df.sort_values(by=["chapter", "verse"], inplace=True)
        df.to_csv(filepath, index=False, encoding="utf-8")
        print(f"✅ Saved {filepath.name} with {len(df)} verses")
    else:
        logger.error(f"No verses scraped for {book_name} (book_id {book_id}).")

# ==========================
# === MAIN LOGIC ===========
# ==========================

def main(selected_books: Optional[List[int]] = None, max_chapters: Optional[int] = None):
    """
    Main entrypoint: scrape selected books and handle top-level errors.

    Args:
        selected_books (Optional[List[int]]): Indices of books to scrape.
        max_chapters (Optional[int]): Maximum chapters per book (for testing).
    """
    logger = setup_logger()

    try:
        books = get_all_books(logger=logger)
    except Exception as e:
        logger.error(f"Book list retrieval failed: {e}")
        print("❌ Could not fetch book list.")
        return

    for idx, (book_id, book_name) in books.items():
        if selected_books and idx not in selected_books:
            continue
        print(f"\n🔍 Processing book {idx}: {book_name}")
        try:
            scrape_book(idx, book_id, book_name, logger, max_chapters)
        except Exception as e:
            logger.error(f"Failed to scrape book {book_name} ({book_id}): {e}")
            print(f"❌ Failed to scrape {book_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="📘 Spanish Bible Scraper (RV60)")
    parser.add_argument("--books", type=str, help="Comma-separated list of book indices to scrape")
    parser.add_argument("--chapters", type=int, help="Max chapters per book to scrape (for testing)")
    args = parser.parse_args()

    selected = [int(i.strip()) for i in args.books.split(",")] if args.books else None
    main(selected_books=selected, max_chapters=args.chapters)
