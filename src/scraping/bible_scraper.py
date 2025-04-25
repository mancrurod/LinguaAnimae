import requests
import pandas as pd
from bs4 import BeautifulSoup
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
import logging
from time import sleep
from random import uniform

# ==========================
# === CONSTANTS ============
# ==========================

# URL to fetch the HTML page containing the list of books
BASE_FORM_URL = "https://www.biblia.es/biblia-buscar-libros.php"

# URL template used to scrape the content of specific chapters by formatting book ID and chapter number
BASE_SCRAPE_URL = "https://www.biblia.es/biblia-buscar-libros-1.php?libro={}&capitulo={}&version=rv60"

# Directory where the scraped CSV files will be saved
OUTPUT_DIR = Path("data/raw/bible_rv60")

# Directory where error logs will be stored
LOG_DIR = Path("logs")

# ==========================
# === LOGGER SETUP =========
# ==========================

def setup_logger() -> logging.Logger:
    """Set up a logger to capture error logs into logs/errors_<date>.log.

    Returns:
        logging.Logger: Configured logger instance.
    """
    # Ensure that the log directory exists (create it if necessary)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Define the path for today's log file based on current date
    log_filename = LOG_DIR / f"errors_{datetime.now().strftime('%Y-%m-%d')}.log"

    # Create and configure the logger
    logger = logging.getLogger("BibleScraper")
    logger.setLevel(logging.ERROR)  # Only capture error-level messages

    # Create a file handler to write logs to the defined file
    file_handler = logging.FileHandler(log_filename, encoding="utf-8")

    # Define the format for log messages (timestamp, level, message)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Attach the file handler to the logger if not already attached
    if not logger.handlers:
        logger.addHandler(file_handler)

    return logger

# ==========================
# === SCRAPING UTILITIES ===
# ==========================

def get_all_books() -> Dict[int, tuple[str, str]]:
    """Extract all book values from the HTML <select name='libro'> element.

    Returns:
        Dict[int, tuple[str, str]]: Dictionary mapping index to (identifier, display name).
    """
    # Send a GET request to fetch the HTML content of the main page
    response = requests.get(BASE_FORM_URL)
    response.raise_for_status()

    # Parse the HTML response to find the <select> element that contains the list of books
    soup = BeautifulSoup(response.text, "html.parser")
    select = soup.find("select", {"name": "libro"})

    # Build a dictionary mapping the index to (book identifier, display name)
    # Skip the first invalid option (value="0") which is just a placeholder
    books = {
        idx: (option["value"], option.text.strip())
        for idx, option in enumerate(select.find_all("option"), start=0)
        if option["value"] != "0"
    }

    # Return the structured dictionary with all available books
    return books


def scrape_chapter(book_id: str, chapter_number: int) -> List[Dict[str, Any]]:
    """Scrape a specific chapter from a given book.

    Args:
        book_id (str): Book identifier used in URL parameters.
        chapter_number (int): Chapter number to scrape.

    Returns:
        List[Dict[str, Any]]: List of dictionaries containing verses and metadata.
    """
    # Build the chapter URL dynamically using the book ID and chapter number
    url = BASE_SCRAPE_URL.format(book_id, chapter_number)

    # Send a GET request to fetch the chapter page
    response = requests.get(url)
    response.raise_for_status()

    # Parse the HTML content of the chapter page
    soup = BeautifulSoup(response.text, "html.parser")

    # Locate the main content block containing the verses
    content_div = soup.find("div", class_="col_i_1_1_inf")
    if not content_div:
        raise ValueError(f"No content found for {book_id} chapter {chapter_number}")

    verses = []
    current_subtitle = ""  # Track the active subtitle (if any)
    current_verse = None   # Track the verse currently being built

    # Traverse through HTML elements to reconstruct verses
    for element in content_div.find_all(["h2", "span", "br"], recursive=True):
        if element.name == "h2" and "estudio" in element.get("class", []):
            # Found a subtitle (section heading)
            current_subtitle = element.get_text(strip=True)
        elif element.name == "span" and "versiculo" in element.get("class", []):
            # Found a new verse number
            current_verse = {
                "book": book_id,
                "chapter": chapter_number,
                "verse": int(element.get_text(strip=True)),
                "subtitle": current_subtitle,
                "text": "",
                "source_url": url
            }
        elif element.name == "span" and "texto" in element.get("class", []) and current_verse is not None:
            # Found verse text; append it to the current verse
            current_verse["text"] += element.get_text(strip=True) + " "
        elif element.name == "br" and current_verse:
            # Line break indicates end of the current verse
            verses.append(current_verse)
            current_verse = None

    return verses

# ==========================
# === MAIN PROCESS =========
# ==========================

def scrape_entire_bible() -> None:
    """Scrape all books and chapters of the Bible.

    Saves each book into a CSV named <index>_<identifier>.csv.
    Errors are logged to the logs/ directory.
    """
    # Ensure that the output directory exists (create it if necessary)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize the logger to capture errors
    logger = setup_logger()

    try:
        # Fetch the full list of books available for scraping
        books = get_all_books()
    except Exception as e:
        # Log the error if the book list could not be retrieved
        logger.error(f"Failed to retrieve book list: {e}")
        print("❌ Failed to load book list. Exiting process.")
        return

    # Iterate over each book by its index and identifier
    for idx, (book_id, book_name) in books.items():
        print(f"\n📖 Scraping book {idx}: {book_name} ({book_id})")
        all_verses = []
        chapter = 1

        while True:
            try:
                print(f"   🔎 Chapter {chapter}")
                # Scrape the current chapter's verses
                verses = scrape_chapter(book_id, chapter)
                
                # Stop if no verses are found (end of available chapters)
                if not verses:
                    break

                # Accumulate the scraped verses
                all_verses.extend(verses)
                chapter += 1

                # Sleep randomly to mimic human behavior and avoid bans
                sleep(uniform(1.5, 3.0))
            except Exception as e:
                # Log the error and proceed to the next book
                logger.error(f"{book_id} chapter {chapter}: {e}")
                print(f"   ❌ Error at {book_id} chapter {chapter}. Moving to next book.")
                break

        if all_verses:
            # Save the collected verses into a CSV file named by index and book ID
            df = pd.DataFrame(all_verses)
            file_path = OUTPUT_DIR / f"{idx}_{book_id}.csv"
            df.to_csv(file_path, index=False, encoding="utf-8")
            print(f"   ✅ Saved {file_path.name} with {len(df)} verses")


if __name__ == "__main__":
    scrape_entire_bible()
