import requests
from bs4 import BeautifulSoup
import pandas as pd
from time import sleep
from random import uniform
from pathlib import Path

BASE_URL = "https://www.biblia.es/biblia-buscar-libros-1.php?libro=juan&capitulo={}&version=rv60"


def scrape_chapter(chapter_number: int) -> list[dict]:
    url = BASE_URL.format(chapter_number)
    response = requests.get(url)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    content_div = soup.find("div", class_="col_i_1_1_inf")

    verses = []
    current_subtitle = ""
    current_verse = None

    for element in content_div.find_all(["h2", "span", "br"], recursive=True):
        if element.name == "h2" and "estudio" in element.get("class", []):
            current_subtitle = element.get_text(strip=True)

        elif element.name == "span" and "versiculo" in element.get("class", []):
            current_verse = {
                "chapter": chapter_number,
                "verse": int(element.get_text(strip=True)),
                "subtitle": current_subtitle,
                "text": "",
                "source_url": url
            }

        elif element.name == "span" and "texto" in element.get("class", []) and current_verse is not None:
            if current_verse["text"]:
                current_verse["text"] += " "
            current_verse["text"] += element.get_text(strip=True)

        elif element.name == "br" and current_verse:
            verses.append(current_verse)
            current_verse = None

    return verses

def scrape_gospel_of_john(save_path: str = "data/raw/john_gospel_rv60.csv"):
    all_verses = []
    for chapter in range(1, 22):
        print(f"Scraping chapter {chapter}...")
        try:
            verses = scrape_chapter(chapter)
            all_verses.extend(verses)
            sleep(uniform(1.5, 3.0))
        except Exception as e:
            print(f"Error in {chapter}: {e}")
            continue

    df = pd.DataFrame(all_verses)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False, encoding="utf-8")
    print(f"Scraping completed. File saved: {save_path}")


if __name__ == "__main__":
    scrape_gospel_of_john()
