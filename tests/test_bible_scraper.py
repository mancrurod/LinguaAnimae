"""
Unit tests for bible_scraper.py in Lingua Animae.

These tests cover core logic and error handling for network and HTML parsing utilities,
using mocks to avoid external web requests.

Covers:
- HTTP retry logic (fetch_url)
- Parsing of book dropdown (get_all_books)
- Parsing of chapter HTML (scrape_chapter), including verse extraction and edge cases

Usage:
pytest tests/test_bible_scraper.py
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from scraping.bible_scraper import fetch_url, get_all_books, scrape_chapter

class DummyLogger:
    def error(self, *args, **kwargs): pass

def test_fetch_url_success(monkeypatch):
    # Simula respuesta exitosa
    def fake_get(url):
        class Response:
            def raise_for_status(self): pass
            text = "<html>OK</html>"
        return Response()
    monkeypatch.setattr("requests.get", fake_get)
    result = fetch_url("http://fake", logger=DummyLogger())
    assert result == "<html>OK</html>"

def test_fetch_url_fail(monkeypatch):
    # Simula fallo, debe devolver None tras reintentos
    def fake_get(url):
        raise Exception("Network error")
    monkeypatch.setattr("requests.get", fake_get)
    result = fetch_url("http://fake", retries=2, logger=DummyLogger())
    assert result is None

def test_get_all_books_success(monkeypatch):
    # Simula HTML de dropdown de libros
    fake_html = '''
    <select name="libro">
      <option value="0">Seleccione libro</option>
      <option value="GEN">Génesis</option>
      <option value="EXO">Éxodo</option>
    </select>
    '''
    monkeypatch.setattr("scraping.bible_scraper.fetch_url", lambda url, **kwargs: fake_html)
    books = get_all_books(logger=DummyLogger())
    # El valor 0 se ignora, solo 2 libros reales
    assert books == {
        1: ("GEN", "Génesis"),
        2: ("EXO", "Éxodo")
    }

def test_get_all_books_no_dropdown(monkeypatch):
    # Simula HTML sin select
    monkeypatch.setattr("scraping.bible_scraper.fetch_url", lambda url, **kwargs: "<html></html>")
    with pytest.raises(ValueError):
        get_all_books(logger=DummyLogger())

def test_scrape_chapter_basic(monkeypatch):
    # Simula HTML de un capítulo con dos versículos
    fake_html = '''
    <div class="col_i_1_1_inf">
      <span class="versiculo">1</span><span class="texto">En el principio...</span><br/>
      <span class="versiculo">2</span><span class="texto">Y la tierra estaba...</span><br/>
    </div>
    '''
    monkeypatch.setattr("scraping.bible_scraper.fetch_url", lambda url, **kwargs: fake_html)
    verses = scrape_chapter("GEN", 1, logger=DummyLogger())
    assert len(verses) == 2
    assert verses[0]["book"] == "GEN"
    assert verses[0]["chapter"] == 1
    assert verses[0]["verse"] == 1
    assert "principio" in verses[0]["text"]
    assert verses[1]["verse"] == 2

def test_scrape_chapter_missing_content(monkeypatch):
    # Simula capítulo sin div esperado
    fake_html = "<html><body>Sin contenido relevante</body></html>"
    monkeypatch.setattr("scraping.bible_scraper.fetch_url", lambda url, **kwargs: fake_html)
    with pytest.raises(ValueError):
        scrape_chapter("GEN", 1, logger=DummyLogger())

def test_scrape_chapter_non_integer_verse(monkeypatch):
    # Simula un versículo con número no entero
    fake_html = '''
    <div class="col_i_1_1_inf">
      <span class="versiculo">A</span><span class="texto">Texto raro...</span><br/>
    </div>
    '''
    monkeypatch.setattr("scraping.bible_scraper.fetch_url", lambda url, **kwargs: fake_html)
    verses = scrape_chapter("GEN", 1, logger=DummyLogger())
    # El versículo inválido no debe estar en el output
    assert verses == []
