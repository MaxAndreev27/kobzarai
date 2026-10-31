"""Збір творів Т. Г. Шевченка з ukrlib.com.ua у data/processed/corpus.jsonl."""

import json
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup, Comment

BASE = "https://www.ukrlib.com.ua/books/"
AUTHOR_ID = 1
ENCODING = "windows-1251"
HEADERS = {"User-Agent": "kobzarai/0.1 (educational project)"}
DELAY = 1.5

ML_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = ML_DIR / "data" / "raw" / "ukrlib"
OUT_PATH = ML_DIR / "data" / "processed" / "corpus.jsonl"

session = requests.Session()
session.headers.update(HEADERS)


def fetch(url: str, cache: Path) -> str:
    """Скачує сторінку з кешем на диску і повторами при помилках."""
    if cache.exists():
        return cache.read_text(encoding="utf-8")
    for attempt in range(4):
        try:
            resp = session.get(url, timeout=30)
            resp.raise_for_status()
            break
        except requests.RequestException:
            if attempt == 3:
                raise
            time.sleep(5 * (attempt + 1))
    html = resp.content.decode(ENCODING, errors="replace")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(html, encoding="utf-8")
    time.sleep(DELAY)
    return html


def get_work_links() -> list[tuple[int, str]]:
    """Повертає (tid, назва) усіх творів автора з усіх сторінок списку."""
    works: dict[int, str] = {}
    page = 1
    while True:
        url = f"{BASE}author.php?id={AUTHOR_ID}&page={page}"
        soup = BeautifulSoup(fetch(url, RAW_DIR / f"list_{page}.html"), "html.parser")
        items = soup.select("div.content ul.list a[href*='printit.php?tid=']")
        new = [
            a
            for a in items
            if int(re.search(r"tid=(\d+)", a["href"]).group(1)) not in works
        ]
        if not new:
            break
        for a in new:
            tid = int(re.search(r"tid=(\d+)", a["href"]).group(1))
            works[tid] = a.get_text(strip=True)
        page += 1
    return list(works.items())


def clean_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = "\n".join(line.strip() for line in text.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def parse_work(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    article = soup.find(id="content")
    if article is None:
        return ""
    for tag in article.find_all(["script", "ins", "style", "div", "img"]):
        tag.decompose()
    # у HTML після кожного <br> стоїть службовий перенос рядка, його треба прибрати
    for node in article.find_all(string=True):
        if isinstance(node, Comment):
            node.extract()
        else:
            node.replace_with(node.replace("\r", "").replace("\n", ""))
    for br in article.find_all("br"):
        br.replace_with("\n")
    for para in article.find_all("p"):
        para.append("\n")
        para.unwrap()
    return clean_text(article.get_text())


def fetch_work_text(tid: int) -> str:
    """Збирає текст твору з усіх сторінок пагінації."""
    parts = []
    page = 1
    while True:
        suffix = "" if page == 1 else f"&page={page}"
        cache = RAW_DIR / (f"{tid}.html" if page == 1 else f"{tid}_p{page}.html")
        html = fetch(f"{BASE}printit.php?tid={tid}{suffix}", cache)
        parts.append(parse_work(html))
        if (
            BeautifulSoup(html, "html.parser").select_one("div.paginator a.next")
            is None
        ):
            break
        page += 1
    return clean_text("\n".join(p for p in parts if p))


def main() -> None:
    links = get_work_links()
    print(f"Знайдено творів: {len(links)}")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    saved = 0
    with OUT_PATH.open("w", encoding="utf-8") as out:
        for i, (tid, title) in enumerate(links, 1):
            text = fetch_work_text(tid)
            if not text:
                print(f"[{i}/{len(links)}] порожній текст: {title} ({tid})")
                continue
            record = {
                "id": tid,
                "title": title,
                "source": f"{BASE}printit.php?tid={tid}",
                "text": text,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            saved += 1
            if i % 25 == 0:
                print(f"[{i}/{len(links)}]")
    print(f"Збережено: {saved} -> {OUT_PATH}")


if __name__ == "__main__":
    main()
