"""Відбір українськомовних віршів і поем із corpus.jsonl у poems_uk.jsonl."""

import json
import re
from pathlib import Path

ML_DIR = Path(__file__).resolve().parents[2]
IN_PATH = ML_DIR / "data" / "processed" / "corpus.jsonl"
OUT_PATH = ML_DIR / "data" / "processed" / "poems_uk.jsonl"

COLLECTION_TITLE = "Кобзар"
REST_TITLE = "Кобзар (вірші без окремої сторінки)"
MAX_VERSE_LINE = 110  # довші рядки у віршах — це редакційні примітки
MIN_SHORT_SHARE = 0.9  # частка коротких рядків, за якої текст вважаємо віршем
MAX_RU_PER_1000 = 5  # літери ы/э/ъ/ё на 1000 літер: ознака російського тексту
MIN_UK_PER_1000 = 15  # літери і/ї/є/ґ на 1000 літер: ознака українського тексту
MIN_LINES = 4
MIN_DEDUP_LINE = 15  # коротші рядки (рефрени, «Амінь») не вважаємо дублікатами
MAX_CONTAINED = 0.8  # частка рядків тексту, за якої він вважається частиною іншого
MIN_CONTAINED_LINES = 5  # менше рядків для порівняння недостатньо
MIN_REST_RUN = 8  # мінімум рядків в уривку збірника, якого немає в окремих творах
FIRST_POEM = "ПРИЧИННА"  # заголовок першого вірша збірника: до нього йде зміст
MIN_CHARS = 150  # коротші уривки не дають достатньо матеріалу для стилю
HEAD_LINES = 4  # у скількох перших рядках шукаємо жанрову позначку та присвяту
TAIL_LINES = 2  # у скількох останніх рядках шукаємо дату і місце написання
MAX_META_LINE = 80  # довші рядки не вважаємо службовими
YEAR_RE = re.compile(r"\b1[78]\d\d\b")
GENRE_RE = re.compile(r"^\(?(поема|містерія|комедія|драма)\.?\)?$", re.IGNORECASE)
DEDICATION_RE = re.compile(r"^(на пам'?ять|.*\bна память)\b", re.IGNORECASE)
NOTE_RE = re.compile(r"^\((?:в автографі|в рукопису|уривок)[^)]*\)$", re.IGNORECASE)
POEM_MIN_LINES = 150  # без жанрової позначки довгі тексти вважаємо поемами


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def non_empty_lines(text: str) -> list[str]:
    return [line.strip() for line in text.split("\n") if line.strip()]


def letter_rate(text: str, pattern: str) -> float:
    letters = len(re.findall(r"[а-яіїєґё]", text, re.IGNORECASE)) or 1
    return len(re.findall(pattern, text)) / letters * 1000


def is_ukrainian(text: str) -> bool:
    return (
        letter_rate(text, r"[ыэъёЫЭЪЁ]") < MAX_RU_PER_1000
        and letter_rate(text, r"[іїєґІЇЄҐ]") >= MIN_UK_PER_1000
    )


def is_verse(text: str) -> bool:
    lines = non_empty_lines(text)
    if len(lines) < MIN_LINES:
        return False
    short = sum(len(line) <= MAX_VERSE_LINE for line in lines)
    return short / len(lines) >= MIN_SHORT_SHARE


def strip_notes(text: str) -> str:
    """Прибирає примітки: довгі рядки та рядки зі зірочкою на початку."""
    kept = []
    for line in text.split("\n"):
        s = line.strip()
        if len(s) > MAX_VERSE_LINE or s.startswith("*"):
            continue
        kept.append(s)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()


def extract_meta(text: str) -> tuple[str, dict]:
    """Виносить службові рядки (жанр, присвяту, дату і місце) з тексту в метадані."""
    lines = text.split("\n")
    meta: dict = {"genre": None, "dedication": None, "date_note": None, "year": None}

    def indices(rng):
        return [i for i in rng if lines[i].strip()]

    head = indices(range(min(HEAD_LINES, len(lines))))
    drop = set()
    for i in head:
        line = lines[i].strip()
        if len(line) > MAX_META_LINE:
            continue
        if m := GENRE_RE.match(line):
            meta["genre"] = m.group(1).lower()
            drop.add(i)
        elif DEDICATION_RE.match(line):
            meta["dedication"] = line
            drop.add(i)
            # адресат стоїть окремим рядком перед рядком, що починається з «на …»
            prev = i - 1
            if (
                line.startswith("на ")
                and prev >= 0
                and prev not in drop
                and lines[prev].strip()
            ):
                meta["dedication"] = f"{lines[prev].strip()} {line}"
                drop.add(prev)
        elif NOTE_RE.match(line):
            drop.add(i)

    tail = indices(range(len(lines) - 1, -1, -1))[:TAIL_LINES]
    for i in tail:
        line = lines[i].strip()
        if len(line) <= MAX_META_LINE and YEAR_RE.search(line):
            meta["date_note"] = line
            drop.add(i)
            prev = i - 1
            while prev >= 0 and not lines[prev].strip():
                prev -= 1
            # дата розірвана на два рядки: «[Кос-Арал» і «1849]»
            if (
                prev >= 0
                and lines[prev].lstrip().startswith("[")
                and "]" not in lines[prev]
            ):
                meta["date_note"] = f"{lines[prev].strip()} {line}"
                drop.add(prev)
            break

    for source in (meta["date_note"], meta["dedication"]):
        if source and (m := YEAR_RE.search(source)):
            meta["year"] = int(m.group())
            break

    cleaned = "\n".join(line for i, line in enumerate(lines) if i not in drop)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip(), meta


def guess_genre(text: str, meta: dict) -> str:
    if meta["genre"]:
        return meta["genre"]
    return "поема" if len(non_empty_lines(text)) >= POEM_MIN_LINES else "вірш"


def line_key(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip().lower()


def collection_rest(collection: dict, others: list[dict]) -> dict:
    """Лишає у збірнику «Кобзар» лише ті вірші, яких немає в окремих творах.

    Збірник майже повністю дублює окремі сторінки, тому викидаємо рядки, що вже
    є в інших текстах, а з решти беремо лише достатньо довгі суцільні уривки.
    """
    seen = {
        line_key(line)
        for r in others
        for line in r["text"].split("\n")
        if len(line.strip()) >= MIN_DEDUP_LINE
    }
    lines = collection["text"].split("\n")
    start = next((i for i, line in enumerate(lines) if line.strip() == FIRST_POEM), 0)

    runs, current = [], []

    def flush() -> None:
        if len(non_empty_lines("\n".join(current))) >= MIN_REST_RUN:
            runs.append("\n".join(current).strip())
        current.clear()

    for line in lines[start:]:
        if len(line.strip()) >= MIN_DEDUP_LINE and line_key(line) in seen:
            flush()
        else:
            current.append(line)
    flush()
    return {**collection, "title": REST_TITLE, "text": "\n\n".join(runs)}


def line_set(text: str) -> set[str]:
    return {
        line_key(line)
        for line in text.split("\n")
        if len(line.strip()) >= MIN_DEDUP_LINE
    }


def drop_contained(poems: list[dict]) -> tuple[list[dict], list[tuple[str, str]]]:
    """Видаляє тексти, більшість рядків яких міститься в іншому, більшому тексті.

    Так прибираються глави поеми, що лежать і окремо, і всередині неї («Гайдамаки»),
    та вірші з циклів («В казематі»). Повертає тексти й пари (видалений, власник).
    """
    sets = [line_set(p["text"]) for p in poems]
    order = sorted(range(len(poems)), key=lambda i: (len(sets[i]), i))
    removed: dict[int, int] = {}
    for pos, i in enumerate(order):
        if len(sets[i]) < MIN_CONTAINED_LINES:
            continue
        for j in order[pos + 1 :]:
            if j in removed:
                continue
            if len(sets[i] & sets[j]) / len(sets[i]) > MAX_CONTAINED:
                removed[i] = j
                break
    kept = [p for i, p in enumerate(poems) if i not in removed]
    report = [(poems[i]["title"], poems[j]["title"]) for i, j in removed.items()]
    return kept, report


def build() -> tuple[list[dict], list[tuple[str, str]]]:
    corpus = read_jsonl(IN_PATH)
    collection = next(r for r in corpus if r["title"] == COLLECTION_TITLE)
    others = [r for r in corpus if r is not collection]
    candidates = [*others, collection_rest(collection, others)]

    poems = []
    for r in candidates:
        if not is_ukrainian(r["text"]) or not is_verse(r["text"]):
            continue
        text = strip_notes(r["text"])
        meta = {"genre": None, "dedication": None, "date_note": None, "year": None}
        if r["title"] != REST_TITLE:
            text, meta = extract_meta(text)
        meta["genre"] = guess_genre(text, meta)
        if len(non_empty_lines(text)) >= MIN_LINES and len(text) >= MIN_CHARS:
            poems.append({**r, "text": text, **meta})
    return drop_contained(poems)


def main() -> None:
    corpus = read_jsonl(IN_PATH)
    poems, duplicates = build()
    kept = {p["id"] for p in poems}
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for p in poems:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    chars = sum(len(p["text"]) for p in poems)
    print(f"Вхід: {len(corpus)} текстів, вихід: {len(poems)} віршів, {chars} символів")
    years = [p["year"] for p in poems if p["year"]]
    print(f"Рік визначено для {len(years)} із {len(poems)}")
    print(f"Видалено як частини інших текстів: {len(duplicates)}")
    for title, owner in duplicates:
        print(f"  {title}  ⊂  {owner}")
    print(
        "Відкинуто фільтром:", sorted(r["title"] for r in corpus if r["id"] not in kept)
    )
    print(f"-> {OUT_PATH}")


if __name__ == "__main__":
    main()
