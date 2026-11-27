"""Local, experimental single-turn API."""

import hashlib
import logging
import os
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import requests
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, field_validator, model_validator

from backend.hybrid import HybridSearch
from ml.evaluation import retrieval

ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = Path(
    os.environ.get("KOBZAR_CORPUS", ROOT / "ml/outputs/poems-uk-audited-v1.jsonl")
)
MODEL = "qwen3:8b"
OLLAMA_CHAT = "http://127.0.0.1:11434/api/chat"
OPTIONS = {
    "num_ctx": 4096,
    "num_predict": 160,
    "temperature": 0.2,
    "top_p": 0.8,
    "top_k": 20,
    "seed": 42,
}
SYSTEM_PROMPT = (
    "Ти — ШІ-співрозмовник, натхненний Тарасом Шевченком, а не історична "
    "особа. Відповідай українською максимум двома короткими реченнями й "
    "лише за наведеними джерелами. Якщо питають «про що твір», скажи, що "
    "робить або просить ліричний герой, а не переказуй загальні теми. Додай "
    "коротку дослівну цитату, яка безпосередньо підтверджує відповідь. Не "
    "додавай біографічних дат, жанрових тверджень чи фактів, відсутніх у "
    "фрагментах. Якщо фрагменти не відповідають, скажи про нестачу даних. "
    "Не вигадуй цитат і не роби висновків про відсутність фактів в усьому "
    "творі або корпусі."
)
UNGROUNDED_SYSTEM_PROMPT = (
    "Ти — ШІ-співрозмовник, натхненний Тарасом Шевченком, а не історична "
    "особа. Не називай себе людиною чи Шевченком і не приписуй собі "
    "людських почуттів або досвіду. Відповідай українською максимум двома "
    "короткими реченнями, до 45 слів. Джерел до цього запитання не знайдено: "
    "не вигадуй цитат, частоти вживання, причин чи особистих поглядів "
    "Шевченка. Якщо питають про частоту або причину вживання слова автором, "
    "не стверджуй їх без джерел: скажи, що без назви твору це неможливо "
    "перевірити, і попроси назвати твір. За потреби зазнач, що слово сьогодні "
    "може сприйматися як образливе, не приписуючи цього авторові. Не "
    "узагальнюй щодо груп людей. Якщо не знаєш відповіді, чесно скажи про це."
)
REFUSALS = {
    "no_quote": "Наведіть одну цитату в лапках.",
    "multiple_quotes": "Наведіть одну цитату за раз.",
    "quote_too_short": "Наведіть цитату щонайменше з чотирьох слів.",
    "not_found": "Не вдалося знайти текст у доступному корпусі; це не доводить, що він не існує.",
    "ambiguous": "Не вдалося однозначно визначити запис. Уточніть запит.",
    "no_relevant_chunk": "Не вдалося знайти відповідний фрагмент у доступному корпусі.",
}
logger = logging.getLogger(__name__)
ALL_OCCURRENCES_REQUEST = re.compile(
    r"\b(?:знайди|покажи|виведи|переліч)\b.*"
    r"\b(?:всі|усі)\s+(?:місц\w*|випадк\w*|згадк\w*)\b.*\bслово\b",
    re.IGNORECASE | re.DOTALL,
)
WORD_AFTER_LABEL = re.compile(r"\bслово\s+([^\W_]+)", re.IGNORECASE)
WORD_DEFINITION_QUERY = re.compile(
    r"\b(?:що\s+таке|що\s+означає|що\s+значить|значенн\w*|"
    r"визначенн\w*|поясни(?:ти)?\s+слово)\b",
    re.IGNORECASE,
)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    mode: Literal["title", "quote", "auto"]
    retrieval_method: Literal["lexical", "hybrid"] = "lexical"
    answer_mode: Literal["generated", "sources"] = "generated"
    allow_ungrounded: bool = False

    @field_validator("question")
    @classmethod
    def question_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Question must not be blank")
        return value

    @model_validator(mode="after")
    def hybrid_requires_title(self) -> "ChatRequest":
        if self.mode == "quote" and self.retrieval_method != "lexical":
            raise ValueError("Hybrid retrieval is only available in title mode")
        return self


class Source(BaseModel):
    id: int
    title: str
    url: str
    start_char: int
    end_char: int
    text: str


class ChatResponse(BaseModel):
    experimental: Literal[True] = True
    status: Literal["generated", "ungrounded", "sources", "refused"]
    retrieval_status: str
    retrieval_method: Literal["lexical", "hybrid"]
    answer: str
    sources: list[Source]
    model: str | None
    corpus_sha256: str


def should_generate_ungrounded(body: ChatRequest) -> bool:
    if not body.allow_ungrounded or body.mode == "quote":
        return False

    quotations = list(retrieval.QUOTE_RE.finditer(body.question))
    if not quotations:
        return True

    if len(quotations) == 1:
        quoted_text = next(
            group for group in quotations[0].groups() if group is not None
        )
        return (
            len(retrieval.QUOTE_WORD_RE.findall(quoted_text)) == 1
            and WORD_DEFINITION_QUERY.search(body.question) is not None
        )

    return False


def find_word_occurrences(
    question: str, corpus: list[retrieval.Work]
) -> tuple[str, int, list[Source]] | None:
    if not ALL_OCCURRENCES_REQUEST.search(question):
        return None

    quotations = list(retrieval.QUOTE_RE.finditer(question))
    if quotations:
        if len(quotations) != 1:
            return None
        raw_term = next(
            group for group in quotations[0].groups() if group is not None
        ).strip()
    else:
        term_match = WORD_AFTER_LABEL.search(question)
        if term_match is None:
            return None
        raw_term = term_match.group(1)
    terms = retrieval.QUOTE_WORD_RE.findall(raw_term)
    if len(terms) != 1 or terms[0].casefold() != raw_term.casefold():
        return None
    term = terms[0].casefold()

    occurrence_count = 0
    places: dict[tuple[int, int, int], Source] = {}
    for work in corpus:
        text = work["text"]
        for match in retrieval.TOKEN_RE.finditer(text):
            if not match.group().casefold().startswith(term):
                continue
            occurrence_count += 1
            start_char = text.rfind("\n", 0, match.start()) + 1
            end_char = text.find("\n", match.end())
            if end_char == -1:
                end_char = len(text)
            key = (work["id"], start_char, end_char)
            places.setdefault(
                key,
                Source(
                    id=work["id"],
                    title=work["title"],
                    url=work["source"],
                    start_char=start_char,
                    end_char=end_char,
                    text=text[start_char:end_char].strip(),
                ),
            )
    return term, occurrence_count, list(places.values())


@asynccontextmanager
async def lifespan(app: FastAPI):
    data = CORPUS_PATH.read_bytes()
    app.state.corpus = retrieval.load_corpus(data)
    app.state.chunks = retrieval.build_chunks(app.state.corpus)
    app.state.corpus_sha256 = hashlib.sha256(data).hexdigest()
    app.state.hybrid_search = HybridSearch()
    yield


app = FastAPI(title="KobzarAI experimental API", version="0.1.0", lifespan=lifespan)


def find_sources(
    question: str,
    mode: Literal["title", "quote"],
    corpus: list[retrieval.Work],
    chunks: list[retrieval.Chunk],
    retrieval_method: Literal["lexical", "hybrid"] = "lexical",
    hybrid_search: HybridSearch | None = None,
) -> tuple[str, list[retrieval.SearchHit]]:
    if mode == "quote":
        return retrieval.search_quote(question, corpus, chunks)
    status, work = retrieval.find_work(question, corpus)
    if work is None:
        return status, []
    if retrieval_method == "hybrid":
        if hybrid_search is None:
            raise ValueError("Hybrid retrieval is not initialized")
        try:
            hits = hybrid_search.search(question, work, chunks)
        except requests.Timeout as exc:
            logger.warning("Hybrid retrieval timed out: %s", exc)
            raise HTTPException(
                status_code=504, detail="Hybrid retrieval timed out"
            ) from exc
        except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
            logger.warning("Hybrid retrieval failed: %s", exc)
            raise HTTPException(
                status_code=502, detail="Hybrid retrieval failed"
            ) from exc
    else:
        hits = retrieval.search_chunks(
            question, chunks, work_id=work["id"], exclude_title=work["title"]
        )
    return ("matched", hits) if hits else ("no_relevant_chunk", [])


def generate_answer(
    question: str, mode: Literal["title", "quote"], hits: list[retrieval.SearchHit]
) -> str:
    prompt = retrieval.add_chunk_context(question, hits) if hits else question
    if hits and mode == "quote":
        prompt += f"\n\nНазва знайденого твору: «{hits[0]['chunk']['title']}»."
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT if hits else UNGROUNDED_SYSTEM_PROMPT,
            },
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "think": False,
        "options": OPTIONS,
    }
    try:
        response = requests.post(OLLAMA_CHAT, json=payload, timeout=(10, 300))
        response.raise_for_status()
        result = response.json()
    except requests.Timeout as exc:
        logger.warning("Ollama request timed out")
        raise HTTPException(status_code=504, detail="Ollama request timed out") from exc
    except (requests.RequestException, ValueError) as exc:
        logger.warning("Ollama request failed: %s", exc)
        raise HTTPException(status_code=502, detail="Ollama request failed") from exc
    if (
        not isinstance(result, dict)
        or result.get("done") is not True
        or result.get("done_reason") != "stop"
        or not isinstance(result.get("message"), dict)
        or not isinstance(result["message"].get("content"), str)
        or not result["message"]["content"].strip()
        or type(result.get("prompt_eval_count")) is not int
        or not 0
        < result["prompt_eval_count"]
        < OPTIONS["num_ctx"] - OPTIONS["num_predict"]
    ):
        logger.warning("Ollama returned an incomplete or invalid response")
        raise HTTPException(
            status_code=502, detail="Ollama returned an incomplete or invalid response"
        )
    return result["message"]["content"]


@app.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest, request: Request) -> ChatResponse:
    if body.mode == "auto":
        occurrence_result = find_word_occurrences(
            body.question, request.app.state.corpus
        )
        if occurrence_result is not None:
            term, count, sources = occurrence_result
            answer = (
                f"Збігів з основою «{term}» у корпусі: {count}; "
                f"рядків з такими збігами: {len(sources)}. "
                "Нижче наведено місця з текстів; "
                "сюди можуть входити словоформи й похідні слова."
                if count
                else f"У доступному корпусі не знайдено слова з основою «{term}»."
            )
            return ChatResponse(
                status="sources",
                retrieval_status="term_matches" if count else "term_not_found",
                retrieval_method="lexical",
                answer=answer,
                sources=sources,
                model=None,
                corpus_sha256=request.app.state.corpus_sha256,
            )

    mode: Literal["title", "quote"] = "title" if body.mode == "auto" else body.mode
    method = body.retrieval_method
    if body.mode == "auto":
        quote_status, quote_hits = retrieval.search_quote(
            body.question, request.app.state.corpus, request.app.state.chunks
        )
        if quote_hits or quote_status == "ambiguous":
            mode = "quote"
            method = "lexical"
            status, hits = quote_status, quote_hits
        else:
            status, hits = find_sources(
                body.question,
                mode,
                request.app.state.corpus,
                request.app.state.chunks,
                method,
                request.app.state.hybrid_search,
            )
    else:
        status, hits = find_sources(
            body.question,
            mode,
            request.app.state.corpus,
            request.app.state.chunks,
            method,
            request.app.state.hybrid_search,
        )
    if not hits:
        if body.answer_mode == "generated" and should_generate_ungrounded(body):
            return ChatResponse(
                status="ungrounded",
                retrieval_status=status,
                retrieval_method=method,
                answer=generate_answer(body.question, mode, hits),
                sources=[],
                model=MODEL,
                corpus_sha256=request.app.state.corpus_sha256,
            )
        return ChatResponse(
            status="refused",
            retrieval_status=status,
            retrieval_method=method,
            answer=REFUSALS[status],
            sources=[],
            model=None,
            corpus_sha256=request.app.state.corpus_sha256,
        )
    sources = [
        Source(
            id=hit["chunk"]["id"],
            title=hit["chunk"]["title"],
            url=hit["chunk"]["source"],
            start_char=hit["chunk"]["start_char"],
            end_char=hit["chunk"]["end_char"],
            text=hit["chunk"]["text"],
        )
        for hit in hits
    ]
    return ChatResponse(
        status=body.answer_mode,
        retrieval_status=status,
        retrieval_method=method,
        answer=(
            generate_answer(body.question, mode, hits)
            if body.answer_mode == "generated"
            else "Знайдені фрагменти можуть не містити відповіді; перевірте їх самостійно."
        ),
        sources=sources,
        model=MODEL if body.answer_mode == "generated" else None,
        corpus_sha256=request.app.state.corpus_sha256,
    )
