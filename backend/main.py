"""Local, experimental source-grounded single-turn API."""

import hashlib
import logging
import os
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
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "seed": 42,
}
SYSTEM_PROMPT = (
    "Ти — ШІ-співрозмовник, натхненний Тарасом Шевченком, а не історична "
    "особа. Відповідай українською стисло, лише за наведеними джерелами. "
    "Якщо джерела не містять відповіді, скажи, що даних недостатньо. "
    "Не вигадуй фактів чи цитат і не роби висновків про відсутність фактів "
    "в усьому творі або корпусі."
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


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    mode: Literal["title", "quote"]
    retrieval_method: Literal["lexical", "hybrid"] = "lexical"
    answer_mode: Literal["generated", "sources"] = "generated"

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
    status: Literal["generated", "sources", "refused"]
    retrieval_status: str
    retrieval_method: Literal["lexical", "hybrid"]
    answer: str
    sources: list[Source]
    model: str | None
    corpus_sha256: str


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
    prompt = retrieval.add_chunk_context(question, hits)
    if mode == "quote":
        prompt += f"\n\nНазва знайденого твору: «{hits[0]['chunk']['title']}»."
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
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
    status, hits = find_sources(
        body.question,
        body.mode,
        request.app.state.corpus,
        request.app.state.chunks,
        body.retrieval_method,
        request.app.state.hybrid_search,
    )
    if not hits:
        return ChatResponse(
            status="refused",
            retrieval_status=status,
            retrieval_method=body.retrieval_method,
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
        retrieval_method=body.retrieval_method,
        answer=(
            generate_answer(body.question, body.mode, hits)
            if body.answer_mode == "generated"
            else "Знайдені фрагменти можуть не містити відповіді; перевірте їх самостійно."
        ),
        sources=sources,
        model=MODEL if body.answer_mode == "generated" else None,
        corpus_sha256=request.app.state.corpus_sha256,
    )
