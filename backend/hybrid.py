"""Optional per-work lexical and dense retrieval via local Ollama."""

import threading

import requests

from ml.evaluation import compare_retrieval as comparison
from ml.evaluation import evaluate_hybrid, retrieval

API = "http://127.0.0.1:11434/api"
MODEL = comparison.MODEL


class HybridSearch:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.digest: str | None = None
        self.vectors: dict[int, list[list[float]]] = {}

    def _model_digest(self) -> str:
        response = requests.get(f"{API}/tags", timeout=10)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
            raise TypeError("Invalid Ollama model list")
        models = payload["models"]
        if any(
            not isinstance(item, dict) or not isinstance(item.get("name"), str)
            for item in models
        ):
            raise ValueError("Invalid Ollama model entry")
        model = next((item for item in models if item["name"] == MODEL), None)
        if (
            model is None
            or not isinstance(model.get("digest"), str)
            or not model["digest"]
        ):
            raise ValueError(f"Local embedding model unavailable: {MODEL}")
        return model["digest"]

    def _embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for start in range(0, len(texts), comparison.BATCH_SIZE):
            batch = texts[start : start + comparison.BATCH_SIZE]
            response = requests.post(
                f"{API}/embed",
                json={
                    "model": MODEL,
                    "input": batch,
                    "truncate": False,
                    "keep_alive": "10m",
                    "options": comparison.OPTIONS,
                },
                timeout=(10, 300),
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict) or not isinstance(
                payload.get("embeddings"), list
            ):
                raise TypeError("Invalid Ollama embeddings response")
            raw = payload["embeddings"]
            if len(raw) != len(batch):
                raise ValueError("Unexpected number of embeddings")
            for vector in raw:
                if not isinstance(vector, list) or any(
                    not isinstance(value, (int, float)) or isinstance(value, bool)
                    for value in vector
                ):
                    raise ValueError("Invalid embedding")
                vectors.append(comparison.normalize_vector(vector))
        return vectors

    def search(
        self,
        query: str,
        work: retrieval.Work,
        chunks: list[retrieval.Chunk],
    ) -> list[retrieval.SearchHit]:
        candidates = [chunk for chunk in chunks if chunk["id"] == work["id"]]
        if not candidates:
            return []
        with self.lock:
            digest = self._model_digest()
            if digest != self.digest:
                self.vectors.clear()
                self.digest = digest
            vectors = self.vectors.get(work["id"])
            if vectors is None:
                vectors = self._embed([chunk["text"] for chunk in candidates])
            query_vector = self._embed([query])[0]
            if self._model_digest() != digest:
                self.vectors.clear()
                self.digest = None
                raise ValueError("Embedding model changed during retrieval")
            self.vectors[work["id"]] = vectors
        lexical = retrieval.search_chunks(
            query,
            candidates,
            work_id=work["id"],
            exclude_title=work["title"],
            top_k=len(candidates),
        )
        dense = comparison.rank_dense(
            query_vector, candidates, vectors, top_k=len(candidates)
        )
        return evaluate_hybrid.fuse(lexical, dense)
