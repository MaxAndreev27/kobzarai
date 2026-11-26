"""Compare lexical and local Ollama dense retrieval on fresh development cases."""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import NotRequired, TypedDict

import requests

if __package__:
    from . import retrieval
else:
    import retrieval

API = "http://127.0.0.1:11434/api"
CASES = Path(__file__).with_name("retrieval_semantic_cases.jsonl")
MODEL = "bge-m3:latest"
BATCH_SIZE = 8
OPTIONS = {"num_ctx": 1024}


class Case(TypedDict):
    id: str
    query: str
    expected_status: NotRequired[str]
    work_id: NotRequired[int]
    evidence: NotRequired[str]
    answerable: NotRequired[bool]


def normalize_vector(vector: list[float]) -> list[float]:
    if not vector or not all(math.isfinite(value) for value in vector):
        raise ValueError("Invalid embedding: empty or non-finite vector")
    norm = math.sqrt(sum(value * value for value in vector))
    if not math.isfinite(norm) or norm == 0:
        raise ValueError("Invalid embedding: zero or non-finite norm")
    return [value / norm for value in vector]


def embed(texts: list[str], model: str, digest: str) -> tuple[list[list[float]], dict]:
    response = requests.post(
        f"{API}/embed",
        json={
            "model": model,
            "input": texts,
            "truncate": False,
            "keep_alive": "10m",
            "options": OPTIONS,
        },
        timeout=300,
    )
    response.raise_for_status()
    vectors = response.json()["embeddings"]
    if len(vectors) != len(texts):
        raise ValueError("Ollama returned an unexpected number of embeddings")
    normalized = [normalize_vector(vector) for vector in vectors]
    if normalized and any(len(vector) != len(normalized[0]) for vector in normalized):
        raise ValueError("Ollama returned inconsistent embedding dimensions")
    placement_response = requests.get(f"{API}/ps", timeout=10)
    placement_response.raise_for_status()
    placement = next(
        (
            item
            for item in placement_response.json()["models"]
            if item.get("digest") == digest
        ),
        None,
    )
    if placement is None or placement.get("size_vram", 0) <= 0:
        raise RuntimeError(f"GPU placement not confirmed for {model}: {placement}")
    return normalized, placement


def rank_dense(
    query_vector: list[float],
    candidates: list[retrieval.Chunk],
    vectors: list[list[float]],
    top_k: int = retrieval.DEFAULT_TOP_K,
) -> list[retrieval.SearchHit]:
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if len(candidates) != len(vectors):
        raise ValueError("Chunk and embedding counts differ")
    hits: list[retrieval.SearchHit] = []
    for chunk, vector in zip(candidates, vectors, strict=True):
        if len(query_vector) != len(vector):
            raise ValueError("Query and chunk embedding dimensions differ")
        score = sum(a * b for a, b in zip(query_vector, vector, strict=True))
        hits.append(retrieval.SearchHit(chunk=chunk, score=score))
    hits.sort(key=lambda hit: (-hit["score"], hit["chunk"]["start_char"]))
    return hits[:top_k]


def hit_summary(hits: list[retrieval.SearchHit], evidence: str | None) -> dict:
    rank = next(
        (
            i
            for i, hit in enumerate(hits, 1)
            if evidence and evidence in hit["chunk"]["text"]
        ),
        None,
    )
    return {
        "evidence_rank": rank,
        "hits": [
            {
                "id": hit["chunk"]["id"],
                "start_char": hit["chunk"]["start_char"],
                "end_char": hit["chunk"]["end_char"],
                "score": hit["score"],
            }
            for hit in hits
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")

    corpus_data = retrieval.CORPUS.read_bytes()
    cases_data = CASES.read_bytes()
    corpus = retrieval.load_corpus(corpus_data)
    chunks = retrieval.build_chunks(corpus)
    cases: list[Case] = [
        json.loads(line)
        for line in cases_data.decode("utf-8").splitlines()
        if line.strip()
    ]
    works = {work["id"]: work for work in corpus}
    seen = set()
    for case in cases:
        if case["id"] in seen:
            raise ValueError(f"Duplicate case ID: {case['id']}")
        seen.add(case["id"])
        status = case.get("expected_status", "matched")
        if status not in {"matched", "ambiguous", "not_found"}:
            raise ValueError(f"Invalid expected status: {case['id']}")
        if status == "matched":
            work = works.get(case.get("work_id", -1))
            if work is None:
                raise ValueError(f"Missing expected work: {case['id']}")
            if case.get("answerable", True):
                evidence = case.get("evidence")
                if not evidence or not any(
                    chunk["id"] == work["id"] and evidence in chunk["text"]
                    for chunk in chunks
                ):
                    raise ValueError(
                        f"Expected evidence not contained in a chunk: {case['id']}"
                    )

    version_response = requests.get(f"{API}/version", timeout=10)
    version_response.raise_for_status()
    tags_response = requests.get(f"{API}/tags", timeout=10)
    tags_response.raise_for_status()
    model = next(
        (item for item in tags_response.json()["models"] if item["name"] == MODEL), None
    )
    if model is None:
        raise ValueError(f"Local embedding model unavailable: {MODEL}")
    digest = model["digest"]

    lookups = [retrieval.find_work(case["query"], corpus) for case in cases]
    selected_ids = {work["id"] for _, work in lookups if work is not None}
    indexed_chunks = [chunk for chunk in chunks if chunk["id"] in selected_ids]
    index_start = time.perf_counter()
    indexed_vectors: list[list[float]] = []
    placement = None
    for start in range(0, len(indexed_chunks), BATCH_SIZE):
        batch = indexed_chunks[start : start + BATCH_SIZE]
        vectors, placement = embed([chunk["text"] for chunk in batch], MODEL, digest)
        indexed_vectors.extend(vectors)
        print(
            f"Indexed {len(indexed_vectors)}/{len(indexed_chunks)} chunks", flush=True
        )
    index_seconds = time.perf_counter() - index_start
    if not indexed_vectors:
        raise ValueError("No chunks selected by title lookup")
    dimensions = len(indexed_vectors[0])
    if any(len(vector) != dimensions for vector in indexed_vectors):
        raise ValueError("Inconsistent embedding dimensions across batches")

    results = []
    for case, (status, work) in zip(cases, lookups, strict=True):
        entry: dict = {
            "id": case["id"],
            "query": case["query"],
            "expected_status": case.get("expected_status", "matched"),
            "expected_work_id": case.get("work_id"),
            "answerable": case.get("answerable", True),
            "actual_status": status,
            "actual_work_id": work["id"] if work is not None else None,
        }
        if work is not None:
            lexical_start = time.perf_counter()
            lexical_hits = retrieval.search_chunks(
                case["query"], chunks, work_id=work["id"], exclude_title=work["title"]
            )
            lexical_seconds = time.perf_counter() - lexical_start
            dense_start = time.perf_counter()
            query_vectors, placement = embed([case["query"]], MODEL, digest)
            candidates = [
                (chunk, vector)
                for chunk, vector in zip(indexed_chunks, indexed_vectors, strict=True)
                if chunk["id"] == work["id"]
            ]
            dense_hits = rank_dense(
                query_vectors[0],
                [item[0] for item in candidates],
                [item[1] for item in candidates],
            )
            dense_seconds = time.perf_counter() - dense_start
            for name, hits, seconds in (
                ("lexical", lexical_hits, lexical_seconds),
                ("dense", dense_hits, dense_seconds),
            ):
                entry[name] = hit_summary(hits, case.get("evidence"))
                entry[name]["seconds"] = seconds
        results.append(entry)
        print(f"Evaluated {case['id']}: {status}", flush=True)

    eligible = [
        row
        for row in results
        if row["expected_status"] == "matched" and row["answerable"]
    ]
    negatives = [row for row in results if row["expected_status"] != "matched"]
    unanswerable = [
        row
        for row in results
        if row["expected_status"] == "matched" and not row["answerable"]
    ]
    metrics = {}
    for method in ("lexical", "dense"):
        ranks = [
            row[method]["evidence_rank"]
            if row["actual_work_id"] == row["expected_work_id"] and method in row
            else None
            for row in eligible
        ]
        measured = [row[method]["seconds"] for row in results if method in row]
        metrics[method] = {
            "answerable_cases": len(eligible),
            "evidence_at_1": sum(rank == 1 for rank in ranks),
            "evidence_at_2": sum(rank is not None for rank in ranks),
            "mean_query_seconds": sum(measured) / len(measured) if measured else None,
            "unanswerable_with_candidates": sum(
                bool(row.get(method, {}).get("hits")) for row in unanswerable
            ),
            "unanswerable_cases": len(unanswerable),
        }
    report = {
        "corpus_sha256": hashlib.sha256(corpus_data).hexdigest(),
        "cases_sha256": hashlib.sha256(cases_data).hexdigest(),
        "model": model,
        "ollama_version": version_response.json()["version"],
        "options": OPTIONS,
        "batch_size": BATCH_SIZE,
        "gpu_placement": placement,
        "indexed_work_count": len(selected_ids),
        "indexed_chunk_count": len(indexed_chunks),
        "embedding_dimensions": dimensions,
        "index_seconds": index_seconds,
        "correct_negative_status": sum(
            row["actual_status"] == row["expected_status"] for row in negatives
        ),
        "negative_cases": len(negatives),
        "metrics": metrics,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(report, output, ensure_ascii=False, indent=2)
        output.write("\n")
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2), flush=True)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
