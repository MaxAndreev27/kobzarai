"""Evaluate RRF retrieval and an experimental abstention gate."""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import compare_retrieval as comparison
import requests
import retrieval

VALIDATION = Path(__file__).with_name("retrieval_hybrid_validation_cases.jsonl")
RRF_K = 60
REFUSAL = (
    "Недостатньо підтверджень у доступних фрагментах твору, щоб надійно відповісти."
)


def fuse(
    lexical: list[retrieval.SearchHit],
    dense: list[retrieval.SearchHit],
    top_k: int = retrieval.DEFAULT_TOP_K,
) -> list[retrieval.SearchHit]:
    if top_k < 1:
        raise ValueError("top_k must be positive")
    combined: dict[tuple[int, int, int], retrieval.SearchHit] = {}
    for ranking in (lexical, dense):
        seen: set[tuple[int, int, int]] = set()
        for rank, hit in enumerate(ranking, 1):
            chunk = hit["chunk"]
            key = (chunk["id"], chunk["start_char"], chunk["end_char"])
            if key in seen:
                raise ValueError("Duplicate chunk in a ranking")
            seen.add(key)
            if key not in combined:
                combined[key] = retrieval.SearchHit(chunk=chunk, score=0.0)
            combined[key]["score"] += 1.0 / (RRF_K + rank)
    return sorted(
        combined.values(),
        key=lambda hit: (-hit["score"], hit["chunk"]["start_char"], hit["chunk"]["id"]),
    )[:top_k]


def choose_threshold(samples: list[tuple[float | None, bool]]) -> float:
    positives = sum(answerable for _, answerable in samples)
    negatives = len(samples) - positives
    scores = [score for score, _ in samples if score is not None]
    if (
        not positives
        or not negatives
        or not any(score is not None and answerable for score, answerable in samples)
        or not any(
            score is not None and not answerable for score, answerable in samples
        )
    ):
        raise ValueError(
            "Calibration needs answerable and unanswerable cases with scores"
        )
    if not all(math.isfinite(score) and -1 <= score <= 1 for score in scores):
        raise ValueError("Invalid cosine similarity in calibration")
    thresholds = sorted({-1.0, *scores, math.nextafter(max(scores), math.inf)})

    def quality(threshold: float) -> tuple[float, float]:
        correct_positives = sum(
            answerable and score is not None and score >= threshold
            for score, answerable in samples
        )
        correct_negatives = sum(
            not answerable and (score is None or score < threshold)
            for score, answerable in samples
        )
        balanced_accuracy = (
            correct_positives / positives + correct_negatives / negatives
        ) / 2
        return balanced_accuracy, threshold

    return max(thresholds, key=quality)


def decide(score: float | None, threshold: float) -> bool:
    return score is not None and score >= threshold


def load_cases(
    path: Path, chunks: list[retrieval.Chunk]
) -> tuple[bytes, list[comparison.Case]]:
    data = path.read_bytes()
    cases: list[comparison.Case] = [
        json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()
    ]
    seen = set()
    for case in cases:
        if case["id"] in seen:
            raise ValueError(f"Duplicate case ID: {case['id']}")
        seen.add(case["id"])
        expected_status = case.get("expected_status", "matched")
        if expected_status not in {"matched", "ambiguous", "not_found"}:
            raise ValueError(f"Invalid expected status: {case['id']}")
        if expected_status == "matched":
            work_id = case.get("work_id")
            candidates = [chunk for chunk in chunks if chunk["id"] == work_id]
            if not candidates:
                raise ValueError(f"Missing expected work: {case['id']}")
            evidence = case.get("evidence")
            if case.get("answerable", True) and (
                not evidence
                or not any(evidence in chunk["text"] for chunk in candidates)
            ):
                raise ValueError(
                    f"Expected evidence not contained in a chunk: {case['id']}"
                )
    return data, cases


def evaluate_cases(
    cases: list[comparison.Case],
    lookups: list[tuple[str, retrieval.Work | None]],
    chunks: list[retrieval.Chunk],
    indexed_chunks: list[retrieval.Chunk],
    indexed_vectors: list[list[float]],
    digest: str,
) -> tuple[list[dict], dict | None]:
    rows = []
    placement = None
    for case, (status, work) in zip(cases, lookups, strict=True):
        row: dict = {
            "id": case["id"],
            "query": case["query"],
            "expected_status": case.get("expected_status", "matched"),
            "expected_work_id": case.get("work_id"),
            "answerable": case.get("answerable", True),
            "actual_status": status,
            "actual_work_id": work["id"] if work is not None else None,
            "confidence": None,
        }
        if work is not None:
            started = time.perf_counter()
            candidates = [
                (chunk, vector)
                for chunk, vector in zip(indexed_chunks, indexed_vectors, strict=True)
                if chunk["id"] == work["id"]
            ]
            lexical = retrieval.search_chunks(
                case["query"],
                chunks,
                work_id=work["id"],
                exclude_title=work["title"],
                top_k=len(candidates),
            )
            vectors, placement = comparison.embed(
                [case["query"]], comparison.MODEL, digest
            )
            dense = comparison.rank_dense(
                vectors[0],
                [item[0] for item in candidates],
                [item[1] for item in candidates],
                top_k=len(candidates),
            )
            hybrid = fuse(lexical, dense)
            similarities = {
                (
                    hit["chunk"]["id"],
                    hit["chunk"]["start_char"],
                    hit["chunk"]["end_char"],
                ): hit["score"]
                for hit in dense
            }
            if hybrid:
                row["confidence"] = max(
                    similarities[
                        (
                            hit["chunk"]["id"],
                            hit["chunk"]["start_char"],
                            hit["chunk"]["end_char"],
                        )
                    ]
                    for hit in hybrid
                )
            for name, hits in (
                ("lexical", lexical[:2]),
                ("dense", dense[:2]),
                ("hybrid", hybrid),
            ):
                row[name] = comparison.hit_summary(hits, case.get("evidence"))
            row["search_seconds"] = time.perf_counter() - started
        rows.append(row)
        print(f"Evaluated {case['id']}: {status}", flush=True)
    return rows, placement


def summarize(rows: list[dict], threshold: float) -> dict:
    positives = [
        row for row in rows if row["expected_status"] == "matched" and row["answerable"]
    ]
    negatives = [
        row
        for row in rows
        if row["expected_status"] == "matched" and not row["answerable"]
    ]
    title_negative = [row for row in rows if row["expected_status"] != "matched"]
    metrics: dict = {
        "answerable_cases": len(positives),
        "unanswerable_cases": len(negatives),
    }
    for method in ("lexical", "dense", "hybrid"):
        ranks = [
            row[method]["evidence_rank"]
            if row["actual_work_id"] == row["expected_work_id"] and method in row
            else None
            for row in positives
        ]
        metrics[method] = {
            "evidence_at_1": sum(rank == 1 for rank in ranks),
            "evidence_at_2": sum(rank is not None for rank in ranks),
        }
    for row in rows:
        row["accepted"] = decide(row["confidence"], threshold)
        row["refusal"] = None if row["accepted"] else REFUSAL
    metrics["gate"] = {
        "answerable_rejected": sum(not row["accepted"] for row in positives),
        "unanswerable_accepted": sum(row["accepted"] for row in negatives),
        "accepted_with_expected_evidence": sum(
            row["accepted"]
            and row["actual_work_id"] == row["expected_work_id"]
            and row.get("hybrid", {}).get("evidence_rank") is not None
            for row in positives
        ),
        "title_negative_cases": len(title_negative),
        "title_negative_rejected": sum(not row["accepted"] for row in title_negative),
    }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    corpus_data = retrieval.CORPUS.read_bytes()
    corpus = retrieval.load_corpus(corpus_data)
    chunks = retrieval.build_chunks(corpus)
    training_data, training = load_cases(comparison.CASES, chunks)
    validation_data, validation = load_cases(VALIDATION, chunks)
    if {case["id"] for case in training} & {case["id"] for case in validation}:
        raise ValueError("Calibration and validation case IDs overlap")

    version = requests.get(f"{comparison.API}/version", timeout=10)
    version.raise_for_status()
    tags = requests.get(f"{comparison.API}/tags", timeout=10)
    tags.raise_for_status()
    model = next(
        (item for item in tags.json()["models"] if item["name"] == comparison.MODEL),
        None,
    )
    if model is None:
        raise ValueError(f"Local embedding model unavailable: {comparison.MODEL}")
    digest = model["digest"]
    training_lookups = [retrieval.find_work(case["query"], corpus) for case in training]
    validation_lookups = [
        retrieval.find_work(case["query"], corpus) for case in validation
    ]
    ids = {
        work["id"]
        for _, work in training_lookups + validation_lookups
        if work is not None
    }
    indexed_chunks = [chunk for chunk in chunks if chunk["id"] in ids]
    indexed_vectors: list[list[float]] = []
    index_start = time.perf_counter()
    placement = None
    for start in range(0, len(indexed_chunks), comparison.BATCH_SIZE):
        batch = indexed_chunks[start : start + comparison.BATCH_SIZE]
        vectors, placement = comparison.embed(
            [chunk["text"] for chunk in batch], comparison.MODEL, digest
        )
        indexed_vectors.extend(vectors)
        print(
            f"Indexed {len(indexed_vectors)}/{len(indexed_chunks)} chunks", flush=True
        )
    index_seconds = time.perf_counter() - index_start
    if not indexed_vectors:
        raise ValueError("No chunks selected by title lookup")

    training_rows, training_placement = evaluate_cases(
        training, training_lookups, chunks, indexed_chunks, indexed_vectors, digest
    )
    samples = [
        (row["confidence"], row["answerable"])
        for row in training_rows
        if row["expected_status"] == "matched"
    ]
    threshold = choose_threshold(samples)
    training_metrics = summarize(training_rows, threshold)
    print(f"Frozen calibration threshold: {threshold:.6f}", flush=True)
    validation_rows, validation_placement = evaluate_cases(
        validation, validation_lookups, chunks, indexed_chunks, indexed_vectors, digest
    )
    validation_metrics = summarize(validation_rows, threshold)
    report = {
        "corpus_sha256": hashlib.sha256(corpus_data).hexdigest(),
        "calibration_sha256": hashlib.sha256(training_data).hexdigest(),
        "validation_sha256": hashlib.sha256(validation_data).hexdigest(),
        "model": model,
        "ollama_version": version.json()["version"],
        "options": comparison.OPTIONS,
        "gpu_placement": validation_placement or training_placement or placement,
        "indexed_work_count": len(ids),
        "indexed_chunk_count": len(indexed_chunks),
        "index_seconds": index_seconds,
        "rrf_k": RRF_K,
        "threshold": threshold,
        "threshold_selection": "max balanced accuracy on calibration labels; ties favor higher threshold",
        "calibration_metrics": training_metrics,
        "validation_metrics": validation_metrics,
        "calibration_results": training_rows,
        "validation_results": validation_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(report, output, ensure_ascii=False, indent=2)
        output.write("\n")
    print(json.dumps(validation_metrics, ensure_ascii=False, indent=2), flush=True)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
