"""Compare corpus-wide quote lookup with existing title lookup."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import retrieval

CASES = Path(__file__).with_name("retrieval_quote_cases.jsonl")


def quote_rank(query: str, hits: list[retrieval.SearchHit]) -> int | None:
    quotes = list(retrieval.QUOTE_RE.finditer(query))
    if len(quotes) != 1:
        return None
    phrase = retrieval.normalize(
        next(group for group in quotes[0].groups() if group is not None)
    )
    if not phrase:
        return None
    needle = f" {phrase} "
    return next(
        (
            index
            for index, hit in enumerate(hits, 1)
            if needle in f" {retrieval.normalize(hit['chunk']['text'])} "
        ),
        None,
    )


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
    cases = [
        json.loads(line)
        for line in cases_data.decode("utf-8").splitlines()
        if line.strip()
    ]
    ids = set()
    works = {work["id"]: work for work in corpus}
    for case in cases:
        if case["id"] in ids:
            raise ValueError(f"Duplicate case ID: {case['id']}")
        ids.add(case["id"])
        if case["expected_status"] == "matched":
            work = works.get(case["work_id"])
            if (
                work is None
                or quote_rank(
                    case["query"],
                    [
                        retrieval.SearchHit(chunk=chunk, score=1.0)
                        for chunk in chunks
                        if chunk["id"] == work["id"]
                    ],
                )
                is None
            ):
                raise ValueError(
                    f"Expected quote not found in a source chunk: {case['id']}"
                )
    results = []
    for case in cases:
        query = case["query"]
        started = time.perf_counter()
        title_status, work = retrieval.find_work(query, corpus)
        title_hits = (
            retrieval.search_chunks(
                query, chunks, work_id=work["id"], exclude_title=work["title"]
            )
            if work is not None
            else []
        )
        title_seconds = time.perf_counter() - started
        started = time.perf_counter()
        status, hits = retrieval.search_quote(query, corpus, chunks)
        quote_seconds = time.perf_counter() - started
        results.append(
            {
                "id": case["id"],
                "query": query,
                "expected_status": case["expected_status"],
                "expected_work_id": case.get("work_id"),
                "baseline": {
                    "status": title_status,
                    "work_id": work["id"] if work is not None else None,
                    "quote_rank": quote_rank(query, title_hits),
                    "seconds": title_seconds,
                },
                "quote_search": {
                    "status": status,
                    "work_id": hits[0]["chunk"]["id"] if hits else None,
                    "quote_rank": quote_rank(query, hits),
                    "seconds": quote_seconds,
                    "hits": [
                        {
                            "id": hit["chunk"]["id"],
                            "start_char": hit["chunk"]["start_char"],
                            "end_char": hit["chunk"]["end_char"],
                        }
                        for hit in hits
                    ],
                },
            }
        )
    positives = [row for row in results if row["expected_status"] == "matched"]
    negatives = [row for row in results if row["expected_status"] != "matched"]
    metrics = {}
    for method in ("baseline", "quote_search"):
        metrics[method] = {
            "correct_work": sum(
                row[method]["work_id"] == row["expected_work_id"] for row in positives
            ),
            "quote_at_1": sum(
                row[method]["work_id"] == row["expected_work_id"]
                and row[method]["quote_rank"] == 1
                for row in positives
            ),
            "quote_at_2": sum(
                row[method]["work_id"] == row["expected_work_id"]
                and row[method]["quote_rank"] is not None
                for row in positives
            ),
            "negative_matches": sum(
                row[method]["status"] == "matched" for row in negatives
            ),
        }
    metrics["quote_search"]["correct_negative_status"] = sum(
        row["quote_search"]["status"] == row["expected_status"] for row in negatives
    )
    report = {
        "corpus_sha256": hashlib.sha256(corpus_data).hexdigest(),
        "cases_sha256": hashlib.sha256(cases_data).hexdigest(),
        "positive_cases": len(positives),
        "negative_cases": len(negatives),
        "metrics": metrics,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(report, output, ensure_ascii=False, indent=2)
        output.write("\n")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
