"""Evaluate title lookup and chunk retrieval without passing expected IDs to lookup."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import NotRequired, TypedDict

import retrieval

CASE_FILES = (
    Path(__file__).with_name("retrieval_cases.jsonl"),
    Path(__file__).with_name("retrieval_end_to_end_cases.jsonl"),
)


class Case(TypedDict):
    id: str
    query: str
    expected_status: NotRequired[str]
    work_id: NotRequired[int]
    evidence: NotRequired[str]


def evaluate_case(
    case: Case, corpus: list[retrieval.Work], chunks: list[retrieval.Chunk]
) -> dict:
    status, work = retrieval.find_work(case["query"], corpus)
    hits = (
        retrieval.search_chunks(
            case["query"], chunks, work_id=work["id"], exclude_title=work["title"]
        )
        if work is not None
        else []
    )
    expected_status = case.get("expected_status", "matched")
    title_correct = status == expected_status and (
        work is None or work["id"] == case.get("work_id")
    )
    evidence = case.get("evidence")
    rank = next(
        (
            index
            for index, hit in enumerate(hits, 1)
            if evidence and evidence in hit["chunk"]["text"]
        ),
        None,
    )
    return {
        "id": case["id"],
        "query": case["query"],
        "expected_status": expected_status,
        "expected_work_id": case.get("work_id"),
        "actual_status": status,
        "actual_work_id": work["id"] if work is not None else None,
        "title_correct": title_correct,
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
    parser.add_argument(
        "--output", type=Path, help="Save JSON report without overwriting"
    )
    args = parser.parse_args()

    corpus_data = retrieval.CORPUS.read_bytes()
    corpus = retrieval.load_corpus(corpus_data)
    chunks = retrieval.build_chunks(corpus)
    cases: list[Case] = []
    case_hashes = {}
    for path in CASE_FILES:
        data = path.read_bytes()
        case_hashes[path.name] = hashlib.sha256(data).hexdigest()
        cases.extend(
            json.loads(line)
            for line in data.decode("utf-8").splitlines()
            if line.strip()
        )

    seen = set()
    works = {work["id"]: work for work in corpus}
    for case in cases:
        if case["id"] in seen:
            raise ValueError(f"Duplicate case ID: {case['id']}")
        seen.add(case["id"])
        expected = case.get("expected_status", "matched")
        if expected not in {"matched", "ambiguous", "not_found"}:
            raise ValueError(f"Invalid expected status: {case['id']}")
        if expected == "matched":
            work = works.get(case.get("work_id", -1))
            evidence = case.get("evidence")
            if work is None or not evidence or evidence not in work["text"]:
                raise ValueError(f"Invalid expected work or evidence: {case['id']}")

    results = [evaluate_case(case, corpus, chunks) for case in cases]
    positive = [result for result in results if result["expected_status"] == "matched"]
    negative = [result for result in results if result["expected_status"] != "matched"]
    report = {
        "corpus_sha256": hashlib.sha256(corpus_data).hexdigest(),
        "cases_sha256": case_hashes,
        "work_count": len(corpus),
        "chunk_count": len(chunks),
        "metrics": {
            "positive_cases": len(positive),
            "correct_work": sum(result["title_correct"] for result in positive),
            "evidence_at_1": sum(
                result["title_correct"] and result["evidence_rank"] == 1
                for result in positive
            ),
            "evidence_at_2": sum(
                result["title_correct"] and result["evidence_rank"] is not None
                for result in positive
            ),
            "negative_cases": len(negative),
            "correct_negative_status": sum(
                result["title_correct"] for result in negative
            ),
        },
        "results": results,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as output:
            json.dump(report, output, ensure_ascii=False, indent=2)
            output.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
