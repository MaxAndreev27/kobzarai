"""Replay hybrid retrieval results with Qwen3, comparing the gate with no gate."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import evaluate_hybrid
import requests
import retrieval
import run_ollama

MODEL = "qwen3:8b"
SYSTEM_PROMPT = (
    "Ти — ШІ-співрозмовник, натхненний Тарасом Шевченком, не історична особа. "
    "Відповідай українською коротко, одним-двома реченнями. "
    "На питання про твори відповідай лише за наданими фрагментами. "
    "Якщо відповіді в них немає, скажи, що даних недостатньо; "
    "не вигадуй фактів чи цитат."
)
OPTIONS = {
    "num_ctx": 4096,
    "num_predict": 160,
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "seed": 42,
}


def prepare_cases(
    report: dict, corpus_data: bytes, cases_data: bytes
) -> list[tuple[dict, list[retrieval.SearchHit]]]:
    if hashlib.sha256(corpus_data).hexdigest() != report["corpus_sha256"]:
        raise ValueError("Corpus does not match the hybrid report")
    if hashlib.sha256(cases_data).hexdigest() != report["validation_sha256"]:
        raise ValueError("Validation cases do not match the hybrid report")
    corpus = retrieval.load_corpus(corpus_data)
    chunks = {
        (chunk["id"], chunk["start_char"], chunk["end_char"]): chunk
        for chunk in retrieval.build_chunks(corpus)
    }
    cases = [
        json.loads(line)
        for line in cases_data.decode("utf-8").splitlines()
        if line.strip()
    ]
    rows = report["validation_results"]
    if len(cases) != len(rows):
        raise ValueError("Validation case count does not match the report")
    prepared = []
    for case, row in zip(cases, rows, strict=True):
        if case["id"] != row["id"] or case["query"] != row["query"]:
            raise ValueError(f"Validation case does not match report: {case['id']}")
        status, work = retrieval.find_work(case["query"], corpus)
        if (
            status != row["actual_status"]
            or (work["id"] if work is not None else None) != row["actual_work_id"]
        ):
            raise ValueError(f"Title lookup does not match report: {case['id']}")
        if not isinstance(row["accepted"], bool) or row[
            "accepted"
        ] != evaluate_hybrid.decide(row["confidence"], report["threshold"]):
            raise ValueError(f"Inconsistent gate decision: {case['id']}")
        saved_hits = row["hybrid"]["hits"] if status == "matched" else []
        if status == "matched" and not 1 <= len(saved_hits) <= retrieval.DEFAULT_TOP_K:
            raise ValueError(f"Invalid source chunk count: {case['id']}")
        hits = []
        seen = set()
        for hit in saved_hits:
            key = (hit["id"], hit["start_char"], hit["end_char"])
            chunk = chunks.get(key)
            if chunk is None or chunk["id"] != row["actual_work_id"] or key in seen:
                raise ValueError(f"Invalid source chunk for {case['id']}: {key}")
            seen.add(key)
            if not isinstance(hit["score"], (int, float)) or not math.isfinite(
                hit["score"]
            ):
                raise ValueError(f"Invalid retrieval score for {case['id']}")
            hits.append(retrieval.SearchHit(chunk=chunk, score=hit["score"]))
        if row["accepted"] and not hits:
            raise ValueError(f"Accepted case has no source chunks: {case['id']}")
        prepared.append((row, hits))
    return prepared


def make_messages(row: dict, hits: list[retrieval.SearchHit]) -> list[dict[str, str]]:
    query = row["query"]
    if row["actual_status"] != "matched":
        query = retrieval.add_lookup_warning(query, row["actual_status"])
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": retrieval.add_chunk_context(query, hits)},
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hybrid-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    report_data = args.hybrid_report.read_bytes()
    report = json.loads(report_data)
    corpus_data = retrieval.CORPUS.read_bytes()
    cases_data = evaluate_hybrid.VALIDATION.read_bytes()
    cases = prepare_cases(report, corpus_data, cases_data)

    version = requests.get(f"{run_ollama.API}/version", timeout=10)
    version.raise_for_status()
    tags = requests.get(f"{run_ollama.API}/tags", timeout=10)
    tags.raise_for_status()
    model = next(
        (item for item in tags.json()["models"] if item["name"] == MODEL), None
    )
    if model is None:
        raise ValueError(f"Model unavailable: {MODEL}")
    metadata = {
        "type": "metadata",
        "hybrid_report_sha256": hashlib.sha256(report_data).hexdigest(),
        "validation_sha256": report["validation_sha256"],
        "corpus_sha256": report["corpus_sha256"],
        "threshold": report["threshold"],
        "model": model,
        "ollama_version": version.json()["version"],
        "system_prompt": SYSTEM_PROMPT,
        "options": OPTIONS,
        "think": False,
        "case_count": len(cases),
        "comparison": "ungated model response versus same response when gate accepts, fixed refusal otherwise",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        run_ollama.write_record(output, metadata)
        for index, (row, hits) in enumerate(cases, 1):
            messages = make_messages(row, hits)
            result = None
            try:
                response = requests.post(
                    f"{run_ollama.API}/chat",
                    json={
                        "model": MODEL,
                        "messages": messages,
                        "stream": False,
                        "think": False,
                        "keep_alive": "2m",
                        "options": OPTIONS,
                    },
                    timeout=300,
                )
                response.raise_for_status()
                result = response.json()
                if (
                    not isinstance(result, dict)
                    or result.get("done") is not True
                    or not isinstance(result.get("message"), dict)
                    or not isinstance(result["message"].get("content"), str)
                    or not result["message"]["content"].strip()
                ):
                    raise ValueError("Invalid, incomplete, or empty Ollama response")
                prompt_tokens = result.get("prompt_eval_count")
                if not isinstance(prompt_tokens, int) or prompt_tokens <= 0:
                    raise ValueError("Ollama did not report prompt token count")
                if prompt_tokens >= OPTIONS["num_ctx"] - OPTIONS["num_predict"]:
                    raise ValueError(
                        "Prompt may have been truncated by the context limit"
                    )
                placement_response = requests.get(f"{run_ollama.API}/ps", timeout=10)
                placement_response.raise_for_status()
                placement = next(
                    (
                        item
                        for item in placement_response.json()["models"]
                        if item.get("digest") == model["digest"]
                    ),
                    None,
                )
                if placement is None or placement.get("size_vram", 0) <= 0:
                    raise ValueError(f"GPU placement not confirmed: {placement}")
            except (requests.RequestException, ValueError, KeyError) as exc:
                run_ollama.write_record(
                    output,
                    {
                        "type": "error",
                        "case_id": row["id"],
                        "messages": messages,
                        "error": str(exc),
                        "response": result,
                    },
                )
                raise
            ungated = result["message"]["content"]
            run_ollama.write_record(
                output,
                {
                    "type": "result",
                    "case_id": row["id"],
                    "query": row["query"],
                    "messages": messages,
                    "retrieval_status": row["actual_status"],
                    "retrieval_hits": row.get("hybrid", {}).get("hits", []),
                    "confidence": row["confidence"],
                    "gate_accepted": row["accepted"],
                    "ungated_response": result,
                    "gated_answer": ungated
                    if row["accepted"]
                    else evaluate_hybrid.REFUSAL,
                    "gated_answer_source": "ungated_response"
                    if row["accepted"]
                    else "fixed_refusal",
                    "placement": placement,
                },
            )
            print(
                f"[{index}/{len(cases)}] {row['id']} {result.get('done_reason')}",
                flush=True,
            )
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
