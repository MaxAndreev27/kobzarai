"""Report overlapping corpus records and optionally create a quarantined copy."""

import argparse
import hashlib
import json
from pathlib import Path

import retrieval

MIN_LINE_CHARS = 15
MIN_LINES = 5
MIN_OVERLAP = 0.8


def line_keys(text: str) -> set[str]:
    return {
        retrieval.normalize(line)
        for line in text.splitlines()
        if len(line.strip()) >= MIN_LINE_CHARS
    } - {""}


def overlaps(corpus: list[retrieval.Work]) -> list[dict]:
    lines = [line_keys(work["text"]) for work in corpus]
    pairs = []
    for i, first in enumerate(corpus):
        for j in range(i + 1, len(corpus)):
            second = corpus[j]
            shorter = min(len(lines[i]), len(lines[j]))
            if shorter < MIN_LINES:
                continue
            shared = len(lines[i] & lines[j])
            if shared / shorter >= MIN_OVERLAP:
                pairs.append(
                    {
                        "first_id": first["id"],
                        "first_title": first["title"],
                        "second_id": second["id"],
                        "second_title": second["title"],
                        "first_lines": len(lines[i]),
                        "second_lines": len(lines[j]),
                        "shared_lines": shared,
                        "shorter_overlap": shared / shorter,
                    }
                )
    return sorted(pairs, key=lambda pair: -pair["shorter_overlap"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=retrieval.CORPUS)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--filtered-output", type=Path)
    parser.add_argument("--quarantine-id", type=int, action="append", default=[])
    args = parser.parse_args()
    if bool(args.filtered_output) != bool(args.quarantine_id):
        parser.error("--filtered-output and --quarantine-id must be used together")
    if len(args.quarantine_id) != len(set(args.quarantine_id)):
        parser.error("Duplicate quarantine ID")
    if args.filtered_output and args.filtered_output.resolve() == args.report.resolve():
        parser.error("--report and --filtered-output must be different files")
    if args.report.exists() or (args.filtered_output and args.filtered_output.exists()):
        raise FileExistsError("Output already exists")

    data = args.corpus.read_bytes()
    corpus = retrieval.load_corpus(data)
    excluded = set(args.quarantine_id)
    missing = excluded - {work["id"] for work in corpus}
    if missing:
        raise ValueError(f"Unknown quarantine IDs: {sorted(missing)}")
    filtered_data = b"".join(
        line
        for line in data.splitlines(keepends=True)
        if not line.strip() or json.loads(line)["id"] not in excluded
    )
    if args.filtered_output:
        retrieval.load_corpus(filtered_data)
    report = {
        "corpus_sha256": hashlib.sha256(data).hexdigest(),
        "record_count": len(corpus),
        "criteria": {
            "minimum_line_chars": MIN_LINE_CHARS,
            "minimum_lines": MIN_LINES,
            "minimum_shorter_overlap": MIN_OVERLAP,
            "normalization": "casefold and word tokens; punctuation ignored",
        },
        "overlaps": overlaps(corpus),
        "quarantined": [
            {"id": work["id"], "title": work["title"], "source": work["source"]}
            for work in corpus
            if work["id"] in excluded
        ],
        "filtered_sha256": (
            hashlib.sha256(filtered_data).hexdigest() if args.filtered_output else None
        ),
    }
    if args.filtered_output:
        args.filtered_output.parent.mkdir(parents=True, exist_ok=True)
        with args.filtered_output.open("xb") as output:
            output.write(filtered_data)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x", encoding="utf-8") as output:
        json.dump(report, output, ensure_ascii=False, indent=2)
        output.write("\n")
    print(f"Found {len(report['overlaps'])} overlap candidates; report: {args.report}")


if __name__ == "__main__":
    main()
