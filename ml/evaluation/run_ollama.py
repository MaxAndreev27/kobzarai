"""Run the development chat evaluation against a local Ollama server."""

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import requests
import retrieval

DATASET = Path(__file__).with_name("chat_uk.jsonl")
API = "http://127.0.0.1:11434/api"
SYSTEM_PROMPT = (
    "Ти — літературний ШІ-образ, натхненний Тарасом Шевченком, "
    "а не справжня історична особа. Спілкуйся українською мовою. "
    "Відповідай на запитання користувача зрозуміло і за потреби образно. "
    "Не видавай вигадані факти, цитати або власні припущення за достовірні. "
    "Якщо не знаєш відповіді або не можеш перевірити факт, скажи про це."
)
QUOTE_SYSTEM_PROMPT = (
    SYSTEM_PROMPT
    + " Відповідай коротко, лише за наданим джерелом. Якщо запитують назву "
    "твору, наведи її точно, не додаючи непідтверджених дат чи фактів."
)
QUOTE_REFUSALS = {
    "no_quote": "Наведіть одну цитату в лапках.",
    "multiple_quotes": "Наведіть одну цитату за раз.",
    "quote_too_short": "Наведіть цитату щонайменше з чотирьох слів.",
    "not_found": "Не вдалося знайти цей уривок у доступних текстах; це не доводить, що він не існує.",
    "ambiguous": "Цей уривок знайдено в кількох записах корпусу; уточніть назву твору й скористайтеся пошуком за назвою.",
    "no_relevant_chunk": "Уривок знайдено у творі, але жоден доступний фрагмент не містить його повністю.",
}


def write_record(output, record: dict) -> None:
    output.write(json.dumps(record, ensure_ascii=False) + "\n")
    output.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--corpus", type=Path, default=retrieval.CORPUS)
    parser.add_argument("--model", default="qwen3:1.7b")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--rag", action="store_true", help="Find works by title")
    parser.add_argument(
        "--quote-search",
        action="store_true",
        help="Find one quoted passage instead of a title",
    )
    parser.add_argument("--case-id", action="append", help="Select development case")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.quote_search and not args.rag:
        parser.error("--quote-search requires --rag")

    data = args.dataset.read_bytes()
    cases = [
        json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()
    ]
    cases = [case for case in cases if case["split"] == "development"]
    if args.case_id:
        unknown = set(args.case_id) - {case["id"] for case in cases}
        if unknown:
            parser.error(
                f"Unknown or non-development case IDs: {', '.join(sorted(unknown))}"
            )
        cases = [case for case in cases if case["id"] in args.case_id]
    if args.limit is not None:
        cases = cases[: args.limit]
    if not cases:
        raise ValueError("No development cases found")
    corpus_data = args.corpus.read_bytes() if args.rag else None
    corpus = retrieval.load_corpus(corpus_data) if corpus_data is not None else None
    chunks = retrieval.build_chunks(corpus) if corpus is not None else None

    version_response = requests.get(f"{API}/version", timeout=10)
    version_response.raise_for_status()
    tags_response = requests.get(f"{API}/tags", timeout=10)
    tags_response.raise_for_status()
    model = next(
        (item for item in tags_response.json()["models"] if item["name"] == args.model),
        None,
    )
    if model is None:
        raise ValueError(f"Model {args.model!r} is not available in local Ollama")

    options = {
        "num_thread": args.threads,
        "num_ctx": 2048,
        "num_predict": 256,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "seed": 42,
    }
    if args.device == "cpu":
        options["num_gpu"] = 0
    system_prompt = QUOTE_SYSTEM_PROMPT if args.quote_search else SYSTEM_PROMPT
    metadata = {
        "type": "metadata",
        "created_at": datetime.now(UTC).isoformat(),
        "dataset_sha256": hashlib.sha256(data).hexdigest(),
        "split": "development",
        "case_count": len(cases),
        "model": model,
        "ollama_version": version_response.json()["version"],
        "system_prompt": system_prompt,
        "options": options,
        "requested_device": args.device,
        "think": False,
        "rag": args.rag,
        "quote_search": args.quote_search,
        "corpus_sha256": (
            hashlib.sha256(corpus_data).hexdigest() if corpus_data is not None else None
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        write_record(output, metadata)
        for index, case in enumerate(cases, 1):
            messages = [{"role": "system", "content": system_prompt}, *case["messages"]]
            retrieval_info: dict[str, object] | None = None
            if corpus is not None and chunks is not None:
                hits: list[retrieval.SearchHit] | None = None
                if args.quote_search:
                    status, hits = retrieval.search_quote(
                        case["messages"][-1]["content"], corpus, chunks
                    )
                    retrieval_info = {"status": status}
                    if status != "matched":
                        write_record(
                            output,
                            {
                                "type": "refusal",
                                "case": case,
                                "messages": messages,
                                "answer": QUOTE_REFUSALS[status],
                                "retrieval": retrieval_info,
                            },
                        )
                        print(
                            f"[{index}/{len(cases)}] {case['id']} {status}", flush=True
                        )
                        continue
                else:
                    status, work = retrieval.find_work(
                        case["messages"][-1]["content"], corpus
                    )
                    retrieval_info = {"status": status}
                    if work is not None:
                        hits = retrieval.search_chunks(
                            case["messages"][-1]["content"],
                            chunks,
                            work_id=work["id"],
                            exclude_title=work["title"],
                        )
                        retrieval_info["status"] = (
                            "matched" if hits else "no_relevant_chunk"
                        )
                    else:
                        messages[-1] = {
                            "role": "user",
                            "content": retrieval.add_lookup_warning(
                                messages[-1]["content"], status
                            ),
                        }
                if hits is not None:
                    retrieval_info["hits"] = [
                        {
                            "id": hit["chunk"]["id"],
                            "title": hit["chunk"]["title"],
                            "source": hit["chunk"]["source"],
                            "start_char": hit["chunk"]["start_char"],
                            "end_char": hit["chunk"]["end_char"],
                            "score": hit["score"],
                        }
                        for hit in hits
                    ]
                    messages[-1] = {
                        "role": "user",
                        "content": retrieval.add_chunk_context(
                            messages[-1]["content"], hits
                        )
                        + (
                            f"\n\nНазва знайденого твору: «{hits[0]['chunk']['title']}»."
                            if args.quote_search
                            else ""
                        ),
                    }
            payload = {
                "model": args.model,
                "messages": messages,
                "stream": False,
                "think": False,
                "keep_alive": "2m",
                "options": options,
            }
            print(f"[{index}/{len(cases)}] {case['id']}", flush=True)
            try:
                response = requests.post(f"{API}/chat", json=payload, timeout=300)
                response.raise_for_status()
                result = response.json()
            except requests.RequestException as exc:
                write_record(
                    output,
                    {
                        "type": "error",
                        "case": case,
                        "messages": messages,
                        "error": str(exc),
                        "retrieval": retrieval_info,
                    },
                )
                print(
                    f"Request failed; remaining cases were not run: {exc}",
                    file=sys.stderr,
                )
                return 1
            if (
                not isinstance(result, dict)
                or result.get("done") is not True
                or not isinstance(result.get("message"), dict)
                or not isinstance(result["message"].get("content"), str)
                or not result["message"]["content"].strip()
            ):
                write_record(
                    output,
                    {
                        "type": "error",
                        "case": case,
                        "messages": messages,
                        "error": "Invalid, incomplete, or empty Ollama response",
                        "response": result,
                        "retrieval": retrieval_info,
                    },
                )
                print(
                    "Invalid Ollama response; remaining cases were not run",
                    file=sys.stderr,
                )
                return 1
            placement = None
            if args.device == "gpu":
                try:
                    placement_response = requests.get(f"{API}/ps", timeout=10)
                    placement_response.raise_for_status()
                    placement = next(
                        (
                            item
                            for item in placement_response.json()["models"]
                            if item.get("digest") == model["digest"]
                        ),
                        None,
                    )
                except requests.RequestException as exc:
                    write_record(
                        output,
                        {
                            "type": "error",
                            "case": case,
                            "messages": messages,
                            "error": f"Could not verify GPU placement: {exc}",
                            "response": result,
                            "retrieval": retrieval_info,
                        },
                    )
                    print(f"Could not verify GPU placement: {exc}", file=sys.stderr)
                    return 1
                if placement is None or placement.get("size_vram", 0) <= 0:
                    write_record(
                        output,
                        {
                            "type": "error",
                            "case": case,
                            "messages": messages,
                            "error": "Ollama did not report GPU memory for this model",
                            "response": result,
                            "placement": placement,
                            "retrieval": retrieval_info,
                        },
                    )
                    print("GPU placement not confirmed; stopping", file=sys.stderr)
                    return 1
            write_record(
                output,
                {
                    "type": "result",
                    "case": case,
                    "messages": messages,
                    "response": result,
                    "placement": placement,
                    "retrieval": retrieval_info,
                },
            )
            reason = result.get("done_reason", "unknown")
            print(f"  done_reason={reason}", flush=True)
    print(f"Saved results to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
