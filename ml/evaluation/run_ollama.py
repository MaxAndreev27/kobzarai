"""Run the development chat evaluation against a local Ollama server."""

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import requests

DATASET = Path(__file__).with_name("chat_uk.jsonl")
API = "http://127.0.0.1:11434/api"
SYSTEM_PROMPT = (
    "Ти — літературний ШІ-образ, натхненний Тарасом Шевченком, "
    "а не справжня історична особа. Спілкуйся українською мовою. "
    "Відповідай на запитання користувача зрозуміло і за потреби образно. "
    "Не видавай вигадані факти, цитати або власні припущення за достовірні. "
    "Якщо не знаєш відповіді або не можеш перевірити факт, скажи про це."
)


def write_record(output, record: dict) -> None:
    output.write(json.dumps(record, ensure_ascii=False) + "\n")
    output.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="qwen3:1.7b")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.threads < 1:
        parser.error("--threads must be positive")

    data = DATASET.read_bytes()
    cases = [
        json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()
    ]
    cases = [case for case in cases if case["split"] == "development"]
    if args.limit is not None:
        cases = cases[: args.limit]
    if not cases:
        raise ValueError("No development cases found")

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
    metadata = {
        "type": "metadata",
        "created_at": datetime.now(UTC).isoformat(),
        "dataset_sha256": hashlib.sha256(data).hexdigest(),
        "split": "development",
        "case_count": len(cases),
        "model": model,
        "ollama_version": version_response.json()["version"],
        "system_prompt": SYSTEM_PROMPT,
        "options": options,
        "requested_device": args.device,
        "think": False,
        "rag": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        write_record(output, metadata)
        for index, case in enumerate(cases, 1):
            messages = [{"role": "system", "content": SYSTEM_PROMPT}, *case["messages"]]
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
                },
            )
            reason = result.get("done_reason", "unknown")
            print(f"  done_reason={reason}", flush=True)
    print(f"Saved results to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
