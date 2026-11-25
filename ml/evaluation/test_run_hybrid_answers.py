import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import evaluate_hybrid
import run_hybrid_answers


def fixture():
    text = "Вітер гне дерево."
    corpus = (
        json.dumps(
            {
                "id": 1,
                "title": "Тополя",
                "source": "https://example.org/1",
                "text": text,
            },
            ensure_ascii=False,
        )
        + "\n"
    ).encode()
    cases = [
        {"id": "known", "query": "Що робить вітер у «Тополя»?", "evidence": "GOLD"},
        {"id": "unknown", "query": "Про що вірш «Невідомий твір»?"},
    ]
    cases_data = "".join(
        json.dumps(case, ensure_ascii=False) + "\n" for case in cases
    ).encode()
    report = {
        "corpus_sha256": hashlib.sha256(corpus).hexdigest(),
        "validation_sha256": hashlib.sha256(cases_data).hexdigest(),
        "threshold": 0.5,
        "validation_results": [
            {
                "id": cases[0]["id"],
                "query": cases[0]["query"],
                "actual_status": "matched",
                "actual_work_id": 1,
                "confidence": 0.6,
                "accepted": True,
                "hybrid": {
                    "hits": [
                        {"id": 1, "start_char": 0, "end_char": len(text), "score": 0.1}
                    ]
                },
            },
            {
                "id": cases[1]["id"],
                "query": cases[1]["query"],
                "actual_status": "not_found",
                "actual_work_id": None,
                "confidence": None,
                "accepted": False,
            },
        ],
    }
    return report, corpus, cases_data


class HybridAnswerTests(unittest.TestCase):
    def test_reconstructs_sources_without_gold_and_warns_for_unknown_title(self):
        report, corpus, cases = fixture()
        prepared = run_hybrid_answers.prepare_cases(report, corpus, cases)
        messages = run_hybrid_answers.make_messages(*prepared[0])
        self.assertIn("Вітер гне дерево.", messages[1]["content"])
        self.assertNotIn("GOLD", json.dumps(messages))
        unknown = run_hybrid_answers.make_messages(*prepared[1])
        self.assertIn("не доводить", unknown[1]["content"])
        self.assertNotIn("<джерела>", unknown[1]["content"])

    def test_source_guard_only_refuses_without_sources(self):
        report, corpus, cases = fixture()
        prepared = run_hybrid_answers.prepare_cases(report, corpus, cases)
        answer = "Відповідь моделі"
        self.assertEqual(
            run_hybrid_answers.source_guard_answer("matched", prepared[0][1], answer),
            answer,
        )
        self.assertEqual(
            run_hybrid_answers.source_guard_answer("not_found", [], answer),
            run_hybrid_answers.NO_SOURCE_REFUSAL,
        )
        self.assertEqual(
            run_hybrid_answers.source_guard_answer("ambiguous", [], answer),
            run_hybrid_answers.AMBIGUOUS_REFUSAL,
        )
        report["validation_results"][0]["hybrid"]["hits"] = []
        report["validation_results"][0]["confidence"] = None
        report["validation_results"][0]["accepted"] = False
        no_hits = run_hybrid_answers.prepare_cases(report, corpus, cases)[0]
        self.assertEqual(
            run_hybrid_answers.source_guard_answer("matched", no_hits[1], answer),
            run_hybrid_answers.NO_SOURCE_REFUSAL,
        )

    def test_rejects_mismatched_hashes_offsets_and_decisions(self):
        report, corpus, cases = fixture()
        with self.assertRaisesRegex(ValueError, "Corpus does not match"):
            run_hybrid_answers.prepare_cases(report, corpus + b" ", cases)
        with self.assertRaisesRegex(ValueError, "Validation cases do not match"):
            run_hybrid_answers.prepare_cases(report, corpus, cases + b" ")
        broken = copy.deepcopy(report)
        broken["validation_results"][0]["hybrid"]["hits"][0]["end_char"] -= 1
        with self.assertRaisesRegex(ValueError, "Invalid source chunk"):
            run_hybrid_answers.prepare_cases(broken, corpus, cases)
        broken = copy.deepcopy(report)
        broken["validation_results"][0]["accepted"] = False
        with self.assertRaisesRegex(ValueError, "Inconsistent gate decision"):
            run_hybrid_answers.prepare_cases(broken, corpus, cases)

    def test_reuses_generation_for_accepted_case_and_refuses_other_case(self):
        report, corpus, cases = fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus_path, cases_path, report_path, output_path = (
                root / name for name in ("corpus", "cases", "report", "output")
            )
            corpus_path.write_bytes(corpus)
            cases_path.write_bytes(cases)
            report_path.write_text(json.dumps(report), encoding="utf-8")
            version, tags, placement, response = (Mock() for _ in range(4))
            version.json.return_value = {"version": "test"}
            tags.json.return_value = {
                "models": [{"name": "qwen3:8b", "digest": "digest"}]
            }
            placement.json.return_value = {
                "models": [{"digest": "digest", "size_vram": 1}]
            }
            response.json.return_value = {
                "done": True,
                "done_reason": "stop",
                "message": {"content": "Відповідь моделі"},
                "prompt_eval_count": 30,
            }
            argv = [
                "run_hybrid_answers.py",
                "--hybrid-report",
                str(report_path),
                "--output",
                str(output_path),
            ]
            with (
                patch.object(run_hybrid_answers.retrieval, "CORPUS", corpus_path),
                patch.object(
                    run_hybrid_answers.evaluate_hybrid, "VALIDATION", cases_path
                ),
                patch.object(sys, "argv", argv),
                patch.object(
                    run_hybrid_answers.requests,
                    "get",
                    side_effect=[version, tags, placement, placement],
                ),
                patch.object(
                    run_hybrid_answers.requests, "post", return_value=response
                ) as post,
            ):
                run_hybrid_answers.main()
            records = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(post.call_count, 2)
            self.assertEqual(
                records[1]["gated_answer"],
                records[1]["ungated_response"]["message"]["content"],
            )
            self.assertEqual(records[2]["gated_answer"], evaluate_hybrid.REFUSAL)
            self.assertEqual(
                records[2]["ungated_response"]["message"]["content"], "Відповідь моделі"
            )
            self.assertEqual(records[1]["source_guard_answer"], "Відповідь моделі")
            self.assertEqual(
                records[2]["source_guard_answer"], run_hybrid_answers.NO_SOURCE_REFUSAL
            )


if __name__ == "__main__":
    unittest.main()
