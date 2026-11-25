import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import retrieval
import run_evidence_answers
import run_hybrid_answers
from test_run_hybrid_answers import fixture


class EvidenceAnswerTests(unittest.TestCase):
    def test_selected_lines_use_exact_source_offsets(self):
        text = "Перший рядок.\nДругий рядок.\nТретій рядок."
        hit: retrieval.SearchHit = {
            "chunk": {
                "id": 1,
                "title": "Приклад",
                "source": "https://example.org/1",
                "start_char": 100,
                "end_char": 100 + len(text),
                "text": text,
                "is_first": False,
            },
            "score": 1.0,
        }
        selected = run_evidence_answers.select_evidence(
            '{"chunk":1,"start_line":2,"end_line":3}', [hit]
        )
        self.assertIsNotNone(selected)
        assert selected is not None
        chunk = selected["chunk"]
        self.assertEqual(chunk["text"], "Другий рядок.\nТретій рядок.")
        start = chunk["start_char"] - hit["chunk"]["start_char"]
        end = chunk["end_char"] - hit["chunk"]["start_char"]
        self.assertEqual(text[start:end], chunk["text"])
        self.assertIsNone(
            run_evidence_answers.select_evidence(
                '{"chunk":null,"start_line":null,"end_line":null}', [hit]
            )
        )

    def test_invalid_or_empty_selection_cannot_be_used_as_evidence(self):
        hit: retrieval.SearchHit = {
            "chunk": {
                "id": 1,
                "title": "Приклад",
                "source": "https://example.org/1",
                "start_char": 0,
                "end_char": 8,
                "text": "\n" * 7 + "а",
                "is_first": True,
            },
            "score": 1.0,
        }
        for content in (
            "not json",
            '{"chunk":true,"start_line":1,"end_line":1}',
            '{"chunk":2,"start_line":1,"end_line":1}',
            '{"chunk":1,"start_line":3,"end_line":2}',
            '{"chunk":1,"start_line":1,"end_line":7}',
            '{"chunk":1,"start_line":1,"end_line":1}',
            '{"chunk":null,"start_line":1,"end_line":null}',
            '{"chunk":1,"start_line":1}',
        ):
            with self.subTest(content=content), self.assertRaises(ValueError):
                run_evidence_answers.select_evidence(content, [hit])

    def test_two_stages_and_no_source_refusal_use_same_baseline_inputs(self):
        report, corpus, case_data = fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus_path, cases_path, report_path, baseline_path, output_path = (
                root / name
                for name in ("corpus", "cases", "report", "baseline", "output")
            )
            corpus_path.write_bytes(corpus)
            cases_path.write_bytes(case_data)
            report_data = json.dumps(report).encode()
            report_path.write_bytes(report_data)
            prepared = run_hybrid_answers.prepare_cases(report, corpus, case_data)
            baseline_metadata = {
                "hybrid_report_sha256": hashlib.sha256(report_data).hexdigest(),
                "validation_sha256": report["validation_sha256"],
                "corpus_sha256": report["corpus_sha256"],
                "model": {"digest": "digest"},
                "ollama_version": "test",
                "options": run_hybrid_answers.OPTIONS,
                "system_prompt": run_hybrid_answers.SYSTEM_PROMPT,
                "think": False,
            }
            records = [baseline_metadata]
            for row, hits in prepared:
                records.append(
                    {
                        "type": "result",
                        "case_id": row["id"],
                        "query": row["query"],
                        "messages": run_hybrid_answers.make_messages(row, hits),
                        "retrieval_hits": row.get("hybrid", {}).get("hits", []),
                        "ungated_response": {
                            "message": {"content": "Базова відповідь"}
                        },
                        "source_guard_answer": run_hybrid_answers.source_guard_answer(
                            row["actual_status"], hits, "Базова відповідь"
                        ),
                    }
                )
            baseline_path.write_text(
                "".join(
                    json.dumps(record, ensure_ascii=False) + "\n" for record in records
                ),
                encoding="utf-8",
            )
            version, tags, placement = (Mock() for _ in range(3))
            version.json.return_value = {"version": "test"}
            tags.json.return_value = {
                "models": [{"name": "qwen3:8b", "digest": "digest"}]
            }
            placement.json.return_value = {
                "models": [{"digest": "digest", "size_vram": 1}]
            }
            selection, answer = (Mock() for _ in range(2))
            selection.json.return_value = {
                "done": True,
                "done_reason": "stop",
                "message": {"content": '{"chunk":1,"start_line":1,"end_line":1}'},
                "prompt_eval_count": 30,
            }
            answer.json.return_value = {
                "done": True,
                "done_reason": "stop",
                "message": {"content": "Вітер гне дерево."},
                "prompt_eval_count": 30,
            }
            argv = [
                "run_evidence_answers.py",
                "--hybrid-report",
                str(report_path),
                "--cases",
                str(cases_path),
                "--baseline",
                str(baseline_path),
                "--output",
                str(output_path),
            ]
            with (
                patch.object(run_hybrid_answers.retrieval, "CORPUS", corpus_path),
                patch.object(sys, "argv", argv),
                patch.object(
                    run_evidence_answers.requests,
                    "get",
                    side_effect=[version, tags, placement, placement],
                ),
                patch.object(
                    run_evidence_answers.requests,
                    "post",
                    side_effect=[selection, answer],
                ) as post,
            ):
                run_evidence_answers.main()
            output = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(post.call_count, 2)
            self.assertEqual(
                post.call_args_list[0].kwargs["json"]["format"],
                run_evidence_answers.SELECTION_SCHEMA,
            )
            self.assertEqual(output[1]["outcome"], "answered")
            self.assertEqual(output[1]["evidence"]["text"], "Вітер гне дерево.")
            self.assertEqual(output[1]["answer"], "Вітер гне дерево.")
            self.assertEqual(output[2]["outcome"], "no_source")
            self.assertEqual(output[2]["answer"], run_hybrid_answers.NO_SOURCE_REFUSAL)

            argv[-1] = str(root / "invalid-output")
            selection.json.return_value["message"]["content"] = (
                '{"chunk":1,"start_line":1,"end_line":7}'
            )
            with (
                patch.object(run_hybrid_answers.retrieval, "CORPUS", corpus_path),
                patch.object(sys, "argv", argv),
                patch.object(
                    run_evidence_answers.requests,
                    "get",
                    side_effect=[version, tags, placement],
                ),
                patch.object(
                    run_evidence_answers.requests, "post", return_value=selection
                ) as post,
            ):
                run_evidence_answers.main()
            invalid = [
                json.loads(line)
                for line in (root / "invalid-output")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(post.call_count, 1)
            self.assertEqual(invalid[1]["outcome"], "invalid_selection")
            self.assertIn("Invalid selection line range", invalid[1]["selection_error"])
            self.assertNotIn("evidence", invalid[1])


if __name__ == "__main__":
    unittest.main()
