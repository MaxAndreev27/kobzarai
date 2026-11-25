import unittest

import evaluate_hybrid
import retrieval


def hit(start: int) -> retrieval.SearchHit:
    return {
        "chunk": {
            "id": 1,
            "title": "Приклад",
            "source": "https://example.org/1",
            "start_char": start,
            "end_char": start + 5,
            "text": "текст",
            "is_first": start == 0,
        },
        "score": 1.0,
    }


class HybridEvaluationTests(unittest.TestCase):
    def test_rrf_combines_rankings_and_rejects_duplicate_chunks(self):
        lexical = [hit(0), hit(10)]
        dense = [hit(20), hit(10)]
        fused = evaluate_hybrid.fuse(lexical, dense)
        self.assertEqual([item["chunk"]["start_char"] for item in fused], [10, 0])
        self.assertEqual(evaluate_hybrid.fuse([], []), [])
        with self.assertRaises(ValueError):
            evaluate_hybrid.fuse([hit(0), hit(0)], dense)
        with self.assertRaises(ValueError):
            evaluate_hybrid.fuse(lexical, dense, top_k=0)

    def test_calibration_threshold_and_refusal(self):
        threshold = evaluate_hybrid.choose_threshold(
            [(0.8, True), (0.6, True), (0.4, False), (0.2, False)]
        )
        self.assertEqual(threshold, 0.6)
        self.assertTrue(evaluate_hybrid.decide(0.6, threshold))
        self.assertFalse(evaluate_hybrid.decide(0.4, threshold))
        self.assertFalse(evaluate_hybrid.decide(None, threshold))
        with self.assertRaises(ValueError):
            evaluate_hybrid.choose_threshold([(0.8, True), (None, False)])

    def test_metrics_distinguish_accepted_without_expected_evidence(self):
        rows = [
            {
                "expected_status": "matched",
                "answerable": True,
                "actual_work_id": 1,
                "expected_work_id": 1,
                "confidence": 0.8,
                "hybrid": {"evidence_rank": None},
                "lexical": {"evidence_rank": None},
                "dense": {"evidence_rank": None},
            },
            {
                "expected_status": "matched",
                "answerable": False,
                "confidence": 0.2,
            },
            {"expected_status": "not_found", "confidence": None},
        ]
        metrics = evaluate_hybrid.summarize(rows, 0.6)
        self.assertTrue(rows[0]["accepted"])
        self.assertEqual(metrics["gate"]["accepted_with_expected_evidence"], 0)
        self.assertEqual(metrics["gate"]["unanswerable_accepted"], 0)
        self.assertEqual(metrics["gate"]["title_negative_rejected"], 1)
        self.assertEqual(rows[1]["refusal"], evaluate_hybrid.REFUSAL)


if __name__ == "__main__":
    unittest.main()
