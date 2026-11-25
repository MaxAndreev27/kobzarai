import json
import unittest
from pathlib import Path

import retrieval


def work(
    identifier: int, title: str, text: str = "Початок твору."
) -> retrieval.Work:
    return {
        "id": identifier,
        "title": title,
        "source": f"https://example.org/{identifier}",
        "text": text,
    }


class RetrievalTests(unittest.TestCase):
    def test_title_and_inflected_title_match(self):
        corpus = [work(1, "Катерина"), work(2, "Заповіт")]
        for query, expected in [
            ("Про що поема «Катерина»?", 1),
            ("Які теми «Катерини»?", 1),
            ("Розкажи про «Заповіту».", 2),
        ]:
            with self.subTest(query=query):
                status, result = retrieval.find_work(query, corpus)
                self.assertEqual(status, "matched")
                self.assertIsNotNone(result)
                assert result is not None
                self.assertEqual(result["id"], expected)

    def test_unknown_boundary_and_ambiguous_names(self):
        corpus = [work(1, "Катерина"), work(2, "Кавказ")]
        self.assertEqual(
            retrieval.find_work("Катеринбург", corpus), ("not_found", None)
        )
        self.assertEqual(
            retrieval.find_work("«Катерина» і «Кавказ»", corpus), ("ambiguous", None)
        )
        query = "Чи існує твір «Невідомий»?"
        self.assertEqual(retrieval.find_work(query, corpus), ("not_found", None))
        self.assertIn("не доводить", retrieval.add_lookup_warning(query, "not_found"))

    def test_longer_overlapping_title_wins(self):
        corpus = [work(1, "Берег"), work(2, "Берег мрій")]
        status, result = retrieval.find_work("Про «Берег мрій»", corpus)
        self.assertEqual(status, "matched")
        assert result is not None
        self.assertEqual(result["id"], 2)

    def test_context_is_bounded_and_identifies_source(self):
        entry = work(1, "Катерина", ("Перший рядок.\n" * 200))
        fragment = retrieval.excerpt(entry)
        context = retrieval.add_context("Про що твір?", entry)
        self.assertLessEqual(len(fragment), retrieval.MAX_EXCERPT_CHARS)
        self.assertTrue(entry["text"].startswith(fragment))
        self.assertIn(entry["source"], context)
        self.assertIn("лише початковий фрагмент", context)

    def test_corpus_validation(self):
        valid = work(1, "Катерина")
        raw = json.dumps(valid, ensure_ascii=False).encode()
        self.assertEqual(retrieval.load_corpus(raw)[0]["id"], 1)
        with self.assertRaises(ValueError):
            retrieval.load_corpus(b"")
        with self.assertRaises(ValueError):
            retrieval.load_corpus(raw + b"\n" + raw)
        with self.assertRaises(ValueError):
            retrieval.load_corpus(b'{"id": 1}')

    def test_retrieval_cases_find_the_expected_source_fragment(self):
        project = Path(__file__).resolve().parents[1]
        corpus = retrieval.load_corpus(retrieval.CORPUS.read_bytes())
        chunks = retrieval.build_chunks(corpus)
        cases = [
            json.loads(line)
            for line in (project / "evaluation/retrieval_cases.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        self.assertEqual(len(cases), 8)
        for case in cases:
            with self.subTest(case=case["id"]):
                work = next(item for item in corpus if item["id"] == case["work_id"])
                status, found = retrieval.find_work(case["query"], corpus)
                self.assertEqual(status, "matched")
                self.assertIsNotNone(found)
                assert found is not None
                self.assertEqual(found["id"], work["id"])
                hits = retrieval.search_chunks(
                    case["query"],
                    chunks,
                    work_id=work["id"],
                    exclude_title=work["title"],
                )
                self.assertTrue(hits)
                excerpt = "\n".join(hit["chunk"]["text"] for hit in hits)
                self.assertIn(case["evidence"], excerpt)


if __name__ == "__main__":
    unittest.main()
