import json
import unittest

import retrieval


def work(identifier, title, text="Початок твору."):
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


if __name__ == "__main__":
    unittest.main()
