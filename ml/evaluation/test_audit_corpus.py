import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import audit_corpus


class CorpusAuditTests(unittest.TestCase):
    def test_overlap_ignores_punctuation_and_reports_containment(self):
        lines = [f"Довгий рядок номер {i} для прикладу" for i in range(6)]
        works = [
            {"id": 1, "title": "Перший", "source": "a", "text": "\n".join(lines)},
            {
                "id": 2,
                "title": "Другий",
                "source": "b",
                "text": "\n".join(line.upper() + "!" for line in lines),
            },
            {"id": 3, "title": "Третій", "source": "c", "text": "Інший текст"},
        ]
        pairs = audit_corpus.overlaps(works)
        self.assertEqual(len(pairs), 1)
        self.assertEqual((pairs[0]["first_id"], pairs[0]["second_id"]), (1, 2))
        self.assertEqual(pairs[0]["shorter_overlap"], 1.0)

    def test_quarantine_keeps_original_and_writes_hashed_copy(self):
        records = [
            {"id": 1, "title": "Перший", "source": "a", "text": "один"},
            {"id": 2, "title": "Другий", "source": "b", "text": "два"},
        ]
        data = "".join(
            json.dumps(r, ensure_ascii=False) + "\n" for r in records
        ).encode()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus, report, filtered = (root / name for name in ("in", "report", "out"))
            corpus.write_bytes(data)
            argv = [
                "audit_corpus.py",
                "--corpus",
                str(corpus),
                "--report",
                str(report),
                "--filtered-output",
                str(filtered),
                "--quarantine-id",
                "2",
            ]
            with patch.object(sys, "argv", argv):
                audit_corpus.main()
            result = json.loads(report.read_text())
            self.assertEqual(corpus.read_bytes(), data)
            self.assertEqual(result["quarantined"][0]["id"], 2)
            self.assertEqual(
                result["filtered_sha256"],
                hashlib.sha256(filtered.read_bytes()).hexdigest(),
            )
            self.assertEqual(json.loads(filtered.read_bytes())["id"], 1)
            with patch.object(sys, "argv", argv), self.assertRaises(FileExistsError):
                audit_corpus.main()

    def test_unknown_quarantine_id_does_not_create_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus, report, filtered = (root / name for name in ("in", "report", "out"))
            corpus.write_text(
                json.dumps({"id": 1, "title": "Один", "source": "a", "text": "текст"})
            )
            argv = [
                "audit_corpus.py",
                "--corpus",
                str(corpus),
                "--report",
                str(report),
                "--filtered-output",
                str(filtered),
                "--quarantine-id",
                "2",
            ]
            with (
                patch.object(sys, "argv", argv),
                self.assertRaisesRegex(ValueError, "Unknown quarantine"),
            ):
                audit_corpus.main()
            self.assertFalse(report.exists())
            self.assertFalse(filtered.exists())


if __name__ == "__main__":
    unittest.main()
