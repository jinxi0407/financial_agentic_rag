import json
import tempfile
import unittest
from pathlib import Path

from mysql_qa.faq_import import load_faq_records


def write_records(records):
    temporary_file = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".json", delete=False)
    with temporary_file:
        json.dump(records, temporary_file, ensure_ascii=False)
    return Path(temporary_file.name)


class FaqImportValidationTests(unittest.TestCase):
    def test_loads_the_structured_json_contract(self):
        path = write_records([{
            "question": "什么是营业收入？",
            "answer": "企业在日常经营活动中取得的收入。",
            "category": "财务指标",
            "keywords": ["营业收入", "收入"],
            "intent_id": "revenue_definition",
        }])
        self.addCleanup(path.unlink)

        records = load_faq_records(path)

        self.assertEqual(1, len(records))
        self.assertEqual("revenue_definition", records[0].intent_id)
        self.assertEqual('["营业收入","收入"]', records[0].keywords_json)

    def test_rejects_duplicate_intent_ids_before_database_access(self):
        record = {
            "question": "什么是营业收入？",
            "answer": "定义。",
            "category": "财务指标",
            "keywords": ["营业收入"],
            "intent_id": "revenue_definition",
        }
        second = dict(record, question="营业收入的含义是什么？")
        path = write_records([record, second])
        self.addCleanup(path.unlink)

        with self.assertRaisesRegex(ValueError, "duplicate intent_id"):
            load_faq_records(path)

    def test_rejects_malformed_keywords_before_database_access(self):
        path = write_records([{
            "question": "什么是营业收入？",
            "answer": "定义。",
            "category": "财务指标",
            "keywords": "营业收入",
            "intent_id": "revenue_definition",
        }])
        self.addCleanup(path.unlink)

        with self.assertRaisesRegex(ValueError, "keywords must be an array"):
            load_faq_records(path)


if __name__ == "__main__":
    unittest.main()
