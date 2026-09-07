"""Idempotent importer for the versioned financial FAQ JSON dataset."""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from base.config import config
from base.logger import logger
from mysql_qa.cache.redis_client import RedisClient
from mysql_qa.db.mysql_client import MysqlClient


REQUIRED_FIELDS = ("question", "answer", "category", "keywords", "intent_id")
FAQ_INDEX_KEYS = (
    f"{config.REDIS_KEY_PREFIX}faq:origin_questions",
    f"{config.REDIS_KEY_PREFIX}faq:questions",
)


@dataclass(frozen=True)
class FaqRecord:
    question: str
    answer: str
    category: str
    keywords: tuple[str, ...]
    intent_id: str

    @property
    def keywords_json(self):
        return json.dumps(list(self.keywords), ensure_ascii=False, separators=(",", ":"))


@dataclass
class FaqImportResult:
    json_rows: int
    inserted: int = 0
    updated: int = 0
    skipped: int = 0
    cache_keys_deleted: int = 0

    def to_dict(self):
        return asdict(self)


def load_faq_records(json_path):
    """Load and validate the JSON contract before touching MySQL."""
    path = Path(json_path)
    with path.open("r", encoding="utf-8") as file:
        raw_records = json.load(file)

    if not isinstance(raw_records, list):
        raise ValueError("FAQ JSON must be an array")

    records = []
    seen_intent_ids = set()
    seen_questions = set()
    for index, raw_record in enumerate(raw_records, start=1):
        if not isinstance(raw_record, dict):
            raise ValueError(f"FAQ record {index} must be an object")
        missing = [field for field in REQUIRED_FIELDS if field not in raw_record]
        if missing:
            raise ValueError(f"FAQ record {index} is missing fields: {', '.join(missing)}")

        values = {field: raw_record[field] for field in REQUIRED_FIELDS if field != "keywords"}
        if any(not isinstance(value, str) or not value.strip() for value in values.values()):
            raise ValueError(f"FAQ record {index} has an empty or invalid string field")
        if not isinstance(raw_record["keywords"], list) or any(
            not isinstance(keyword, str) or not keyword.strip()
            for keyword in raw_record["keywords"]
        ):
            raise ValueError(f"FAQ record {index} keywords must be an array of non-empty strings")

        intent_id = raw_record["intent_id"].strip()
        question = raw_record["question"].strip()
        if intent_id in seen_intent_ids:
            raise ValueError(f"duplicate intent_id in FAQ JSON: {intent_id}")
        if question in seen_questions:
            raise ValueError(f"duplicate question in FAQ JSON: {question}")
        seen_intent_ids.add(intent_id)
        seen_questions.add(question)
        records.append(
            FaqRecord(
                question=question,
                answer=raw_record["answer"].strip(),
                category=raw_record["category"].strip(),
                keywords=tuple(keyword.strip() for keyword in raw_record["keywords"]),
                intent_id=intent_id,
            )
        )
    return records


def _normalize_keywords(value):
    if value is None:
        return None
    if isinstance(value, str):
        value = json.loads(value)
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _row_matches_record(row, record):
    _, category, question, answer, keywords, intent_id = row
    return (
        category == record.category
        and question == record.question
        and answer == record.answer
        and _normalize_keywords(keywords) == record.keywords_json
        and intent_id == record.intent_id
    )


def import_faq_records(json_path, mysql_client=None, redis_client=None):
    """Insert or update FAQ records by stable intent_id without duplicate rows."""
    records = load_faq_records(json_path)
    mysql_client = mysql_client or MysqlClient()
    mysql_client.ensure_faq_schema()
    result = FaqImportResult(json_rows=len(records))
    table = config.MYSQL_FAQ_TABLE

    try:
        for record in records:
            mysql_client.cursor.execute(
                f"SELECT id, category, question, answer, keywords, intent_id "
                f"FROM {table} WHERE intent_id = %s OR question = %s FOR UPDATE",
                (record.intent_id, record.question),
            )
            matches = mysql_client.cursor.fetchall()
            if len(matches) > 1:
                raise ValueError(
                    "conflicting FAQ rows for intent_id {} and question {}; resolve manually"
                    .format(record.intent_id, record.question)
                )

            if not matches:
                mysql_client.cursor.execute(
                    f"INSERT INTO {table} "
                    "(category, question, answer, keywords, intent_id) VALUES (%s, %s, %s, %s, %s)",
                    (
                        record.category,
                        record.question,
                        record.answer,
                        record.keywords_json,
                        record.intent_id,
                    ),
                )
                result.inserted += 1
            elif _row_matches_record(matches[0], record):
                result.skipped += 1
            else:
                mysql_client.cursor.execute(
                    f"UPDATE {table} SET category = %s, question = %s, answer = %s, "
                    "keywords = %s, intent_id = %s WHERE id = %s",
                    (
                        record.category,
                        record.question,
                        record.answer,
                        record.keywords_json,
                        record.intent_id,
                        matches[0][0],
                    ),
                )
                result.updated += 1
        mysql_client.connect.commit()
    except Exception:
        mysql_client.connect.rollback()
        raise

    if result.inserted or result.updated:
        redis_client = redis_client or RedisClient()
        answer_keys = [
            f"{config.REDIS_KEY_PREFIX}answer:{record.question}" for record in records
        ]
        result.cache_keys_deleted = redis_client.delete_keys(*FAQ_INDEX_KEYS, *answer_keys)

    logger.info("financial FAQ import finished: {}".format(result.to_dict()))
    return result


def main():
    parser = argparse.ArgumentParser(description="Import financial FAQ JSON into configured MySQL.")
    parser.add_argument(
        "json_path",
        nargs="?",
        default=str(Path(config.FINANCIAL_DATA_DIR) / "faq" / "financial_faq.json"),
    )
    args = parser.parse_args()
    print(json.dumps(import_faq_records(args.json_path).to_dict(), ensure_ascii=False))


if __name__ == "__main__":
    main()
