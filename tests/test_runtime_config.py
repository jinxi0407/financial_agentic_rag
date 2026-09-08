import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from base.config import Config
from evaluations.financial_eval import _evaluation_report


class RetrievalConfigurationTests(unittest.TestCase):
    def _config_file(self):
        temporary_file = tempfile.NamedTemporaryFile("w", suffix=".ini", delete=False)
        temporary_file.write("[retrieval]\nretrieval_k = 7\ncandidate_m = 4\n")
        temporary_file.close()
        self.addCleanup(Path(temporary_file.name).unlink)
        return temporary_file.name

    def test_config_ini_values_are_used_without_environment_overrides(self):
        with patch("base.config.load_dotenv"), patch.dict(
            os.environ, {"RETRIEVAL_K": "", "CANDIDATE_M": ""}, clear=False
        ):
            os.environ.pop("RETRIEVAL_K")
            os.environ.pop("CANDIDATE_M")
            configured = Config(self._config_file())

        self.assertEqual(7, configured.RETRIEVAL_K)
        self.assertEqual(4, configured.CANDIDATE_M)

    def test_environment_values_override_config_ini(self):
        with patch.dict(os.environ, {"RETRIEVAL_K": "30", "CANDIDATE_M": "3"}, clear=False):
            configured = Config(self._config_file())

        self.assertEqual(30, configured.RETRIEVAL_K)
        self.assertEqual(3, configured.CANDIDATE_M)

    def test_runtime_snapshot_contains_only_reproducibility_fields(self):
        with patch("base.config.load_dotenv"), patch.dict(
            os.environ, {"RETRIEVAL_K": "", "CANDIDATE_M": ""}, clear=False
        ):
            os.environ.pop("RETRIEVAL_K")
            os.environ.pop("CANDIDATE_M")
            configured = Config(self._config_file())
        snapshot = configured.runtime_config_snapshot()

        self.assertEqual(
            {"retrieval_k", "candidate_m", "git_commit", "milvus_database", "milvus_collection"},
            set(snapshot),
        )
        self.assertEqual(7, snapshot["retrieval_k"])
        self.assertEqual(4, snapshot["candidate_m"])

    def test_evaluation_report_persists_runtime_snapshot(self):
        runtime_config = {
            "retrieval_k": 30,
            "candidate_m": 3,
            "git_commit": "abc1234",
            "milvus_database": "financial",
            "milvus_collection": "financial_rag_v1",
        }
        report = _evaluation_report(Path("gold.json"), runtime_config, [])

        self.assertEqual(runtime_config, report["runtime_config"])
        self.assertEqual(0, report["metrics"]["completed"])


if __name__ == "__main__":
    unittest.main()
