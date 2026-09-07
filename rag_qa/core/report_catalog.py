"""Deterministic catalog for the locally available listed-company reports."""

from dataclasses import dataclass
from pathlib import Path

from rag_qa.core.document_processor import parse_annual_report_filename


PERIOD_TYPE_LABELS = {
    "H1": "半年度",
    "FY": "年度",
}
PERIOD_TYPE_ORDER = {
    "H1": 0,
    "FY": 1,
}


@dataclass(frozen=True)
class ReportRecord:
    company_name: str
    company_code: str
    report_year: int
    period_type: str
    report_period: str
    source_filename: str

    @property
    def display_name(self):
        return f"{self.company_name}{self.report_year}年{PERIOD_TYPE_LABELS[self.period_type]}报告"


class ReportCatalog:
    """A filename-derived catalog; it never opens report content or calls an LLM."""

    def __init__(self, reports=()):
        self._reports = tuple(sorted(
            reports,
            key=lambda report: (
                report.company_code,
                report.report_year,
                PERIOD_TYPE_ORDER[report.period_type],
                report.source_filename,
            ),
        ))

    @classmethod
    def from_directory(cls, directory_path):
        directory = Path(directory_path)
        if not directory.exists():
            return cls()
        if not directory.is_dir():
            raise ValueError(f"annual_reports path is not a directory: {directory}")

        reports = []
        for report_path in sorted(directory.glob("*.pdf")):
            metadata = parse_annual_report_filename(report_path.name)
            reports.append(ReportRecord(
                source_filename=report_path.name,
                **metadata,
            ))
        return cls(reports)

    def find_reports(self, company_code, report_year=None, period_type=None):
        return tuple(
            report for report in self._reports
            if report.company_code == company_code
            and (report_year is None or report.report_year == report_year)
            and (period_type is None or report.period_type == period_type)
        )

    @property
    def reports(self):
        return self._reports
