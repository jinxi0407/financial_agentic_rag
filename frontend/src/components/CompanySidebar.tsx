import { useState } from "react";
import { SUPPORTED_COMPANIES, type SupportedCompany } from "../data/companies";

interface CompanySidebarProps {
  selectedCompany: string | null;
  onSelect: (company: SupportedCompany) => void;
}

export function CompanySidebar({ selectedCompany, onSelect }: CompanySidebarProps) {
  const [isOpen, setIsOpen] = useState(false);
  const selectCompany = (company: SupportedCompany) => {
    onSelect(company);
    setIsOpen(false);
  };

  return (
    <aside className={`company-sidebar ${isOpen ? "is-open" : ""}`} aria-label="当前支持公司">
      <button
        className="company-mobile-toggle"
        type="button"
        onClick={() => setIsOpen((open) => !open)}
        aria-expanded={isOpen}
        aria-controls="supported-company-list"
      >
        <span>当前支持公司</span>
        <small>8 家 A 股公司</small>
        <span aria-hidden="true">{isOpen ? "−" : "+"}</span>
      </button>
      <div className="company-sidebar-content" id="supported-company-list">
        <div className="company-sidebar-heading">
          <h2>当前支持</h2>
          <p>8 家 A 股公司</p>
        </div>
        <div className="company-list">
          {SUPPORTED_COMPANIES.map((company) => {
            const selected = selectedCompany === company.name;
            return (
              <button
                key={company.code}
                type="button"
                className={`company-item ${selected ? "is-selected" : ""}`}
                aria-label={`将${company.name}插入提问框，股票代码${company.code}`}
                aria-pressed={selected}
                onClick={() => selectCompany(company)}
              >
                <span className="company-mark" aria-hidden="true">{company.mark}</span>
                <span className="company-copy">
                  <strong>{company.name}</strong>
                  <small>{company.code}</small>
                </span>
              </button>
            );
          })}
        </div>
        <p className="report-period-note">当前财报语料：<br />2025H1 · 2025FY · 2026H1</p>
      </div>
    </aside>
  );
}
