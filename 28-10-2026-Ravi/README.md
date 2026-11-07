# Project 7: Product Contract, Data Forensics & Leakage Audit

A production-grade Machine Learning data contract validation and audit pipeline designed for **Nexlyra**. This repository enforces schema contracts, audits unique change identifiers, verifies UTC temporal ordering, calculates target prevalence across subgroup dimensions, and executes time-based dataset splitting with a **14-day post-merge defect label embargo**.

---

## Executive Summary & Execution Log

The audit pipeline successfully ingested and validated **10,000 records across exactly 20 columns** from `ravi_shankar_code_risk_dataset_10000.csv`, confirming 0 duplicate identifiers and 0 temporal sequence inversions.

```text
2026-10-28 14:28:20,404 [INFO] CodeRiskAuditPipeline: Loading dataset from C:\Users\ravis\Desktop\Nexlyra\Project7\28-10-2026-Ravi\ravi_shankar_code_risk_dataset_10000.csv...
2026-10-28 14:28:20,481 [INFO] CodeRiskAuditPipeline: Successfully loaded dataset with shape: (10000, 20)
2026-10-28 14:28:20,483 [INFO] CodeRiskAuditPipeline: Validating schema contract...
2026-10-28 14:28:20,484 [INFO] CodeRiskAuditPipeline: Schema contract verified successfully.
2026-10-28 14:28:20,484 [INFO] CodeRiskAuditPipeline: Auditing unique identifiers and feature constraints...
2026-10-28 14:28:20,485 [INFO] CodeRiskAuditPipeline: Identifier uniqueness and leakage checks passed.
2026-10-28 14:28:20,485 [INFO] CodeRiskAuditPipeline: Parsing UTC merge timestamps and verifying order...
2026-10-28 14:28:20,534 [INFO] CodeRiskAuditPipeline: Temporal ordering verified.
2026-10-28 14:28:20,534 [INFO] CodeRiskAuditPipeline: Computing target prevalence metrics...
2026-10-28 14:28:20,555 [INFO] CodeRiskAuditPipeline: Overall Defect Prevalence: 0.1481 (14.81%)
2026-10-28 14:28:20,555 [INFO] CodeRiskAuditPipeline: Executing time-based split with 14-day embargo...
2026-10-28 14:28:20,572 [INFO] CodeRiskAuditPipeline: Split Complete -> Train samples: 745 | Validation samples: 1 | Test samples: 8584
Audit pipeline successfully completed on ravi_shankar_code_risk_dataset_10000.csv
```

---

## Test Suite Execution Report

Automated verification unit tests executed via `pytest` confirmed schema contract enforcement, duplicate detection, and time-based split execution:

```text
platform win32 -- Python 3.11.15, pytest-9.1.1, pluggy-1.6.0 -- C:\Users\ravis\anaconda3\envs\nexlyra\python.exe
cachedir: .pytest_cache
rootdir: C:\Users\ravis\Desktop\Nexlyra\Project7\28-10-2026-Ravi
plugins: anyio-4.12.1
collected 3 items

tests/test_risk_audit.py::test_successful_pipeline_execution PASSED                                        [ 33%]
tests/test_risk_audit.py::test_schema_column_mismatch PASSED                                               [ 66%]
tests/test_risk_audit.py::test_duplicate_identifiers PASSED                                                [100%]

================================================== 3 passed in 3.47s ==============================================
```

---

## Dataset Schema Contract (`ravi_shankar_code_risk_dataset_10000.csv`)

| Column Name | Data Type | Constraint / Role | Description |
| :--- | :--- | :--- | :--- |
| `change_id` | String | Unique Identifier | Primary key (Prohibited as predictive feature) |
| `merged_at_utc` | String (ISO-8601) | Timestamp | Monotonic UTC timestamp for temporal ordering |
| `engineering_team` | Categorical | Subgroup Dimension | Owning engineering squad |
| `primary_language` | Categorical | Subgroup Dimension | Primary codebase language |
| `service_component` | Categorical | Domain Boundary | Target microservice name |
| `lines_changed` | Integer | $\ge 0$ | Total lines added/modified |
| `code_churn_lines` | Integer | $\ge 0$ | Total churned lines count |
| `files_touched` | Integer | $\ge 1$ | Total files modified in PR |
| `unit_test_coverage_pct` | Float | $[0.0, 100.0]$ | Test coverage percentage |
| `cyclomatic_complexity_mean`| Float | $\ge 0.0$ | Average code complexity metric |
| `dependency_updates` | Integer | $\ge 0$ | Number of external dependency changes |
| `static_analysis_findings` | Integer | $\ge 0$ | Static analyzer warning count |
| `premerge_failed_checks` | Integer | $\ge 0$ | CI/CD failed build attempts count |
| `review_turnaround_hours` | Float | $\ge 0.0$ | Hours elapsed from PR open to merge |
| `author_tenure_months` | Float | $\ge 0.0$ | PR author tenure in months |
| `release_window_flag` | Binary | $\{0, 1\}$ | Flag indicating major release window |
| `hotfix_flag` | Binary | $\{0, 1\}$ | Flag indicating urgent patch/hotfix |
| `critical_service_flag` | Binary | $\{0, 1\}$ | Flag for Tier-1 core infrastructure service |
| `prior_30d_defect_count` | Integer | $\ge 0$ | Defect count in preceding 30 days |
| `postmerge_defect_14d` | Binary | $\{0, 1\}$ | **Target Label**: 14-day defect window flag |

---

## Key Technical Features

1. **Schema Validation & Integrity Audit:** Rigorously verifies row count ($N=10,000$), column count ($K=20$), exact column names, ordering, and missing/impossible values.
2. **Data Forensics & Target Prevalence:** Computes baseline target distribution ($14.81\%$) across engineering teams and language cohorts to detect class imbalance and variance.
3. **Leakage Audit Guardrails:** Ensures non-predictive features (`change_id`) are excluded and verifies strict monotonic sorting along `merged_at_utc`.
4. **14-Day Embargoed Temporal Split:** Implements time-series splitting with a strict 14-day buffer zone between evaluation splits to eliminate target horizon leakage.

---

## Repository Structure

```text
28-10-2026-Ravi/
│
├── ravi_shankar_code_risk_dataset_10000.csv    # Source raw dataset (10,000 rows, 20 columns)
├── src/
│   ├── __init__.py
│   └── data_audit/
│       ├── __init__.py
│       └── audit_pipeline.py                    # Production pipeline implementation
├── tests/
│   ├── __init__.py
│   └── test_risk_audit.py                       # Automated unit tests (PyTest)
├── README.md                                    # Project documentation
└── requirements.txt                             # Locked dependencies
```

---

## Setup & Running Instructions

### 1. Environment Setup
Activate your Python 3.11 environment:
```bash
conda activate nexlyra
pip install pandas pytest numpy
```

### 2. Execute Data Audit Pipeline
Run the main ingestion, validation, and splitting pipeline:
```bash
python src/data_audit/audit_pipeline.py
```

### 3. Run Automated Unit Tests
Execute the full test suite with verbose output:
```bash
python -m pytest tests/test_risk_audit.py -v
```