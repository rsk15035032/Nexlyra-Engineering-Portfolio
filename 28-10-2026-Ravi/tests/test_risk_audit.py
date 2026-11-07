"""
Module: Automated Unit Tests for Code Risk Audit Pipeline
Author: Senior ML Engineer
Description: Tests schema contracts, duplicate identifiers, temporal ordering, 
             and embargo splits against mock test fixtures.
"""

from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.data_audit.audit_pipeline import AuditConfig, CodeRiskAuditPipeline


@pytest.fixture
def mock_csv_dataset(tmp_path) -> Path:
    """Creates a temporary valid CSV matching ravi_shankar_code_risk_dataset_10000.csv structure."""
    np.random.seed(42)
    n = 10000
    timestamps = pd.date_range(start="2025-01-01", periods=n, freq="H", tz=timezone.utc)
    
    df = pd.DataFrame({
        "change_id": [f"NX-PR-{i:06d}" for i in range(n)],
        "merged_at_utc": timestamps.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "engineering_team": np.random.choice(["frontend", "backend", "platform"], size=n),
        "primary_language": np.random.choice(["java", "python", "rust", "go"], size=n),
        "service_component": np.random.choice(["api", "auth", "search", "db"], size=n),
        "lines_changed": np.random.randint(1, 500, size=n),
        "code_churn_lines": np.random.randint(1, 400, size=n),
        "files_touched": np.random.randint(1, 15, size=n),
        "unit_test_coverage_pct": np.random.uniform(20.0, 100.0, size=n),
        "cyclomatic_complexity_mean": np.random.uniform(1.0, 15.0, size=n),
        "dependency_updates": np.random.randint(0, 5, size=n),
        "static_analysis_findings": np.random.randint(0, 10, size=n),
        "premerge_failed_checks": np.random.randint(0, 3, size=n),
        "review_turnaround_hours": np.random.uniform(0.5, 48.0, size=n),
        "author_tenure_months": np.random.uniform(0.1, 60.0, size=n),
        "release_window_flag": np.random.choice([0, 1], size=n),
        "hotfix_flag": np.random.choice([0, 1], size=n, p=[0.95, 0.05]),
        "critical_service_flag": np.random.choice([0, 1], size=n),
        "prior_30d_defect_count": np.random.randint(0, 4, size=n),
        "postmerge_defect_14d": np.random.choice([0, 1], size=n, p=[0.85, 0.15]),
    })
    
    file_path = tmp_path / "ravi_shankar_code_risk_dataset_10000.csv"
    df.to_csv(file_path, index=False)
    return file_path


def test_successful_pipeline_execution(mock_csv_dataset):
    """Test full end-to-end execution with a valid dataset."""
    config = AuditConfig(file_path=mock_csv_dataset)
    pipeline = CodeRiskAuditPipeline(config)
    
    df_train, df_val, df_test = pipeline.run_pipeline(
        train_end_date="2025-02-01T00:00:00Z",
        val_end_date="2025-02-15T00:00:00Z"
    )
    
    assert len(df_train) > 0
    assert len(df_val) > 0
    assert len(df_test) > 0
    assert (len(df_train) + len(df_val) + len(df_test)) <= 10000


def test_schema_column_mismatch(mock_csv_dataset):
    """Test that missing columns throw a ValueError schema mismatch."""
    df = pd.read_csv(mock_csv_dataset)
    df = df.drop(columns=["hotfix_flag"]) # Drop one column to invalidate schema
    
    invalid_path = mock_csv_dataset.parent / "invalid_schema.csv"
    df.to_csv(invalid_path, index=False)
    
    config = AuditConfig(file_path=invalid_path, expected_cols=19)
    pipeline = CodeRiskAuditPipeline(config)
    pipeline.load_data()
    
    with pytest.raises(ValueError, match="Schema column mismatch"):
        pipeline.validate_schema()


def test_duplicate_identifiers(mock_csv_dataset):
    """Test that duplicate change IDs raise a ValueError."""
    df = pd.read_csv(mock_csv_dataset)
    df.loc[1, "change_id"] = df.loc[0, "change_id"] # Create duplicate ID
    
    dup_path = mock_csv_dataset.parent / "dup_ids.csv"
    df.to_csv(dup_path, index=False)
    
    config = AuditConfig(file_path=dup_path)
    pipeline = CodeRiskAuditPipeline(config)
    pipeline.load_data()
    pipeline.validate_schema()
    
    with pytest.raises(ValueError, match="duplicate change identifiers"):
        pipeline.audit_leakage_and_identifiers()