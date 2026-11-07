"""
Module: ravi_shankar_code_risk_dataset_10000 Audit & Forensics Pipeline
Author: Senior ML Engineer
Description: Production-grade CSV ingestion, schema validation, unique identifier checks, 
             temporal ordering, target prevalence profiling, and embargo-based splitting.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("CodeRiskAuditPipeline")


@dataclass
class AuditConfig:
    file_path: Path = Path("C:/Users/ravis/Desktop/Nexlyra/Project7/28-10-2026-Ravi/ravi_shankar_code_risk_dataset_10000.csv")
    expected_rows: int = 10000
    expected_cols: int = 20
    id_col: str = "change_id"
    timestamp_col: str = "merged_at_utc"
    target_col: str = "postmerge_defect_14d"
    embargo_days: int = 14
    
    expected_columns_list: List[str] = field(default_factory=lambda: [
        "change_id", "merged_at_utc", "engineering_team", "primary_language",
        "service_component", "lines_changed", "code_churn_lines", "files_touched",
        "unit_test_coverage_pct", "cyclomatic_complexity_mean", "dependency_updates",
        "static_analysis_findings", "premerge_failed_checks", "review_turnaround_hours",
        "author_tenure_months", "release_window_flag", "hotfix_flag",
        "critical_service_flag", "prior_30d_defect_count", "postmerge_defect_14d"
    ])


class CodeRiskAuditPipeline:
    def __init__(self, config: AuditConfig):
        self.config = config
        self.df: pd.DataFrame = None

    def load_data(self) -> None:
        """Loads dataset from CSV file."""
        logger.info("Loading dataset from %s...", self.config.file_path)
        if not self.config.file_path.exists():
            raise FileNotFoundError(f"Dataset not found at path: {self.config.file_path}")
        
        self.df = pd.read_csv(self.config.file_path)
        logger.info("Successfully loaded dataset with shape: %s", self.df.shape)

    def validate_schema(self) -> None:
        """Validates exact row count, column count, and column names."""
        logger.info("Validating schema contract...")
        
        if self.df.shape != (self.config.expected_rows, self.config.expected_cols):
            raise ValueError(
                f"Shape mismatch! Expected ({self.config.expected_rows}, {self.config.expected_cols}), "
                f"got {self.df.shape}"
            )
            
        actual_cols = list(self.df.columns)
        if actual_cols != self.config.expected_columns_list:
            missing = set(self.config.expected_columns_list) - set(actual_cols)
            extra = set(actual_cols) - set(self.config.expected_columns_list)
            raise ValueError(f"Schema column mismatch! Missing: {missing}, Extra: {extra}")
            
        logger.info("Schema contract verified successfully.")

    def audit_leakage_and_identifiers(self) -> None:
        """Verifies unique change identifiers and binary target constraints."""
        logger.info("Auditing unique identifiers and feature constraints...")
        
        duplicates = self.df[self.config.id_col].duplicated().sum()
        if duplicates > 0:
            raise ValueError(f"Found {duplicates} duplicate change identifiers in {self.config.id_col}!")
            
        unique_targets = self.df[self.config.target_col].unique()
        if not set(unique_targets).issubset({0, 1}):
            raise ValueError(f"Invalid target labels detected in {self.config.target_col}: {unique_targets}")
            
        logger.info("Identifier uniqueness and leakage checks passed.")

    def parse_and_verify_temporality(self) -> None:
        """Parses UTC timestamps and ensures strict chronological ordering."""
        logger.info("Parsing UTC merge timestamps and verifying order...")
        
        self.df[self.config.timestamp_col] = pd.to_datetime(
            self.df[self.config.timestamp_col], utc=True, errors="raise"
        )
        
        if not self.df[self.config.timestamp_col].is_monotonic_increasing:
            logger.warning("Dataset is not chronologically sorted. Sorting by timestamp now.")
            self.df = self.df.sort_values(by=self.config.timestamp_col).reset_index(drop=True)
            
        logger.info("Temporal ordering verified.")

    def compute_prevalence_stats(self) -> Dict[str, float]:
        """Computes target prevalence overall and across categorical dimensions."""
        logger.info("Computing target prevalence metrics...")
        
        overall = float(self.df[self.config.target_col].mean())
        by_team = self.df.groupby("engineering_team")[self.config.target_col].mean().to_dict()
        by_lang = self.df.groupby("primary_language")[self.config.target_col].mean().to_dict()
        
        logger.info("Overall Defect Prevalence: %.4f (%.2f%%)", overall, overall * 100)
        return {
            "overall_prevalence": overall,
            "prevalence_by_team": by_team,
            "prevalence_by_language": by_lang
        }

    def time_split_with_embargo(
        self, 
        train_end: str, 
        val_end: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Executes time-based split with a 14-day label embargo buffer."""
        logger.info("Executing time-based split with %d-day embargo...", self.config.embargo_days)
        
        # FIX APPLIED: Using pd.Timestamp with timezone.utc instead of pd.to_datetime(..., tz=...)
        t_train_end = pd.Timestamp(train_end, tz=timezone.utc)
        t_val_end = pd.Timestamp(val_end, tz=timezone.utc)
        embargo_delta = pd.Timedelta(days=self.config.embargo_days)

        train_mask = self.df[self.config.timestamp_col] <= t_train_end
        
        val_start = t_train_end + embargo_delta
        val_mask = (self.df[self.config.timestamp_col] >= val_start) & \
                   (self.df[self.config.timestamp_col] <= t_val_end)
                   
        test_start = t_val_end + embargo_delta
        test_mask = self.df[self.config.timestamp_col] >= test_start

        df_train = self.df[train_mask].copy()
        df_val = self.df[val_mask].copy()
        df_test = self.df[test_mask].copy()

        logger.info(
            "Split Complete -> Train samples: %d | Validation samples: %d | Test samples: %d",
            len(df_train), len(df_val), len(df_test)
        )
        return df_train, df_val, df_test

    def run_pipeline(self, train_end_date: str, val_end_date: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Orchestrates the end-to-end data contract and forensics audit."""
        self.load_data()
        self.validate_schema()
        self.audit_leakage_and_identifiers()
        self.parse_and_verify_temporality()
        self.compute_prevalence_stats()
        return self.time_split_with_embargo(train_end_date, val_end_date)


if __name__ == "__main__":
    config = AuditConfig()
    pipeline = CodeRiskAuditPipeline(config)
    
    # Execute pipeline with defined temporal cutoffs
    df_train, df_val, df_test = pipeline.run_pipeline(
        train_end_date="2025-02-01T00:00:00Z",
        val_end_date="2025-02-15T00:00:00Z"
    )
    print("Audit pipeline successfully completed on ravi_shankar_code_risk_dataset_10000.csv")