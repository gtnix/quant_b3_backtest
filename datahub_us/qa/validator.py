"""Data validation for OHLCV data quality."""

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import List, Optional
from enum import Enum
import pandas as pd

logger = logging.getLogger(__name__)


class IssueType(str, Enum):
    """Types of validation issues."""
    SCHEMA_ERROR = "schema_error"
    MISSING_COLUMN = "missing_column"
    INVALID_DATE = "invalid_date"
    DUPLICATE_DATE = "duplicate_date"
    NON_MONOTONIC = "non_monotonic"
    OHLC_VIOLATION = "ohlc_violation"
    NEGATIVE_VOLUME = "negative_volume"
    ZERO_VOLUME = "zero_volume"
    PRICE_OUTLIER = "price_outlier"
    GAP_DETECTED = "gap_detected"
    NULL_VALUE = "null_value"


@dataclass
class ValidationIssue:
    """A single validation issue."""
    issue_type: IssueType
    severity: str  # "error", "warning", "info"
    message: str
    row_index: Optional[int] = None
    date: Optional[str] = None
    column: Optional[str] = None
    value: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of data validation."""
    symbol: str
    is_valid: bool
    total_rows: int
    issues: List[ValidationIssue] = field(default_factory=list)
    validated_at: datetime = field(default_factory=datetime.utcnow)
    gaps: List[dict] = field(default_factory=list)
    
    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "error")
    
    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "warning")
    
    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "is_valid": self.is_valid,
            "total_rows": self.total_rows,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "gaps": self.gaps,
            "issues": [
                {
                    "type": i.issue_type.value,
                    "severity": i.severity,
                    "message": i.message,
                    "date": i.date,
                }
                for i in self.issues[:20]  # Limit to first 20
            ]
        }


class DataValidator:
    """Validates OHLCV data quality."""
    
    REQUIRED_COLUMNS = ["date", "open", "high", "low", "close", "volume"]
    
    def __init__(
        self,
        max_price_change_pct: float = 50.0,
        max_gap_days: int = 5,
        check_volume: bool = True,
    ):
        self.max_price_change_pct = max_price_change_pct
        self.max_gap_days = max_gap_days
        self.check_volume = check_volume
    
    def validate(self, symbol: str, df: pd.DataFrame) -> ValidationResult:
        """Validate OHLCV DataFrame.
        
        Args:
            symbol: Ticker symbol
            df: DataFrame with OHLCV data
            
        Returns:
            ValidationResult with issues found
        """
        issues = []
        gaps = []
        
        if df.empty:
            return ValidationResult(
                symbol=symbol,
                is_valid=False,
                total_rows=0,
                issues=[ValidationIssue(
                    issue_type=IssueType.SCHEMA_ERROR,
                    severity="error",
                    message="DataFrame is empty"
                )]
            )
        
        # Check schema
        issues.extend(self._check_schema(df))
        
        if any(i.severity == "error" for i in issues):
            return ValidationResult(
                symbol=symbol,
                is_valid=False,
                total_rows=len(df),
                issues=issues
            )
        
        # Check dates
        issues.extend(self._check_dates(df))
        
        # Check OHLC sanity
        issues.extend(self._check_ohlc_sanity(df))
        
        # Check volume
        if self.check_volume:
            issues.extend(self._check_volume(df))
        
        # Check for null values
        issues.extend(self._check_nulls(df))
        
        # Check for price outliers
        issues.extend(self._check_outliers(df))
        
        # Detect gaps
        gaps = self._detect_gaps(df)
        if gaps:
            for gap in gaps:
                issues.append(ValidationIssue(
                    issue_type=IssueType.GAP_DETECTED,
                    severity="warning",
                    message=f"Gap of {gap['days']} days from {gap['start']} to {gap['end']}",
                    date=gap['start']
                ))
        
        is_valid = not any(i.severity == "error" for i in issues)
        
        return ValidationResult(
            symbol=symbol,
            is_valid=is_valid,
            total_rows=len(df),
            issues=issues,
            gaps=gaps
        )
    
    def _check_schema(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Check DataFrame has required columns."""
        issues = []
        
        for col in self.REQUIRED_COLUMNS:
            if col not in df.columns:
                issues.append(ValidationIssue(
                    issue_type=IssueType.MISSING_COLUMN,
                    severity="error",
                    message=f"Missing required column: {col}",
                    column=col
                ))
        
        return issues
    
    def _check_dates(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Check date column validity."""
        issues = []
        
        if 'date' not in df.columns:
            return issues
        
        # Check for duplicates
        duplicates = df[df['date'].duplicated()]
        if not duplicates.empty:
            for idx, row in duplicates.iterrows():
                issues.append(ValidationIssue(
                    issue_type=IssueType.DUPLICATE_DATE,
                    severity="error",
                    message=f"Duplicate date: {row['date']}",
                    row_index=idx,
                    date=str(row['date'])
                ))
        
        # Check monotonicity
        dates = pd.to_datetime(df['date'])
        if not dates.is_monotonic_increasing:
            issues.append(ValidationIssue(
                issue_type=IssueType.NON_MONOTONIC,
                severity="warning",
                message="Dates are not monotonically increasing"
            ))
        
        return issues
    
    def _check_ohlc_sanity(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Check OHLC relationships: low <= open/close <= high."""
        issues = []
        
        for idx, row in df.iterrows():
            o, h, l, c = row.get('open'), row.get('high'), row.get('low'), row.get('close')
            
            if pd.isna(o) or pd.isna(h) or pd.isna(l) or pd.isna(c):
                continue
            
            violations = []
            if l > o:
                violations.append(f"low ({l}) > open ({o})")
            if l > c:
                violations.append(f"low ({l}) > close ({c})")
            if h < o:
                violations.append(f"high ({h}) < open ({o})")
            if h < c:
                violations.append(f"high ({h}) < close ({c})")
            if l > h:
                violations.append(f"low ({l}) > high ({h})")
            
            if violations:
                issues.append(ValidationIssue(
                    issue_type=IssueType.OHLC_VIOLATION,
                    severity="error",
                    message=f"OHLC violation: {', '.join(violations)}",
                    row_index=idx,
                    date=str(row.get('date', ''))
                ))
        
        return issues
    
    def _check_volume(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Check volume validity."""
        issues = []
        
        if 'volume' not in df.columns:
            return issues
        
        # Negative volume
        negative = df[df['volume'] < 0]
        for idx, row in negative.iterrows():
            issues.append(ValidationIssue(
                issue_type=IssueType.NEGATIVE_VOLUME,
                severity="error",
                message=f"Negative volume: {row['volume']}",
                row_index=idx,
                date=str(row.get('date', '')),
                value=str(row['volume'])
            ))
        
        # Zero volume (warning only)
        zero_volume_count = (df['volume'] == 0).sum()
        if zero_volume_count > 0:
            issues.append(ValidationIssue(
                issue_type=IssueType.ZERO_VOLUME,
                severity="info",
                message=f"{zero_volume_count} rows with zero volume"
            ))
        
        return issues
    
    def _check_nulls(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Check for null values in required columns."""
        issues = []
        
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns:
                null_count = df[col].isna().sum()
                if null_count > 0:
                    issues.append(ValidationIssue(
                        issue_type=IssueType.NULL_VALUE,
                        severity="warning",
                        message=f"{null_count} null values in {col}",
                        column=col
                    ))
        
        return issues
    
    def _check_outliers(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Check for extreme price changes."""
        issues = []
        
        if 'close' not in df.columns or len(df) < 2:
            return issues
        
        df_sorted = df.sort_values('date').reset_index(drop=True)
        pct_change = df_sorted['close'].pct_change() * 100
        
        for idx, change in enumerate(pct_change):
            if pd.notna(change) and abs(change) > self.max_price_change_pct:
                issues.append(ValidationIssue(
                    issue_type=IssueType.PRICE_OUTLIER,
                    severity="warning",
                    message=f"Extreme price change: {change:.1f}%",
                    row_index=idx,
                    date=str(df_sorted.iloc[idx].get('date', '')),
                    value=f"{change:.1f}%"
                ))
        
        return issues
    
    def _detect_gaps(self, df: pd.DataFrame) -> List[dict]:
        """Detect gaps in trading dates."""
        gaps = []
        
        if 'date' not in df.columns or len(df) < 2:
            return gaps
        
        df_sorted = df.sort_values('date').reset_index(drop=True)
        dates = pd.to_datetime(df_sorted['date'])
        
        for i in range(1, len(dates)):
            prev_date = dates.iloc[i-1]
            curr_date = dates.iloc[i]
            
            # Calculate business days gap (approximate)
            delta = (curr_date - prev_date).days
            
            # Allow for weekends (2 days) but flag larger gaps
            if delta > self.max_gap_days:
                gaps.append({
                    "start": prev_date.strftime("%Y-%m-%d"),
                    "end": curr_date.strftime("%Y-%m-%d"),
                    "days": delta
                })
        
        return gaps





























