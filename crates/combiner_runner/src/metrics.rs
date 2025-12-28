//! Metrics parsing utilities.

use crate::executor::BacktestMetrics;
use serde_json::Value;
use std::path::Path;
use thiserror::Error;

/// Errors during metrics parsing.
#[derive(Debug, Error)]
pub enum MetricsError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON parse error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Missing field: {0}")]
    MissingField(String),

    #[error("Invalid value for field {0}: {1}")]
    InvalidValue(String, String),
}

/// Parser for metrics.json files.
pub struct MetricsParser;

impl MetricsParser {
    /// Parse metrics from a JSON file.
    pub fn parse_file(path: &Path) -> Result<BacktestMetrics, MetricsError> {
        let content = std::fs::read_to_string(path)?;
        Self::parse_str(&content)
    }

    /// Parse metrics from a JSON string.
    pub fn parse_str(json: &str) -> Result<BacktestMetrics, MetricsError> {
        let value: Value = serde_json::from_str(json)?;
        Self::parse_value(&value)
    }

    /// Parse metrics from a serde_json Value.
    pub fn parse_value(value: &Value) -> Result<BacktestMetrics, MetricsError> {
        Ok(BacktestMetrics {
            cagr: Self::get_f64(value, "cagr")?,
            volatility: Self::get_f64_opt(value, "volatility"),
            sharpe_ratio: Self::get_f64(value, "sharpe_ratio")?,
            sortino_ratio: Self::get_f64_opt(value, "sortino_ratio"),
            calmar_ratio: Self::get_f64_opt(value, "calmar_ratio"),
            max_drawdown: Self::get_f64(value, "max_drawdown")?,
            max_drawdown_duration_days: Self::get_u32_opt(value, "max_drawdown_duration_days"),
            hit_rate: Self::get_f64_opt(value, "hit_rate"),
            profit_factor: Self::get_f64_opt(value, "profit_factor"),
            turnover_annual: Self::get_f64_opt(value, "turnover_annual"),
            total_trades: Self::get_u32(value, "total_trades").unwrap_or(0),
            winning_trades: Self::get_u32_opt(value, "winning_trades"),
            losing_trades: Self::get_u32_opt(value, "losing_trades"),
        })
    }

    fn get_f64(value: &Value, field: &str) -> Result<f64, MetricsError> {
        value
            .get(field)
            .and_then(|v| v.as_f64())
            .ok_or_else(|| MetricsError::MissingField(field.into()))
    }

    fn get_f64_opt(value: &Value, field: &str) -> Option<f64> {
        value.get(field).and_then(|v| v.as_f64())
    }

    fn get_u32(value: &Value, field: &str) -> Result<u32, MetricsError> {
        value
            .get(field)
            .and_then(|v| v.as_u64())
            .map(|v| v as u32)
            .ok_or_else(|| MetricsError::MissingField(field.into()))
    }

    fn get_u32_opt(value: &Value, field: &str) -> Option<u32> {
        value.get(field).and_then(|v| v.as_u64()).map(|v| v as u32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_metrics() {
        let json = r#"{
            "cagr": 0.15,
            "volatility": 0.20,
            "sharpe_ratio": 0.75,
            "sortino_ratio": 1.0,
            "calmar_ratio": 1.5,
            "max_drawdown": -0.10,
            "max_drawdown_duration_days": 30,
            "hit_rate": 0.55,
            "profit_factor": 1.5,
            "turnover_annual": 2.5,
            "total_trades": 120,
            "winning_trades": 66,
            "losing_trades": 54
        }"#;

        let metrics = MetricsParser::parse_str(json).unwrap();

        assert!((metrics.cagr - 0.15).abs() < 0.01);
        assert!((metrics.sharpe_ratio - 0.75).abs() < 0.01);
        assert!((metrics.max_drawdown - (-0.10)).abs() < 0.01);
        assert_eq!(metrics.total_trades, 120);
    }

    #[test]
    fn test_parse_minimal() {
        let json = r#"{
            "cagr": 0.10,
            "sharpe_ratio": 0.5,
            "max_drawdown": -0.15
        }"#;

        let metrics = MetricsParser::parse_str(json).unwrap();

        assert!((metrics.cagr - 0.10).abs() < 0.01);
        assert!(metrics.volatility.is_none());
        assert!(metrics.calmar_ratio.is_none());
    }
}

