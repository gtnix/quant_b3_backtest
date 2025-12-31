//! Dividend Engine Module
//!
//! Provides dividend loading, indexing, and application for backtesting.
//!
//! # Anti-Double-Count Policy
//!
//! This module implements the critical anti-double-count policy:
//! - **Signals/Indicators**: Use adjusted prices (dividend-adjusted)  
//! - **Mark-to-Market/Equity**: Use raw prices + dividend cashflow
//!
//! Dividends are credited on the **ex_date** (not payment_date) to:
//! 1. Match behavior of adjusted price series
//! 2. Simplify accounting (no "dividend receivable" tracking)
//!
//! # Example Usage
//!
//! ```rust,ignore
//! use backtester_intelligence::dividends::{DividendLoader, DividendIndex};
//!
//! // Load dividends for simulation period
//! let loader = DividendLoader::new();
//! let dividends = loader.load_range("TAEE11", start_date, end_date)?;
//!
//! // Build efficient index
//! let index = DividendIndex::from_entries(dividends);
//!
//! // Apply on ex_date
//! for div in index.get_by_date(ex_date) {
//!     let cashflow = div.rate * shares_held;
//!     portfolio.add_cash(cashflow);
//! }
//! ```

pub mod loader;
pub mod types;

pub use loader::{DividendLoader, DividendLoadError};
pub use types::{DividendEntry, DividendIndex, DividendApplication, PriceType};












