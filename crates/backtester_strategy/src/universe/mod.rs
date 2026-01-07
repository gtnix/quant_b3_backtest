//! Parameter Universe System
//!
//! Controls and limits the generation of trading strategies through 4 axes:
//! - Robustness Profile
//! - Training Strategy
//! - Training Tech
//! - Training Model (Strategy Family)
//!
//! This module provides:
//! - Type definitions for all universe components
//! - Loaders for configuration files
//! - Validators for compatibility between axes

mod types;
mod loader;
mod validator;

pub use types::*;
pub use loader::*;
pub use validator::*;




