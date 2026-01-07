//! Symbol Registry - Maps String symbols to dense SymbolId for O(1) indexing.
//!
//! This module eliminates HashMap<String, _> from the hot path by providing
//! a pre-registration phase (setup) and O(1) lookups during simulation.

use std::collections::HashMap;
use std::fmt;

/// Symbol identifier for O(1) array indexing.
///
/// Uses u32 to support up to 4 billion symbols while keeping the type small.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
pub struct SymbolId(u32);

impl SymbolId {
    /// Create a new SymbolId.
    #[must_use]
    pub const fn new(id: u32) -> Self {
        SymbolId(id)
    }

    /// Get as usize for array indexing.
    #[must_use]
    pub const fn as_usize(self) -> usize {
        self.0 as usize
    }

    /// Get raw value.
    #[must_use]
    pub const fn raw(self) -> u32 {
        self.0
    }
}

impl fmt::Debug for SymbolId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SymbolId({})", self.0)
    }
}

impl fmt::Display for SymbolId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<u32> for SymbolId {
    fn from(id: u32) -> Self {
        SymbolId(id)
    }
}

impl From<usize> for SymbolId {
    fn from(id: usize) -> Self {
        SymbolId(id as u32)
    }
}

/// Registry for String <-> SymbolId mapping.
///
/// # Design
///
/// - `symbol_to_id`: HashMap used only during setup phase
/// - `id_to_symbol`: Vec for O(1) reverse lookup (rarely used in hot path)
///
/// # Usage
///
/// ```ignore
/// let mut registry = SymbolRegistry::new();
/// let id = registry.register("PETR4");
/// assert_eq!(registry.resolve(id), "PETR4");
/// ```
#[derive(Debug, Clone, Default)]
pub struct SymbolRegistry {
    symbol_to_id: HashMap<String, SymbolId>,
    id_to_symbol: Vec<String>,
}

impl SymbolRegistry {
    /// Create a new empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a registry with pre-allocated capacity.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            symbol_to_id: HashMap::with_capacity(capacity),
            id_to_symbol: Vec::with_capacity(capacity),
        }
    }

    /// Register a symbol and return its ID.
    ///
    /// If already registered, returns existing ID.
    /// This is O(1) amortized.
    pub fn register(&mut self, symbol: &str) -> SymbolId {
        if let Some(&id) = self.symbol_to_id.get(symbol) {
            return id;
        }

        let id = SymbolId::new(self.id_to_symbol.len() as u32);
        self.id_to_symbol.push(symbol.to_string());
        self.symbol_to_id.insert(symbol.to_string(), id);
        id
    }

    /// Get ID for a symbol (returns None if not registered).
    #[must_use]
    pub fn get(&self, symbol: &str) -> Option<SymbolId> {
        self.symbol_to_id.get(symbol).copied()
    }

    /// Get or register a symbol.
    ///
    /// Convenience method that combines get + register.
    pub fn get_or_register(&mut self, symbol: &str) -> SymbolId {
        self.register(symbol)
    }

    /// Resolve SymbolId back to String.
    ///
    /// # Panics
    ///
    /// Panics if ID is out of bounds (should never happen with valid IDs).
    #[must_use]
    pub fn resolve(&self, id: SymbolId) -> &str {
        &self.id_to_symbol[id.as_usize()]
    }

    /// Try to resolve SymbolId, returning None if invalid.
    #[must_use]
    pub fn try_resolve(&self, id: SymbolId) -> Option<&str> {
        self.id_to_symbol.get(id.as_usize()).map(String::as_str)
    }

    /// Number of registered symbols.
    #[must_use]
    pub fn len(&self) -> usize {
        self.id_to_symbol.len()
    }

    /// Check if registry is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.id_to_symbol.is_empty()
    }

    /// Iterate over all (SymbolId, symbol) pairs in registration order.
    pub fn iter(&self) -> impl Iterator<Item = (SymbolId, &str)> {
        self.id_to_symbol
            .iter()
            .enumerate()
            .map(|(i, s)| (SymbolId::new(i as u32), s.as_str()))
    }

    /// Get all symbols in registration order.
    #[must_use]
    pub fn symbols(&self) -> &[String] {
        &self.id_to_symbol
    }

    /// Register multiple symbols at once, returning their IDs.
    pub fn register_all<'a>(&mut self, symbols: impl IntoIterator<Item = &'a str>) -> Vec<SymbolId> {
        symbols.into_iter().map(|s| self.register(s)).collect()
    }

    /// Check if a symbol is registered.
    #[must_use]
    pub fn contains(&self, symbol: &str) -> bool {
        self.symbol_to_id.contains_key(symbol)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_and_resolve() {
        let mut registry = SymbolRegistry::new();
        
        let id1 = registry.register("PETR4");
        let id2 = registry.register("VALE3");
        let id3 = registry.register("PETR4"); // duplicate
        
        assert_eq!(id1, id3); // same symbol = same ID
        assert_ne!(id1, id2);
        assert_eq!(registry.resolve(id1), "PETR4");
        assert_eq!(registry.resolve(id2), "VALE3");
    }

    #[test]
    fn test_sequential_ids() {
        let mut registry = SymbolRegistry::new();
        
        let id1 = registry.register("A");
        let id2 = registry.register("B");
        let id3 = registry.register("C");
        
        assert_eq!(id1.raw(), 0);
        assert_eq!(id2.raw(), 1);
        assert_eq!(id3.raw(), 2);
    }

    #[test]
    fn test_get_unregistered() {
        let registry = SymbolRegistry::new();
        assert!(registry.get("MISSING").is_none());
    }

    #[test]
    fn test_iter_order() {
        let mut registry = SymbolRegistry::new();
        registry.register("B");
        registry.register("A");
        registry.register("C");
        
        let symbols: Vec<&str> = registry.iter().map(|(_, s)| s).collect();
        assert_eq!(symbols, vec!["B", "A", "C"]); // registration order
    }

    #[test]
    fn test_register_all() {
        let mut registry = SymbolRegistry::new();
        let ids = registry.register_all(["PETR4", "VALE3", "ITUB4"]);
        
        assert_eq!(ids.len(), 3);
        assert_eq!(registry.len(), 3);
    }

    #[test]
    fn test_symbol_id_indexing() {
        let id = SymbolId::new(42);
        let vec: Vec<i32> = (0..100).collect();
        assert_eq!(vec[id.as_usize()], 42);
    }
}



