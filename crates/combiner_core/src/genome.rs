//! Genome types for the Generative Combiner.
//!
//! A `StrategyGenome` represents a complete trading strategy as a sequence of
//! `BlockGene`s, where each gene encodes a block type, block ID, and parameters.

use rustc_hash::FxHasher;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use uuid::Uuid;

/// Block type in the strategy pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BlockType {
    Selection,
    Entry,
    Exit,
    Sizing,
}

impl BlockType {
    /// Convert to string for TOML serialization.
    pub fn as_str(&self) -> &'static str {
        match self {
            BlockType::Selection => "selection",
            BlockType::Entry => "entry",
            BlockType::Exit => "exit",
            BlockType::Sizing => "sizing",
        }
    }

    /// Order priority for sorting genes (Selection first, Sizing last).
    pub fn order(&self) -> u8 {
        match self {
            BlockType::Selection => 0,
            BlockType::Entry => 1,
            BlockType::Exit => 2,
            BlockType::Sizing => 3,
        }
    }
}

impl std::fmt::Display for BlockType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Parameter value with range for mutation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ParamValue {
    /// Floating-point parameter with range and step.
    Float {
        value: f64,
        min: f64,
        max: f64,
        step: f64,
    },
    /// Integer parameter with range and step.
    Int {
        value: i64,
        min: i64,
        max: i64,
        step: i64,
    },
    /// Boolean parameter.
    Bool { value: bool },
}

impl ParamValue {
    /// Create a new float parameter.
    pub fn float(value: f64, min: f64, max: f64, step: f64) -> Self {
        Self::Float {
            value: value.clamp(min, max),
            min,
            max,
            step,
        }
    }

    /// Create a new integer parameter.
    pub fn int(value: i64, min: i64, max: i64, step: i64) -> Self {
        Self::Int {
            value: value.clamp(min, max),
            min,
            max,
            step,
        }
    }

    /// Create a new boolean parameter.
    pub fn bool(value: bool) -> Self {
        Self::Bool { value }
    }

    /// Get the current value as f64 (for fitness calculations).
    pub fn as_f64(&self) -> f64 {
        match self {
            ParamValue::Float { value, .. } => *value,
            ParamValue::Int { value, .. } => *value as f64,
            ParamValue::Bool { value } => {
                if *value {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }

    /// Get the current value as i64.
    pub fn as_i64(&self) -> i64 {
        match self {
            ParamValue::Float { value, .. } => *value as i64,
            ParamValue::Int { value, .. } => *value,
            ParamValue::Bool { value } => {
                if *value {
                    1
                } else {
                    0
                }
            }
        }
    }

    /// Get the current value as bool.
    pub fn as_bool(&self) -> bool {
        match self {
            ParamValue::Float { value, .. } => *value != 0.0,
            ParamValue::Int { value, .. } => *value != 0,
            ParamValue::Bool { value } => *value,
        }
    }

    /// Convert to TOML value.
    pub fn to_toml_value(&self) -> toml::Value {
        match self {
            ParamValue::Float { value, .. } => toml::Value::Float(*value),
            ParamValue::Int { value, .. } => toml::Value::Integer(*value),
            ParamValue::Bool { value } => toml::Value::Boolean(*value),
        }
    }

    /// Check if value is within valid range.
    pub fn is_valid(&self) -> bool {
        match self {
            ParamValue::Float {
                value, min, max, ..
            } => *value >= *min && *value <= *max,
            ParamValue::Int {
                value, min, max, ..
            } => *value >= *min && *value <= *max,
            ParamValue::Bool { .. } => true,
        }
    }

    /// Clamp value to valid range.
    pub fn clamp(&mut self) {
        match self {
            ParamValue::Float {
                value, min, max, ..
            } => {
                *value = value.clamp(*min, *max);
            }
            ParamValue::Int {
                value, min, max, ..
            } => {
                *value = (*value).clamp(*min, *max);
            }
            ParamValue::Bool { .. } => {}
        }
    }
}

/// Individual gene representing a block and its parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockGene {
    /// Type of block (Selection, Entry, Exit, Sizing).
    pub block_type: BlockType,
    /// Block identifier (e.g., "momentum", "rsi", "equal_weight").
    pub block_id: String,
    /// Parameters for this block.
    pub params: HashMap<String, ParamValue>,
}

impl BlockGene {
    /// Create a new gene with the given block type, id, and parameters.
    pub fn new(
        block_type: BlockType,
        block_id: impl Into<String>,
        params: impl IntoIterator<Item = (impl Into<String>, ParamValue)>,
    ) -> Self {
        Self {
            block_type,
            block_id: block_id.into(),
            params: params.into_iter().map(|(k, v)| (k.into(), v)).collect(),
        }
    }

    /// Create a gene with default parameters (empty).
    pub fn with_defaults(block_type: BlockType, block_id: impl Into<String>) -> Self {
        Self {
            block_type,
            block_id: block_id.into(),
            params: HashMap::new(),
        }
    }

    /// Get a parameter value by name.
    pub fn get_param(&self, name: &str) -> Option<&ParamValue> {
        self.params.get(name)
    }

    /// Set a parameter value.
    pub fn set_param(&mut self, name: impl Into<String>, value: ParamValue) {
        self.params.insert(name.into(), value);
    }

    /// Convert parameters to BlockParams format.
    pub fn to_block_params(&self) -> HashMap<String, toml::Value> {
        self.params
            .iter()
            .map(|(k, v)| (k.clone(), v.to_toml_value()))
            .collect()
    }

    /// Compute a deterministic hash for this gene.
    /// Floats are rounded to 4 decimal places to avoid floating-point noise.
    pub fn compute_hash(&self) -> u64 {
        let mut hasher = FxHasher::default();
        self.block_type.as_str().hash(&mut hasher);
        self.block_id.hash(&mut hasher);

        // Sort params for deterministic hashing
        let mut params: Vec<_> = self.params.iter().collect();
        params.sort_by_key(|(k, _)| *k);

        for (key, value) in params {
            key.hash(&mut hasher);
            // Hash the value based on type - round floats to avoid FP noise
            match value {
                ParamValue::Float { value, .. } => {
                    // Round to 4 decimal places before hashing
                    let rounded = (*value * 10000.0).round() as i64;
                    rounded.hash(&mut hasher);
                }
                ParamValue::Int { value, .. } => {
                    value.hash(&mut hasher);
                }
                ParamValue::Bool { value } => {
                    value.hash(&mut hasher);
                }
            }
        }

        hasher.finish()
    }
    
    /// Round all float parameters to avoid floating-point noise.
    /// Returns true if any parameter was modified.
    #[inline]
    pub fn sanitize_params(&mut self) -> bool {
        let mut modified = false;
        for (_, param) in &mut self.params {
            if let ParamValue::Float { value, min, max, step } = param {
                // Round to step precision, minimum 4 decimal places
                let precision = step.log10().abs().ceil() as i32 + 1;
                let factor = 10_f64.powi(precision.min(4));
                let rounded = (*value * factor).round() / factor;
                let clamped = rounded.clamp(*min, *max);
                if (*value - clamped).abs() > 1e-10 {
                    *value = clamped;
                    modified = true;
                }
            }
        }
        modified
    }
}

/// Complete genome representing a trading strategy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyGenome {
    /// Unique identifier for this genome.
    pub id: Uuid,
    /// Sequence of genes (blocks) in the strategy pipeline.
    pub genes: Vec<BlockGene>,
    /// Fitness scores (populated after evaluation).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fitness: Option<crate::fitness::MultiObjectiveFitness>,
    /// Generation number when this genome was created.
    pub generation: u32,
    /// Parent genome IDs (for lineage tracking).
    #[serde(default)]
    pub parent_ids: Vec<Uuid>,
    /// Cached hash for deduplication.
    #[serde(skip)]
    cached_hash: Option<u64>,
}

impl StrategyGenome {
    /// Create a new genome with the given genes.
    pub fn new(genes: Vec<BlockGene>) -> Self {
        Self {
            id: Uuid::new_v4(),
            genes,
            fitness: None,
            generation: 0,
            parent_ids: Vec::new(),
            cached_hash: None,
        }
    }

    /// Create a new genome with a specific ID.
    pub fn with_id(id: Uuid, genes: Vec<BlockGene>) -> Self {
        Self {
            id,
            genes,
            fitness: None,
            generation: 0,
            parent_ids: Vec::new(),
            cached_hash: None,
        }
    }

    /// Set the generation number.
    pub fn with_generation(mut self, generation: u32) -> Self {
        self.generation = generation;
        self
    }

    /// Set parent IDs for lineage tracking.
    pub fn with_parents(mut self, parent_ids: Vec<Uuid>) -> Self {
        self.parent_ids = parent_ids;
        self
    }

    /// Get genes by block type.
    pub fn genes_by_type(&self, block_type: BlockType) -> Vec<&BlockGene> {
        self.genes
            .iter()
            .filter(|g| g.block_type == block_type)
            .collect()
    }

    /// Check if genome has at least one gene of the given type.
    pub fn has_block_type(&self, block_type: BlockType) -> bool {
        self.genes.iter().any(|g| g.block_type == block_type)
    }

    /// Count genes of a specific type.
    pub fn count_block_type(&self, block_type: BlockType) -> usize {
        self.genes
            .iter()
            .filter(|g| g.block_type == block_type)
            .count()
    }

    /// Sort genes by block type order (Selection -> Entry -> Exit -> Sizing).
    pub fn sort_genes(&mut self) {
        self.genes.sort_by_key(|g| g.block_type.order());
        self.cached_hash = None; // Invalidate cache
    }
    
    /// Remove duplicate consecutive blocks of the same type and id.
    /// Keeps max 2 blocks of each (block_type, block_id) combination.
    /// Returns number of blocks removed.
    pub fn deduplicate_blocks(&mut self) -> usize {
        use std::collections::HashMap;
        let mut counts: HashMap<(BlockType, String), usize> = HashMap::new();
        let original_len = self.genes.len();
        
        self.genes.retain(|gene| {
            let key = (gene.block_type, gene.block_id.clone());
            let count = counts.entry(key).or_insert(0);
            *count += 1;
            // Keep max 2 of each (block_type, block_id) combo
            *count <= 2
        });
        
        let removed = original_len - self.genes.len();
        if removed > 0 {
            self.cached_hash = None;
        }
        removed
    }
    
    /// Sanitize all gene parameters (round floats).
    pub fn sanitize(&mut self) {
        for gene in &mut self.genes {
            gene.sanitize_params();
        }
        self.cached_hash = None;
    }

    /// Compute a stable hash for deduplication and caching.
    pub fn compute_hash(&mut self) -> u64 {
        if let Some(hash) = self.cached_hash {
            return hash;
        }

        let mut hasher = FxHasher::default();

        // Sort genes by type and id for deterministic hashing
        let mut sorted_genes = self.genes.clone();
        sorted_genes.sort_by(|a, b| {
            (a.block_type.order(), &a.block_id).cmp(&(b.block_type.order(), &b.block_id))
        });

        for gene in &sorted_genes {
            gene.compute_hash().hash(&mut hasher);
        }

        let hash = hasher.finish();
        self.cached_hash = Some(hash);
        hash
    }

    /// Get cached hash or compute if not available.
    pub fn hash(&self) -> u64 {
        if let Some(hash) = self.cached_hash {
            hash
        } else {
            // Compute without caching (for immutable access)
            let mut hasher = FxHasher::default();
            let mut sorted_genes = self.genes.clone();
            sorted_genes.sort_by(|a, b| {
                (a.block_type.order(), &a.block_id).cmp(&(b.block_type.order(), &b.block_id))
            });
            for gene in &sorted_genes {
                gene.compute_hash().hash(&mut hasher);
            }
            hasher.finish()
        }
    }

    /// Add a gene to the genome.
    pub fn add_gene(&mut self, gene: BlockGene) {
        self.genes.push(gene);
        self.cached_hash = None;
    }

    /// Remove a gene at the given index.
    pub fn remove_gene(&mut self, index: usize) -> Option<BlockGene> {
        if index < self.genes.len() {
            self.cached_hash = None;
            Some(self.genes.remove(index))
        } else {
            None
        }
    }

    /// Replace a gene at the given index.
    pub fn replace_gene(&mut self, index: usize, gene: BlockGene) -> Option<BlockGene> {
        if index < self.genes.len() {
            self.cached_hash = None;
            Some(std::mem::replace(&mut self.genes[index], gene))
        } else {
            None
        }
    }

    /// Create a deep clone with a new ID.
    pub fn clone_with_new_id(&self) -> Self {
        Self {
            id: Uuid::new_v4(),
            genes: self.genes.clone(),
            fitness: None, // Reset fitness for new individual
            generation: self.generation,
            parent_ids: vec![self.id],
            cached_hash: None,
        }
    }
    
    /// Generate a human-readable strategy name from the genome.
    /// Format: <Selection> • <Entry> • <Exit> • <Sizing>
    /// Example: "Momentum(126d) • MACross(20/50) • ATRTrail • VolTarget"
    pub fn generate_name(&self) -> String {
        let mut parts = Vec::new();
        
        // Format block_id to title case with key param
        let format_block = |gene: &BlockGene| -> String {
            let name = gene.block_id.chars()
                .take(1)
                .map(|c| c.to_uppercase().next().unwrap_or(c))
                .chain(gene.block_id.chars().skip(1))
                .collect::<String>()
                .replace('_', "");
            
            // Add key parameter if present
            if let Some(param) = gene.params.get("lookback_days").or(gene.params.get("fast_period")).or(gene.params.get("period")) {
                format!("{}({})", name, param.as_i64())
            } else if let Some(fast) = gene.params.get("fast_period") {
                if let Some(slow) = gene.params.get("slow_period") {
                    format!("{}({}/{})", name, fast.as_i64(), slow.as_i64())
                } else {
                    name
                }
            } else {
                name
            }
        };
        
        // Get one gene per block type
        for block_type in [BlockType::Selection, BlockType::Entry, BlockType::Exit, BlockType::Sizing] {
            if let Some(gene) = self.genes.iter().find(|g| g.block_type == block_type) {
                parts.push(format_block(gene));
            }
        }
        
        // Join with bullet separator, limit to 48 chars
        let name = parts.join(" • ");
        if name.len() > 48 {
            format!("{}…", &name[..47])
        } else {
            name
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_param_value_float() {
        let param = ParamValue::float(0.15, 0.0, 1.0, 0.05);
        assert_eq!(param.as_f64(), 0.15);
        assert!(param.is_valid());
    }

    #[test]
    fn test_param_value_int() {
        let param = ParamValue::int(126, 21, 252, 21);
        assert_eq!(param.as_i64(), 126);
        assert!(param.is_valid());
    }

    #[test]
    fn test_param_value_clamp() {
        // Create with direct struct to bypass constructor clamping
        let mut param = ParamValue::Float {
            value: 1.5,
            min: 0.0,
            max: 1.0,
            step: 0.1,
        };
        assert!(!param.is_valid());
        param.clamp();
        assert!(param.is_valid());
        assert_eq!(param.as_f64(), 1.0);
    }

    #[test]
    fn test_block_gene_creation() {
        let gene = BlockGene::new(
            BlockType::Selection,
            "momentum",
            vec![
                ("lookback_days", ParamValue::int(126, 21, 252, 21)),
                ("top_pct", ParamValue::float(0.2, 0.05, 0.5, 0.05)),
            ],
        );

        assert_eq!(gene.block_type, BlockType::Selection);
        assert_eq!(gene.block_id, "momentum");
        assert_eq!(gene.params.len(), 2);
    }

    #[test]
    fn test_genome_hash_determinism() {
        let gene1 = BlockGene::new(
            BlockType::Selection,
            "momentum",
            vec![("lookback_days", ParamValue::int(126, 21, 252, 21))],
        );
        let gene2 = BlockGene::new(
            BlockType::Sizing,
            "equal_weight",
            vec![("max_weight", ParamValue::float(0.2, 0.05, 1.0, 0.05))],
        );

        let mut genome1 = StrategyGenome::new(vec![gene1.clone(), gene2.clone()]);
        let mut genome2 = StrategyGenome::new(vec![gene2, gene1]);

        // Order shouldn't matter for hash
        assert_eq!(genome1.compute_hash(), genome2.compute_hash());
    }

    #[test]
    fn test_genome_genes_by_type() {
        let genes = vec![
            BlockGene::with_defaults(BlockType::Selection, "momentum"),
            BlockGene::with_defaults(BlockType::Selection, "quality"),
            BlockGene::with_defaults(BlockType::Sizing, "equal_weight"),
        ];

        let genome = StrategyGenome::new(genes);

        assert_eq!(genome.genes_by_type(BlockType::Selection).len(), 2);
        assert_eq!(genome.genes_by_type(BlockType::Sizing).len(), 1);
        assert_eq!(genome.genes_by_type(BlockType::Entry).len(), 0);
    }
}

