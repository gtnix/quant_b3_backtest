//! Integrity engine for checksums and cryptographic hashing

use anyhow::Result;
use blake3::Hasher as Blake3Hasher;
use xxhash_rust::xxh3::xxh3_64;

use crate::types::IntegritySeal;

/// Integrity level for artifact validation.
///
/// Different levels trade off speed vs security:
/// - `None`: Zero overhead (internal hot path only)
/// - `Fast`: XXH3 only (~10 GB/s) - good for intermediate artifacts
/// - `Secure`: BLAKE3 only (~3 GB/s) - cryptographic security
/// - `Full`: XXH3 + BLAKE3 - cold storage / final artifacts
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum IntegrityLevel {
    /// No integrity checking (internal hot path only)
    None,
    /// XXH3 only - fast validation (~10 GB/s)
    #[default]
    Fast,
    /// BLAKE3 only - cryptographic security (~3 GB/s)
    Secure,
    /// Both XXH3 and BLAKE3 - maximum integrity
    Full,
}

/// Result of integrity computation based on level
#[derive(Debug, Clone)]
pub struct IntegrityResult {
    pub xxh3: Option<u64>,
    pub blake3: Option<[u8; 32]>,
    pub level: IntegrityLevel,
}

impl IntegrityResult {
    pub fn none() -> Self {
        Self { xxh3: None, blake3: None, level: IntegrityLevel::None }
    }
}

/// IntegrityEngine handles all hashing and integrity verification
#[derive(Debug, Clone, Default)]
pub struct IntegrityEngine {
    pub level: IntegrityLevel,
}

impl IntegrityEngine {
    /// Create a new IntegrityEngine with default level (Fast)
    pub fn new() -> Self {
        Self { level: IntegrityLevel::default() }
    }
    
    /// Create with specific integrity level
    pub fn with_level(level: IntegrityLevel) -> Self {
        Self { level }
    }

    /// Compute XXH3 checksum for fast validation (59.4 GB/s throughput)
    #[inline]
    pub fn compute_xxh3(&self, data: &[u8]) -> u64 {
        xxh3_64(data)
    }

    /// Compute BLAKE3 hash for cryptographic integrity (4.4 GB/s throughput)
    #[inline]
    pub fn compute_blake3(&self, data: &[u8]) -> [u8; 32] {
        let mut hasher = Blake3Hasher::new();
        hasher.update(data);
        *hasher.finalize().as_bytes()
    }
    
    /// Compute integrity based on configured level (adaptive overhead)
    #[inline]
    pub fn compute(&self, data: &[u8]) -> IntegrityResult {
        match self.level {
            IntegrityLevel::None => IntegrityResult::none(),
            IntegrityLevel::Fast => IntegrityResult {
                xxh3: Some(self.compute_xxh3(data)),
                blake3: None,
                level: IntegrityLevel::Fast,
            },
            IntegrityLevel::Secure => IntegrityResult {
                xxh3: None,
                blake3: Some(self.compute_blake3(data)),
                level: IntegrityLevel::Secure,
            },
            IntegrityLevel::Full => IntegrityResult {
                xxh3: Some(self.compute_xxh3(data)),
                blake3: Some(self.compute_blake3(data)),
                level: IntegrityLevel::Full,
            },
        }
    }

    /// Create an integrity seal for an artifact (always uses Full level for seals)
    pub fn create_seal(&self, data: &[u8]) -> IntegritySeal {
        let block_xxh3 = self.compute_xxh3(data);
        let artifact_blake3 = self.compute_blake3(data);
        let sealed_at = chrono::Utc::now().timestamp();

        IntegritySeal {
            block_xxh3,
            artifact_blake3,
            sealed_at,
        }
    }
    
    /// Create seal based on current level (adaptive)
    pub fn create_seal_adaptive(&self, data: &[u8]) -> IntegritySeal {
        let sealed_at = chrono::Utc::now().timestamp();
        
        match self.level {
            IntegrityLevel::None => IntegritySeal {
                block_xxh3: 0,
                artifact_blake3: [0u8; 32],
                sealed_at,
            },
            IntegrityLevel::Fast => IntegritySeal {
                block_xxh3: self.compute_xxh3(data),
                artifact_blake3: [0u8; 32],
                sealed_at,
            },
            IntegrityLevel::Secure => IntegritySeal {
                block_xxh3: 0,
                artifact_blake3: self.compute_blake3(data),
                sealed_at,
            },
            IntegrityLevel::Full => IntegritySeal {
                block_xxh3: self.compute_xxh3(data),
                artifact_blake3: self.compute_blake3(data),
                sealed_at,
            },
        }
    }

    /// Validate data against an integrity seal (full validation)
    pub fn validate(&self, data: &[u8], seal: &IntegritySeal) -> Result<()> {
        let computed_xxh3 = self.compute_xxh3(data);
        if computed_xxh3 != seal.block_xxh3 {
            anyhow::bail!(
                "XXH3 checksum mismatch: expected {:x}, got {:x}",
                seal.block_xxh3,
                computed_xxh3
            );
        }

        let computed_blake3 = self.compute_blake3(data);
        if computed_blake3 != seal.artifact_blake3 {
            anyhow::bail!(
                "BLAKE3 hash mismatch: expected {:?}, got {:?}",
                seal.artifact_blake3,
                computed_blake3
            );
        }

        Ok(())
    }
    
    /// Validate based on current level (adaptive)
    pub fn validate_adaptive(&self, data: &[u8], seal: &IntegritySeal) -> Result<()> {
        match self.level {
            IntegrityLevel::None => Ok(()),
            IntegrityLevel::Fast => self.validate_fast(data, seal.block_xxh3),
            IntegrityLevel::Secure => {
                let computed = self.compute_blake3(data);
                if computed != seal.artifact_blake3 {
                    anyhow::bail!("BLAKE3 hash mismatch");
                }
                Ok(())
            },
            IntegrityLevel::Full => self.validate(data, seal),
        }
    }

    /// Fast validation using only XXH3
    pub fn validate_fast(&self, data: &[u8], expected_xxh3: u64) -> Result<()> {
        let computed_xxh3 = self.compute_xxh3(data);
        if computed_xxh3 != expected_xxh3 {
            anyhow::bail!(
                "XXH3 checksum mismatch: expected {:x}, got {:x}",
                expected_xxh3,
                computed_xxh3
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xxh3_consistency() {
        let engine = IntegrityEngine::new();
        let data = b"Hello, OBFS!";

        let hash1 = engine.compute_xxh3(data);
        let hash2 = engine.compute_xxh3(data);

        assert_eq!(hash1, hash2, "XXH3 should be deterministic");
    }

    #[test]
    fn test_blake3_consistency() {
        let engine = IntegrityEngine::new();
        let data = b"Hello, OBFS!";

        let hash1 = engine.compute_blake3(data);
        let hash2 = engine.compute_blake3(data);

        assert_eq!(hash1, hash2, "BLAKE3 should be deterministic");
    }

    #[test]
    fn test_seal_validation() {
        let engine = IntegrityEngine::new();
        let data = b"Test data for seal validation";

        let seal = engine.create_seal(data);

        assert!(engine.validate(data, &seal).is_ok());

        let corrupted_data = b"Test data for seal validatioX";
        assert!(engine.validate(corrupted_data, &seal).is_err());
    }

    #[test]
    fn test_fast_validation() {
        let engine = IntegrityEngine::new();
        let data = b"Fast validation test";

        let xxh3 = engine.compute_xxh3(data);

        assert!(engine.validate_fast(data, xxh3).is_ok());
        assert!(engine.validate_fast(data, xxh3 + 1).is_err());
    }
}



