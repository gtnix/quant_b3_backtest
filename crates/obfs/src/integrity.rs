//! Integrity engine for checksums and cryptographic hashing

use anyhow::Result;
use blake3::Hasher as Blake3Hasher;
use xxhash_rust::xxh3::xxh3_64;

use crate::types::IntegritySeal;

/// IntegrityEngine handles all hashing and integrity verification
#[derive(Debug, Clone, Default)]
pub struct IntegrityEngine;

impl IntegrityEngine {
    /// Create a new IntegrityEngine
    pub fn new() -> Self {
        Self
    }

    /// Compute XXH3 checksum for fast validation (59.4 GB/s throughput)
    pub fn compute_xxh3(&self, data: &[u8]) -> u64 {
        xxh3_64(data)
    }

    /// Compute BLAKE3 hash for cryptographic integrity (4.4 GB/s throughput)
    pub fn compute_blake3(&self, data: &[u8]) -> [u8; 32] {
        let mut hasher = Blake3Hasher::new();
        hasher.update(data);
        *hasher.finalize().as_bytes()
    }

    /// Create an integrity seal for an artifact
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

    /// Validate data against an integrity seal
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

