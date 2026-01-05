//! Report Bundle - Consolidated OBFS storage for SCG candidates and reports.
//!
//! Consolidates hundreds of individual JSON/TOML files into a single ultra-compressed
//! binary bundle with O(1) lookup via LMDB indexing.
//!
//! Storage format:
//! - candidates.obfs: All config TOMLs + validation JSONs in single file
//! - index.lmdb/: Fast lookup by genome_id/uuid
//!
//! Compression: Zstd level 19 (max) with long-distance matching

use anyhow::Result;
use heed::types::*;
use heed::{Database, Env, EnvOpenOptions};
use rkyv::{Archive, Deserialize, Serialize};
use serde::{Deserialize as SerdeDeserialize, Serialize as SerdeSerialize};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::PathBuf;
use uuid::Uuid;
use zstd::stream::{Decoder, Encoder};

/// Ultra compression level (Zstd max)
pub const ULTRA_COMPRESSION_LEVEL: i32 = 19;

/// Candidate entry stored in the bundle
#[derive(Archive, Serialize, Deserialize, Debug, Clone)]
#[rkyv(derive(Debug))]
pub struct CandidateEntry {
    /// UUID as bytes
    pub uuid_bytes: [u8; 16],
    /// Genome hash for deduplication
    pub genome_hash: u64,
    /// Rank in Hall of Fame
    pub rank: u32,
    /// Validated generation
    pub validated_generation: u32,
    /// Production score
    pub production_score: f64,
    /// OOS Sharpe
    pub oos_sharpe: f64,
    /// OOS CAGR  
    pub oos_cagr: f64,
    /// Max drawdown
    pub max_drawdown: f64,
    /// PBO
    pub pbo: f64,
    /// DSR
    pub dsr: f64,
    /// Degradation %
    pub degradation_pct: f64,
    /// Splits passed (e.g., 5)
    pub splits_passed: u16,
    /// Splits evaluated (e.g., 6)
    pub splits_evaluated: u16,
}

impl CandidateEntry {
    pub fn uuid(&self) -> Uuid {
        Uuid::from_bytes(self.uuid_bytes)
    }
}

/// Location of data within the bundle file
#[derive(Debug, Clone, SerdeSerialize, SerdeDeserialize)]
pub struct BundleLocation {
    /// Offset in the data file
    pub offset: u64,
    /// Compressed size
    pub compressed_size: u32,
    /// Original size (for allocation)
    pub original_size: u32,
}

/// Index entry for fast lookups
#[derive(Debug, Clone, SerdeSerialize, SerdeDeserialize)]
pub struct CandidateIndex {
    /// Location of config TOML
    pub config_loc: BundleLocation,
    /// Location of validation JSON
    pub validation_loc: BundleLocation,
    /// Location of candidate entry
    pub entry_loc: BundleLocation,
}

/// Report bundle for consolidated storage
pub struct ReportBundle {
    root_path: PathBuf,
    env: Env,
    /// uuid -> CandidateIndex
    index_db: Database<Bytes, Bytes>,
    /// genome_hash -> uuid (for hash-based lookups)
    hash_db: Database<Bytes, Bytes>,
    /// Data file handle
    data_file: Option<File>,
}

impl ReportBundle {
    /// Open or create a report bundle
    pub fn open(root_path: impl Into<PathBuf>) -> Result<Self> {
        let root_path = root_path.into();
        fs::create_dir_all(&root_path)?;

        let lmdb_path = root_path.join("index.lmdb");
        fs::create_dir_all(&lmdb_path)?;

        let env = unsafe {
            EnvOpenOptions::new()
                .map_size(1024 * 1024 * 1024) // 1 GB max
                .max_dbs(2)
                .open(&lmdb_path)?
        };

        let mut wtxn = env.write_txn()?;
        let index_db = env.create_database(&mut wtxn, Some("index"))?;
        let hash_db = env.create_database(&mut wtxn, Some("hash"))?;
        wtxn.commit()?;

        Ok(Self {
            root_path,
            env,
            index_db,
            hash_db,
            data_file: None,
        })
    }

    /// Get the data file path
    fn data_file_path(&self) -> PathBuf {
        self.root_path.join("candidates.obfs")
    }

    /// Ensure data file is open for writing
    fn ensure_data_file(&mut self) -> Result<&mut File> {
        if self.data_file.is_none() {
            let file = OpenOptions::new()
                .create(true)
                .read(true)
                .append(true)
                .open(self.data_file_path())?;
            self.data_file = Some(file);
        }
        Ok(self.data_file.as_mut().unwrap())
    }

    /// Ultra-compress data using Zstd level 19 with LDM
    fn ultra_compress(data: &[u8]) -> Result<Vec<u8>> {
        let mut encoder = Encoder::new(Vec::new(), ULTRA_COMPRESSION_LEVEL)?;
        encoder.include_checksum(true)?;
        encoder.long_distance_matching(true)?;
        encoder.write_all(data)?;
        Ok(encoder.finish()?)
    }

    /// Decompress data
    fn decompress(compressed: &[u8]) -> Result<Vec<u8>> {
        let mut decoder = Decoder::new(compressed)?;
        let mut decompressed = Vec::new();
        decoder.read_to_end(&mut decompressed)?;
        Ok(decompressed)
    }

    /// Write data to bundle and return location
    fn write_to_bundle(&mut self, data: &[u8]) -> Result<BundleLocation> {
        let compressed = Self::ultra_compress(data)?;
        let file = self.ensure_data_file()?;
        let offset = file.seek(SeekFrom::End(0))?;
        
        // Write length prefix (4 bytes) + compressed data
        let len_bytes = (compressed.len() as u32).to_le_bytes();
        file.write_all(&len_bytes)?;
        file.write_all(&compressed)?;

        Ok(BundleLocation {
            offset,
            compressed_size: compressed.len() as u32,
            original_size: data.len() as u32,
        })
    }

    /// Add a candidate to the bundle
    pub fn add_candidate(
        &mut self,
        entry: &CandidateEntry,
        config_toml: &str,
        validation_json: &str,
    ) -> Result<()> {
        // Serialize entry using rkyv
        let entry_bytes = rkyv::to_bytes::<rkyv::rancor::Error>(entry)
            .map_err(|e| anyhow::anyhow!("Serialization failed: {}", e))?;

        // Write all data to bundle
        let entry_loc = self.write_to_bundle(&entry_bytes)?;
        let config_loc = self.write_to_bundle(config_toml.as_bytes())?;
        let validation_loc = self.write_to_bundle(validation_json.as_bytes())?;

        // Create index entry
        let index = CandidateIndex {
            config_loc,
            validation_loc,
            entry_loc,
        };
        let index_json = serde_json::to_vec(&index)?;

        // Store in LMDB
        let mut wtxn = self.env.write_txn()?;
        self.index_db.put(&mut wtxn, &entry.uuid_bytes, &index_json)?;
        
        // Also store hash -> uuid mapping
        let hash_bytes = entry.genome_hash.to_le_bytes();
        self.hash_db.put(&mut wtxn, &hash_bytes, &entry.uuid_bytes)?;
        
        wtxn.commit()?;

        Ok(())
    }

    /// Read data from a location in the bundle
    fn read_from_bundle(&self, loc: &BundleLocation) -> Result<Vec<u8>> {
        let mut file = File::open(self.data_file_path())?;
        file.seek(SeekFrom::Start(loc.offset))?;

        // Read length prefix
        let mut len_bytes = [0u8; 4];
        file.read_exact(&mut len_bytes)?;
        let compressed_len = u32::from_le_bytes(len_bytes) as usize;

        // Read compressed data
        let mut compressed = vec![0u8; compressed_len];
        file.read_exact(&mut compressed)?;

        Self::decompress(&compressed)
    }

    /// Get candidate index by UUID
    fn get_index(&self, uuid: Uuid) -> Result<Option<CandidateIndex>> {
        let rtxn = self.env.read_txn()?;
        match self.index_db.get(&rtxn, uuid.as_bytes())? {
            Some(data) => {
                let index: CandidateIndex = serde_json::from_slice(data)?;
                Ok(Some(index))
            }
            None => Ok(None),
        }
    }

    /// Get config TOML by UUID
    pub fn get_config(&self, uuid: Uuid) -> Result<Option<String>> {
        match self.get_index(uuid)? {
            Some(index) => {
                let data = self.read_from_bundle(&index.config_loc)?;
                Ok(Some(String::from_utf8(data)?))
            }
            None => Ok(None),
        }
    }

    /// Get validation JSON by UUID
    pub fn get_validation(&self, uuid: Uuid) -> Result<Option<String>> {
        match self.get_index(uuid)? {
            Some(index) => {
                let data = self.read_from_bundle(&index.validation_loc)?;
                Ok(Some(String::from_utf8(data)?))
            }
            None => Ok(None),
        }
    }

    /// Get candidate entry by genome hash
    pub fn get_by_hash(&self, genome_hash: u64) -> Result<Option<Uuid>> {
        let rtxn = self.env.read_txn()?;
        let hash_bytes = genome_hash.to_le_bytes();
        match self.hash_db.get(&rtxn, &hash_bytes)? {
            Some(uuid_bytes) => {
                let uuid = Uuid::from_bytes(uuid_bytes.try_into().unwrap());
                Ok(Some(uuid))
            }
            None => Ok(None),
        }
    }

    /// Check if candidate exists
    pub fn contains(&self, uuid: Uuid) -> Result<bool> {
        let rtxn = self.env.read_txn()?;
        Ok(self.index_db.get(&rtxn, uuid.as_bytes())?.is_some())
    }

    /// Count total candidates
    pub fn count(&self) -> Result<u64> {
        let rtxn = self.env.read_txn()?;
        Ok(self.index_db.len(&rtxn)?)
    }

    /// List all UUIDs
    pub fn list_uuids(&self) -> Result<Vec<Uuid>> {
        let rtxn = self.env.read_txn()?;
        let mut uuids = Vec::new();
        for result in self.index_db.iter(&rtxn)? {
            let (key, _) = result?;
            if key.len() == 16 {
                let uuid = Uuid::from_bytes(key.try_into().unwrap());
                uuids.push(uuid);
            }
        }
        Ok(uuids)
    }

    /// Sync to disk
    pub fn sync(&mut self) -> Result<()> {
        if let Some(ref mut file) = self.data_file {
            file.sync_all()?;
        }
        self.env.force_sync()?;
        Ok(())
    }

    /// Get bundle stats
    pub fn stats(&self) -> Result<BundleStats> {
        let data_path = self.data_file_path();
        let data_size = if data_path.exists() {
            fs::metadata(&data_path)?.len()
        } else {
            0
        };

        Ok(BundleStats {
            candidate_count: self.count()?,
            data_file_size: data_size,
            compression_level: ULTRA_COMPRESSION_LEVEL,
        })
    }
}

/// Bundle statistics
#[derive(Debug, Clone)]
pub struct BundleStats {
    pub candidate_count: u64,
    pub data_file_size: u64,
    pub compression_level: i32,
}

/// Writer for building report bundles
pub struct ReportBundleWriter {
    bundle: ReportBundle,
    written_count: u64,
}

impl ReportBundleWriter {
    /// Create a new bundle writer
    pub fn new(root_path: impl Into<PathBuf>) -> Result<Self> {
        Ok(Self {
            bundle: ReportBundle::open(root_path)?,
            written_count: 0,
        })
    }

    /// Add a candidate
    pub fn add(
        &mut self,
        uuid: Uuid,
        genome_hash: u64,
        rank: u32,
        validated_generation: u32,
        production_score: f64,
        oos_sharpe: f64,
        oos_cagr: f64,
        max_drawdown: f64,
        pbo: f64,
        dsr: f64,
        degradation_pct: f64,
        splits_passed: u16,
        splits_evaluated: u16,
        config_toml: &str,
        validation_json: &str,
    ) -> Result<()> {
        let entry = CandidateEntry {
            uuid_bytes: *uuid.as_bytes(),
            genome_hash,
            rank,
            validated_generation,
            production_score,
            oos_sharpe,
            oos_cagr,
            max_drawdown,
            pbo,
            dsr,
            degradation_pct,
            splits_passed,
            splits_evaluated,
        };

        self.bundle.add_candidate(&entry, config_toml, validation_json)?;
        self.written_count += 1;
        Ok(())
    }

    /// Finish writing and return stats
    pub fn finish(mut self) -> Result<BundleStats> {
        self.bundle.sync()?;
        self.bundle.stats()
    }

    /// Get current count
    pub fn count(&self) -> u64 {
        self.written_count
    }
}

/// Reader for report bundles
pub struct ReportBundleReader {
    bundle: ReportBundle,
}

impl ReportBundleReader {
    /// Open an existing bundle
    pub fn open(root_path: impl Into<PathBuf>) -> Result<Self> {
        Ok(Self {
            bundle: ReportBundle::open(root_path)?,
        })
    }

    /// Get config TOML
    pub fn get_config(&self, uuid: Uuid) -> Result<Option<String>> {
        self.bundle.get_config(uuid)
    }

    /// Get validation JSON
    pub fn get_validation(&self, uuid: Uuid) -> Result<Option<String>> {
        self.bundle.get_validation(uuid)
    }

    /// Get UUID by genome hash
    pub fn get_by_hash(&self, genome_hash: u64) -> Result<Option<Uuid>> {
        self.bundle.get_by_hash(genome_hash)
    }

    /// List all candidates
    pub fn list(&self) -> Result<Vec<Uuid>> {
        self.bundle.list_uuids()
    }

    /// Get stats
    pub fn stats(&self) -> Result<BundleStats> {
        self.bundle.stats()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bundle_roundtrip() {
        let temp_dir = tempfile::tempdir().unwrap();
        let bundle_path = temp_dir.path().join("test_bundle");

        // Write
        {
            let mut writer = ReportBundleWriter::new(&bundle_path).unwrap();
            let uuid = Uuid::new_v4();
            
            writer.add(
                uuid,
                0x123456789ABCDEF0,
                1,
                10,
                0.85,
                0.75,
                0.12,
                -0.15,
                0.10,
                0.65,
                25.0,
                5,
                6,
                "[strategy]\nname = \"test\"",
                "{\"passed\": true}",
            ).unwrap();

            let stats = writer.finish().unwrap();
            assert_eq!(stats.candidate_count, 1);
        }

        // Read
        {
            let reader = ReportBundleReader::open(&bundle_path).unwrap();
            let uuids = reader.list().unwrap();
            assert_eq!(uuids.len(), 1);

            let config = reader.get_config(uuids[0]).unwrap().unwrap();
            assert!(config.contains("test"));

            let validation = reader.get_validation(uuids[0]).unwrap().unwrap();
            assert!(validation.contains("passed"));
        }
    }

    #[test]
    fn test_ultra_compression() {
        // Simulate typical TOML + JSON content
        let data = r#"
        [strategy]
        name = "golden_momentum_v2"
        universe = "IBOV"
        
        [blocks.momentum]
        lookback = 20
        threshold = 0.05
        "#.repeat(10);

        let compressed = ReportBundle::ultra_compress(data.as_bytes()).unwrap();
        let ratio = data.len() as f64 / compressed.len() as f64;

        println!("Original: {} bytes", data.len());
        println!("Compressed: {} bytes", compressed.len());
        println!("Ratio: {:.2}x", ratio);

        assert!(ratio > 2.0, "Should achieve at least 2x compression");

        let decompressed = ReportBundle::decompress(&compressed).unwrap();
        assert_eq!(data.as_bytes(), decompressed.as_slice());
    }

    #[test]
    fn test_hash_lookup() {
        let temp_dir = tempfile::tempdir().unwrap();
        let bundle_path = temp_dir.path().join("hash_bundle");

        let uuid = Uuid::new_v4();
        let hash = 0xDEADBEEFCAFEBABE_u64;

        {
            let mut writer = ReportBundleWriter::new(&bundle_path).unwrap();
            writer.add(
                uuid, hash, 1, 5, 0.9, 0.8, 0.15, -0.10, 0.05, 0.80, 20.0, 6, 6,
                "config", "{}",
            ).unwrap();
            writer.finish().unwrap();
        }

        {
            let reader = ReportBundleReader::open(&bundle_path).unwrap();
            let found_uuid = reader.get_by_hash(hash).unwrap().unwrap();
            assert_eq!(found_uuid, uuid);
        }
    }
}

