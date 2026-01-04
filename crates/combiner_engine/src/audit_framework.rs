//! Audit Framework - Sistema de auditoria passo-a-passo para o SCG.
//!
//! Este módulo fornece uma estrutura para auditar cada etapa do processo SCG,
//! gerando evidências documentadas para cada marco.
//!
//! # Marcos de Auditoria
//!
//! - Marco 0: Inicialização da Campanha
//! - Marco 1: Data Integrity Gate
//! - Marco 2: Evolução Genética (Stage A)
//! - Marco 3: Validação Completa (Stage B)
//! - Marco 4: Promotion Gates
//! - Marco 5: Artefatos Finais

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use thiserror::Error;
use uuid::Uuid;

// =============================================================================
// Types
// =============================================================================

/// Marco de auditoria - representa uma etapa do processo.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuditMarco {
    /// Marco 0: Inicialização da campanha
    Initialization,
    /// Marco 1: Data integrity gate
    DataIntegrity,
    /// Marco 2: Evolução genética (Stage A)
    Evolution,
    /// Marco 3: Validação completa (Stage B)
    Validation,
    /// Marco 4: Promotion gates
    PromotionGates,
    /// Marco 5: Artefatos finais
    Artifacts,
}

impl AuditMarco {
    /// Retorna o índice numérico do marco.
    pub fn index(&self) -> u8 {
        match self {
            AuditMarco::Initialization => 0,
            AuditMarco::DataIntegrity => 1,
            AuditMarco::Evolution => 2,
            AuditMarco::Validation => 3,
            AuditMarco::PromotionGates => 4,
            AuditMarco::Artifacts => 5,
        }
    }

    /// Retorna o nome legível do marco.
    pub fn name(&self) -> &'static str {
        match self {
            AuditMarco::Initialization => "Inicialização da Campanha",
            AuditMarco::DataIntegrity => "Data Integrity Gate",
            AuditMarco::Evolution => "Evolução Genética (Stage A)",
            AuditMarco::Validation => "Validação Completa (Stage B)",
            AuditMarco::PromotionGates => "Promotion Gates",
            AuditMarco::Artifacts => "Artefatos Finais",
        }
    }

    /// Retorna o nome do arquivo de evidência.
    pub fn evidence_filename(&self) -> &'static str {
        match self {
            AuditMarco::Initialization => "marco_0_init.json",
            AuditMarco::DataIntegrity => "marco_1_data.json",
            AuditMarco::Evolution => "marco_2_evolution.json",
            AuditMarco::Validation => "marco_3_validation.json",
            AuditMarco::PromotionGates => "marco_4_gates.json",
            AuditMarco::Artifacts => "marco_5_artifacts.json",
        }
    }

    /// Retorna todos os marcos em ordem.
    pub fn all() -> &'static [AuditMarco] {
        &[
            AuditMarco::Initialization,
            AuditMarco::DataIntegrity,
            AuditMarco::Evolution,
            AuditMarco::Validation,
            AuditMarco::PromotionGates,
            AuditMarco::Artifacts,
        ]
    }

    /// Retorna o próximo marco, se houver.
    pub fn next(&self) -> Option<AuditMarco> {
        match self {
            AuditMarco::Initialization => Some(AuditMarco::DataIntegrity),
            AuditMarco::DataIntegrity => Some(AuditMarco::Evolution),
            AuditMarco::Evolution => Some(AuditMarco::Validation),
            AuditMarco::Validation => Some(AuditMarco::PromotionGates),
            AuditMarco::PromotionGates => Some(AuditMarco::Artifacts),
            AuditMarco::Artifacts => None,
        }
    }
}

/// Veredicto de um check individual.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum CheckVerdict {
    Pass,
    Warn,
    Fail,
    Skip,
}

impl CheckVerdict {
    pub fn is_blocking(&self) -> bool {
        matches!(self, CheckVerdict::Fail)
    }
}

/// Um check individual dentro de um marco.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditCheck {
    /// Nome do check
    pub name: String,
    /// Descrição do que está sendo verificado
    pub description: String,
    /// Veredicto
    pub verdict: CheckVerdict,
    /// Mensagem de detalhes
    pub message: String,
    /// Evidência adicional (valores, hashes, etc.)
    #[serde(default)]
    pub evidence: HashMap<String, serde_json::Value>,
    /// Duração em ms
    pub duration_ms: u64,
}

impl AuditCheck {
    /// Cria um check que passou.
    pub fn pass(name: impl Into<String>, description: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            verdict: CheckVerdict::Pass,
            message: message.into(),
            evidence: HashMap::new(),
            duration_ms: 0,
        }
    }

    /// Cria um check que falhou.
    pub fn fail(name: impl Into<String>, description: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            verdict: CheckVerdict::Fail,
            message: message.into(),
            evidence: HashMap::new(),
            duration_ms: 0,
        }
    }

    /// Cria um check com warning.
    pub fn warn(name: impl Into<String>, description: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            verdict: CheckVerdict::Warn,
            message: message.into(),
            evidence: HashMap::new(),
            duration_ms: 0,
        }
    }

    /// Adiciona evidência ao check.
    pub fn with_evidence(mut self, key: impl Into<String>, value: impl Serialize) -> Self {
        if let Ok(v) = serde_json::to_value(value) {
            self.evidence.insert(key.into(), v);
        }
        self
    }

    /// Define a duração.
    pub fn with_duration(mut self, ms: u64) -> Self {
        self.duration_ms = ms;
        self
    }
}

/// Resultado de um marco de auditoria.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarcoResult {
    /// Qual marco
    pub marco: AuditMarco,
    /// Índice numérico
    pub marco_index: u8,
    /// Nome legível
    pub marco_name: String,
    /// Lista de checks realizados
    pub checks: Vec<AuditCheck>,
    /// Veredicto geral (PASS se todos passaram ou warn, FAIL se algum falhou)
    pub verdict: CheckVerdict,
    /// Mensagem resumo
    pub summary: String,
    /// Timestamp de início
    pub started_at: DateTime<Utc>,
    /// Timestamp de fim
    pub completed_at: DateTime<Utc>,
    /// Duração total em ms
    pub duration_ms: u64,
}

impl MarcoResult {
    /// Cria um novo resultado de marco.
    pub fn new(marco: AuditMarco) -> Self {
        let now = Utc::now();
        Self {
            marco,
            marco_index: marco.index(),
            marco_name: marco.name().to_string(),
            checks: Vec::new(),
            verdict: CheckVerdict::Pass,
            summary: String::new(),
            started_at: now,
            completed_at: now,
            duration_ms: 0,
        }
    }

    /// Adiciona um check ao resultado.
    pub fn add_check(&mut self, check: AuditCheck) {
        // Atualiza veredicto geral
        if check.verdict == CheckVerdict::Fail {
            self.verdict = CheckVerdict::Fail;
        } else if check.verdict == CheckVerdict::Warn && self.verdict != CheckVerdict::Fail {
            self.verdict = CheckVerdict::Warn;
        }
        self.checks.push(check);
    }

    /// Finaliza o resultado com timestamp e duração.
    pub fn finalize(&mut self) {
        self.completed_at = Utc::now();
        self.duration_ms = (self.completed_at - self.started_at).num_milliseconds() as u64;
        
        // Gera summary
        let passed = self.checks.iter().filter(|c| c.verdict == CheckVerdict::Pass).count();
        let warned = self.checks.iter().filter(|c| c.verdict == CheckVerdict::Warn).count();
        let failed = self.checks.iter().filter(|c| c.verdict == CheckVerdict::Fail).count();
        
        self.summary = format!(
            "{} checks: {} passed, {} warnings, {} failed",
            self.checks.len(), passed, warned, failed
        );
    }

    /// Verifica se o marco passou (sem falhas).
    pub fn passed(&self) -> bool {
        self.verdict != CheckVerdict::Fail
    }
}

/// Manifesto completo de uma auditoria.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditManifest {
    /// ID único da auditoria
    pub audit_id: String,
    /// Versão do schema
    pub schema_version: String,
    /// ID da campanha sendo auditada
    pub campaign_id: Option<String>,
    /// Path do config usado
    pub config_path: String,
    /// Hash do config
    pub config_hash: String,
    /// Resultados por marco
    pub marcos: HashMap<String, MarcoResult>,
    /// Ordem dos marcos executados
    pub marco_order: Vec<AuditMarco>,
    /// Veredicto final
    pub final_verdict: CheckVerdict,
    /// Resumo geral
    pub summary: String,
    /// Timestamp de início
    pub started_at: DateTime<Utc>,
    /// Timestamp de fim
    pub completed_at: Option<DateTime<Utc>>,
    /// Duração total em ms
    pub duration_ms: u64,
    /// Diretório de output
    pub output_dir: PathBuf,
}

impl AuditManifest {
    /// Cria um novo manifesto.
    pub fn new(config_path: impl Into<String>, config_hash: impl Into<String>, output_dir: impl Into<PathBuf>) -> Self {
        Self {
            audit_id: format!("audit_{}", Uuid::new_v4().to_string().split('-').next().unwrap()),
            schema_version: "1.0.0".to_string(),
            campaign_id: None,
            config_path: config_path.into(),
            config_hash: config_hash.into(),
            marcos: HashMap::new(),
            marco_order: Vec::new(),
            final_verdict: CheckVerdict::Pass,
            summary: String::new(),
            started_at: Utc::now(),
            completed_at: None,
            duration_ms: 0,
            output_dir: output_dir.into(),
        }
    }

    /// Adiciona resultado de um marco.
    pub fn add_marco_result(&mut self, result: MarcoResult) {
        if result.verdict == CheckVerdict::Fail {
            self.final_verdict = CheckVerdict::Fail;
        } else if result.verdict == CheckVerdict::Warn && self.final_verdict != CheckVerdict::Fail {
            self.final_verdict = CheckVerdict::Warn;
        }
        self.marco_order.push(result.marco);
        self.marcos.insert(result.marco.evidence_filename().to_string(), result);
    }

    /// Finaliza o manifesto.
    pub fn finalize(&mut self) {
        self.completed_at = Some(Utc::now());
        if let Some(end) = self.completed_at {
            self.duration_ms = (end - self.started_at).num_milliseconds() as u64;
        }
        
        // Gera summary
        let total_checks: usize = self.marcos.values().map(|m| m.checks.len()).sum();
        let total_passed: usize = self.marcos.values()
            .flat_map(|m| m.checks.iter())
            .filter(|c| c.verdict == CheckVerdict::Pass)
            .count();
        let total_failed: usize = self.marcos.values()
            .flat_map(|m| m.checks.iter())
            .filter(|c| c.verdict == CheckVerdict::Fail)
            .count();
        
        self.summary = format!(
            "{} marcos executados, {} checks totais ({} passed, {} failed)",
            self.marcos.len(), total_checks, total_passed, total_failed
        );
    }

    /// Salva o manifesto e resultados de marcos.
    pub fn save(&self) -> Result<(), AuditError> {
        // Cria diretório
        fs::create_dir_all(&self.output_dir)?;
        
        // Salva manifesto
        let manifest_path = self.output_dir.join("audit_manifest.json");
        let manifest_json = serde_json::to_string_pretty(self)?;
        fs::write(&manifest_path, manifest_json)?;
        
        // Salva cada marco
        for (filename, result) in &self.marcos {
            let path = self.output_dir.join(filename);
            let json = serde_json::to_string_pretty(result)?;
            fs::write(path, json)?;
        }
        
        // Gera summary.md
        let summary = self.generate_markdown_summary();
        let summary_path = self.output_dir.join("summary.md");
        fs::write(summary_path, summary)?;
        
        Ok(())
    }

    /// Gera um relatório Markdown legível.
    pub fn generate_markdown_summary(&self) -> String {
        let mut md = String::new();
        
        md.push_str("# Relatório de Auditoria SCG\n\n");
        md.push_str(&format!("**Audit ID**: `{}`\n\n", self.audit_id));
        md.push_str(&format!("**Config**: `{}`\n\n", self.config_path));
        md.push_str(&format!("**Config Hash**: `{}`\n\n", self.config_hash));
        md.push_str(&format!("**Veredicto Final**: **{:?}**\n\n", self.final_verdict));
        md.push_str(&format!("**Duração Total**: {} ms\n\n", self.duration_ms));
        md.push_str("---\n\n");
        
        md.push_str("## Resumo por Marco\n\n");
        md.push_str("| Marco | Nome | Veredicto | Checks | Duração |\n");
        md.push_str("|-------|------|-----------|--------|----------|\n");
        
        for marco in &self.marco_order {
            if let Some(result) = self.marcos.get(marco.evidence_filename()) {
                let check_summary = format!(
                    "{}/{} passed",
                    result.checks.iter().filter(|c| c.verdict == CheckVerdict::Pass).count(),
                    result.checks.len()
                );
                md.push_str(&format!(
                    "| {} | {} | {:?} | {} | {} ms |\n",
                    result.marco_index, result.marco_name, result.verdict, check_summary, result.duration_ms
                ));
            }
        }
        md.push_str("\n---\n\n");
        
        md.push_str("## Detalhes por Marco\n\n");
        
        for marco in &self.marco_order {
            if let Some(result) = self.marcos.get(marco.evidence_filename()) {
                md.push_str(&format!("### Marco {}: {}\n\n", result.marco_index, result.marco_name));
                md.push_str(&format!("**Veredicto**: {:?}\n\n", result.verdict));
                md.push_str(&format!("**Summary**: {}\n\n", result.summary));
                
                if !result.checks.is_empty() {
                    md.push_str("| Check | Veredicto | Mensagem |\n");
                    md.push_str("|-------|-----------|----------|\n");
                    for check in &result.checks {
                        let emoji = match check.verdict {
                            CheckVerdict::Pass => "✅",
                            CheckVerdict::Warn => "⚠️",
                            CheckVerdict::Fail => "❌",
                            CheckVerdict::Skip => "⏭️",
                        };
                        md.push_str(&format!("| {} {} | {:?} | {} |\n", emoji, check.name, check.verdict, check.message));
                    }
                    md.push_str("\n");
                }
            }
        }
        
        md
    }
}

// =============================================================================
// Errors
// =============================================================================

#[derive(Debug, Error)]
pub enum AuditError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    
    #[error("Config error: {0}")]
    Config(String),
    
    #[error("Marco {0:?} failed: {1}")]
    MarcoFailed(AuditMarco, String),
}

// =============================================================================
// Audit Runner
// =============================================================================

/// Runner de auditoria que executa os marcos.
pub struct AuditRunner {
    /// Manifesto em construção
    pub manifest: AuditManifest,
    /// Modo verbose
    pub verbose: bool,
}

impl AuditRunner {
    /// Cria um novo runner.
    pub fn new(config_path: &str, config_hash: &str, output_base: &Path) -> Self {
        let audit_id = format!("audit_{}", Uuid::new_v4().to_string().split('-').next().unwrap());
        let output_dir = output_base.join(&audit_id);
        
        Self {
            manifest: AuditManifest::new(config_path, config_hash, output_dir),
            verbose: false,
        }
    }

    /// Ativa modo verbose.
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Define o campaign_id.
    pub fn with_campaign_id(mut self, campaign_id: impl Into<String>) -> Self {
        self.manifest.campaign_id = Some(campaign_id.into());
        self
    }

    /// Executa um marco específico com uma função de verificação.
    pub fn run_marco<F>(&mut self, marco: AuditMarco, check_fn: F) -> Result<&MarcoResult, AuditError>
    where
        F: FnOnce(&mut MarcoResult),
    {
        if self.verbose {
            println!("▶ Iniciando Marco {}: {}", marco.index(), marco.name());
        }

        let mut result = MarcoResult::new(marco);
        
        // Executa os checks
        check_fn(&mut result);
        
        // Finaliza
        result.finalize();
        
        if self.verbose {
            let emoji = match result.verdict {
                CheckVerdict::Pass => "✅",
                CheckVerdict::Warn => "⚠️",
                CheckVerdict::Fail => "❌",
                CheckVerdict::Skip => "⏭️",
            };
            println!("{} Marco {}: {:?} - {}", emoji, marco.index(), result.verdict, result.summary);
        }

        // Verifica se falhou
        let passed = result.passed();
        
        // Adiciona ao manifesto
        self.manifest.add_marco_result(result);
        
        if !passed {
            return Err(AuditError::MarcoFailed(
                marco,
                "One or more checks failed".to_string(),
            ));
        }

        Ok(self.manifest.marcos.get(marco.evidence_filename()).unwrap())
    }

    /// Finaliza a auditoria e salva os resultados.
    pub fn finalize(mut self) -> Result<AuditManifest, AuditError> {
        self.manifest.finalize();
        self.manifest.save()?;
        
        if self.verbose {
            println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            println!("  📋 Auditoria Completa: {}", self.manifest.audit_id);
            println!("  📁 Resultados em: {}", self.manifest.output_dir.display());
            println!("  🏁 Veredicto Final: {:?}", self.manifest.final_verdict);
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        }

        Ok(self.manifest)
    }

    /// Retorna o diretório de output.
    pub fn output_dir(&self) -> &Path {
        &self.manifest.output_dir
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_marco_order() {
        assert_eq!(AuditMarco::Initialization.index(), 0);
        assert_eq!(AuditMarco::DataIntegrity.index(), 1);
        assert_eq!(AuditMarco::Evolution.index(), 2);
        assert_eq!(AuditMarco::Validation.index(), 3);
        assert_eq!(AuditMarco::PromotionGates.index(), 4);
        assert_eq!(AuditMarco::Artifacts.index(), 5);
    }

    #[test]
    fn test_check_creation() {
        let check = AuditCheck::pass("test", "Testing", "OK")
            .with_evidence("value", 42)
            .with_duration(100);
        
        assert_eq!(check.verdict, CheckVerdict::Pass);
        assert_eq!(check.duration_ms, 100);
        assert!(check.evidence.contains_key("value"));
    }

    #[test]
    fn test_marco_result() {
        let mut result = MarcoResult::new(AuditMarco::Initialization);
        result.add_check(AuditCheck::pass("check1", "Desc", "OK"));
        result.add_check(AuditCheck::warn("check2", "Desc", "Warning"));
        result.finalize();
        
        assert_eq!(result.verdict, CheckVerdict::Warn);
        assert!(result.passed());
    }

    #[test]
    fn test_marco_result_with_failure() {
        let mut result = MarcoResult::new(AuditMarco::DataIntegrity);
        result.add_check(AuditCheck::pass("check1", "Desc", "OK"));
        result.add_check(AuditCheck::fail("check2", "Desc", "Failed"));
        result.finalize();
        
        assert_eq!(result.verdict, CheckVerdict::Fail);
        assert!(!result.passed());
    }
}










