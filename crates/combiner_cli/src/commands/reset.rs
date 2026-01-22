//! Reset command - clears Hall of Fame and outputs.

use anyhow::{Result, bail};
use std::path::Path;
use std::fs;

/// Execute reset.
pub fn execute(market: &str, yes_really: bool) -> Result<()> {
    if !yes_really {
        println!("========================================");
        println!("  RESET COMMAND");
        println!("========================================");
        println!("");
        println!("  This will DELETE:");
        println!("  - Hall of Fame entries for market: {}", market);
        println!("  - Output directories in output/scg/");
        println!("  - Pending artifacts");
        println!("");
        println!("  To confirm, use: --yes-really");
        println!("");
        bail!("Reset cancelled. Use --yes-really to confirm.");
    }

    println!("========================================");
    println!("  RESET - DELETING DATA");
    println!("========================================");
    println!("  Market: {}", market);
    println!("");

    // 1. Delete output directories
    println!("[1/3] Cleaning output directories...");
    
    let output_dirs = ["output/scg", "output/scg_br", "output/scg_us"];
    let mut dirs_deleted = 0;
    
    for dir in &output_dirs {
        let path = Path::new(dir);
        if path.exists() {
            match fs::remove_dir_all(path) {
                Ok(_) => {
                    println!("  Deleted: {}", dir);
                    dirs_deleted += 1;
                }
                Err(e) => println!("  Skip: {} ({})", dir, e),
            }
        }
    }
    
    // Recreate empty output dir
    fs::create_dir_all("output/scg")?;
    println!("  Recreated: output/scg/");
    println!("  Dirs deleted: {}", dirs_deleted);

    // 2. Clear pending artifacts
    println!("");
    println!("[2/3] Cleaning pending artifacts...");
    
    let pending_dir = Path::new("artifacts/pending");
    let mut files_deleted = 0;
    
    if pending_dir.exists() {
        for entry in fs::read_dir(pending_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_file() {
                fs::remove_file(&path)?;
                files_deleted += 1;
            }
        }
    }
    println!("  Files deleted: {}", files_deleted);

    // 3. Reset Hall of Fame (via psql command)
    println!("");
    println!("[3/3] Resetting Hall of Fame...");
    
    let db_url = std::env::var("NEON_DATABASE_URL")
        .or_else(|_| std::env::var("DATABASE_URL"));
    
    match db_url {
        Ok(url) => {
            let query = if market == "all" {
                "DELETE FROM hall_of_fame; DELETE FROM scg_runs;".to_string()
            } else {
                format!("DELETE FROM hall_of_fame WHERE market = '{}'; DELETE FROM scg_runs WHERE market = '{}';", market, market)
            };
            
            // Use psql to execute
            let output = std::process::Command::new("psql")
                .arg(&url)
                .arg("-c")
                .arg(&query)
                .output();
            
            match output {
                Ok(out) => {
                    if out.status.success() {
                        let stdout = String::from_utf8_lossy(&out.stdout);
                        println!("  {}", stdout.trim());
                    } else {
                        let stderr = String::from_utf8_lossy(&out.stderr);
                        println!("  Skip: {}", stderr.trim());
                    }
                }
                Err(e) => {
                    println!("  Skip: psql not available - {}", e);
                    println!("  Manual cleanup: psql $DATABASE_URL -c \"{}\"", query);
                }
            }
        }
        Err(_) => {
            println!("  Skip: No DATABASE_URL configured");
        }
    }

    // Summary
    println!("");
    println!("========================================");
    println!("  RESET COMPLETE");
    println!("========================================");
    println!("  Output dirs cleaned: {}", dirs_deleted);
    println!("  Pending files deleted: {}", files_deleted);
    println!("");

    Ok(())
}
