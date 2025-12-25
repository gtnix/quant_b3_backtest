//! Ingestion logic for market data.

use chrono::{NaiveDate, Utc};
use std::io::Write;
use std::path::Path;
use tracing::{debug, error, info, warn};

use crate::brapi::{BrapiClient, BrapiError};
use crate::db::{Database, DbError};

#[derive(Debug)]
pub enum IngestError {
    Brapi(BrapiError),
    Db(DbError),
    QuotaExceeded,
    Io(String),
}

impl From<BrapiError> for IngestError {
    fn from(e: BrapiError) -> Self {
        match e {
            BrapiError::QuotaExceeded => IngestError::QuotaExceeded,
            _ => IngestError::Brapi(e),
        }
    }
}

impl From<DbError> for IngestError {
    fn from(e: DbError) -> Self {
        IngestError::Db(e)
    }
}

impl From<std::io::Error> for IngestError {
    fn from(e: std::io::Error) -> Self {
        IngestError::Io(e.to_string())
    }
}

impl std::fmt::Display for IngestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IngestError::Brapi(e) => write!(f, "Brapi error: {}", e),
            IngestError::Db(e) => write!(f, "Database error: {}", e),
            IngestError::QuotaExceeded => write!(f, "API quota exceeded"),
            IngestError::Io(e) => write!(f, "IO error: {}", e),
        }
    }
}

impl std::error::Error for IngestError {}

/// Refresh the universe with top N stocks by volume.
pub async fn refresh_universe(
    db: &Database,
    brapi: &BrapiClient,
    target: usize,
) -> Result<(), IngestError> {
    info!("Fetching top {} stocks by volume", target);

    let today = Utc::now().date_naive();
    let mut all_stocks = Vec::new();
    let page_size = 100;
    let mut page = 1;

    // Paginate through list endpoint
    while all_stocks.len() < target {
        let (used, limit) = db.get_budget_status().await?;
        if used >= limit {
            return Err(IngestError::QuotaExceeded);
        }

        let (stocks, metrics) = brapi.list_stocks(page_size, page).await?;

        db.log_request("/api/quote/list", &[], &metrics, None)
            .await?;
        db.increment_budget().await?;

        if stocks.is_empty() {
            break;
        }

        for stock in stocks {
            if all_stocks.len() >= target {
                break;
            }
            all_stocks.push(stock);
        }

        page += 1;

        // Safety: max 10 pages for larger universes
        if page > 10 {
            break;
        }
    }

    info!("Found {} stocks", all_stocks.len());

    // Insert instruments and universe membership
    for (rank, stock) in all_stocks.iter().enumerate() {
        db.upsert_instrument(stock).await?;
        db.insert_universe_member("top_volume", &stock.stock, (rank + 1) as i32, today)
            .await?;
    }

    info!(
        "Universe 'top_volume' updated with {} symbols",
        all_stocks.len()
    );
    Ok(())
}

/// Backfill historical data for a universe (daily interval).
pub async fn backfill(
    db: &Database,
    brapi: &BrapiClient,
    universe: &str,
    range: &str,
) -> Result<(), IngestError> {
    let symbols = db.get_universe_symbols(universe).await?;

    if symbols.is_empty() {
        warn!(
            "No symbols in universe '{}'. Run refresh-universe first.",
            universe
        );
        return Ok(());
    }

    info!(
        "Backfilling {} symbols with range '{}' (daily)",
        symbols.len(),
        range
    );

    let batch_size = brapi.max_tickers();
    let mut total_bars = 0;
    let mut processed = 0;

    for chunk in symbols.chunks(batch_size) {
        // Check budget
        let (used, limit) = db.get_budget_status().await?;
        if used >= limit {
            error!(
                "API quota exceeded. Processed {}/{} symbols",
                processed,
                symbols.len()
            );
            return Err(IngestError::QuotaExceeded);
        }

        // GATE CHECK: Filter only ACTIVE tickers
        let mut active_tickers: Vec<&str> = Vec::new();
        for symbol in chunk {
            if db.is_ticker_active(symbol).await.unwrap_or(false) {
                active_tickers.push(symbol.as_str());
            } else {
                debug!("GATE BLOCKED: {} not ACTIVE in provider_universe", symbol);
            }
        }

        if active_tickers.is_empty() {
            processed += chunk.len();
            continue;
        }

        let tickers = active_tickers;

        match brapi.fetch_quotes(&tickers, range, "1d").await {
            Ok((results, metrics)) => {
                db.log_request("/api/quote", &tickers, &metrics, None)
                    .await?;
                db.increment_budget().await?;

                for result in results {
                    db.upsert_instrument_from_quote(&result).await?;

                    let bars = &result.historical_data_price;
                    if bars.is_empty() {
                        eprintln!("  {} - no data", result.symbol);
                        db.update_ingestion_state(
                            &result.symbol,
                            None,
                            None,
                            0,
                            Some("No historical data"),
                        )
                        .await?;
                        continue;
                    }

                    let inserted = db.upsert_ohlcv_batch(&result.symbol, bars).await?;
                    total_bars += inserted;

                    let first_date = bars.first().and_then(|b| b.trading_date());
                    let last_date = bars.last().and_then(|b| b.trading_date());

                    db.update_ingestion_state(
                        &result.symbol,
                        first_date,
                        last_date,
                        bars.len() as i32,
                        None,
                    )
                    .await?;

                    eprintln!(
                        "  {} - {} bars ({:?} to {:?})",
                        result.symbol,
                        bars.len(),
                        first_date,
                        last_date
                    );
                    let _ = std::io::stderr().flush();
                }
            }
            Err(BrapiError::QuotaExceeded) => {
                error!(
                    "API quota exceeded. Processed {}/{} symbols",
                    processed,
                    symbols.len()
                );
                return Err(IngestError::QuotaExceeded);
            }
            Err(e) => {
                warn!("Batch failed: {}. Continuing with next batch.", e);
                for ticker in &tickers {
                    db.update_ingestion_state(ticker, None, None, 0, Some(&e.to_string()))
                        .await?;
                }
            }
        }

        processed += chunk.len();
        eprintln!(
            "\n>>> Progress: {}/{} symbols, {} total bars\n",
            processed,
            symbols.len(),
            total_bars
        );
        let _ = std::io::stderr().flush();
    }

    eprintln!(
        "\n=== Backfill complete: {} symbols, {} bars ===",
        processed, total_bars
    );
    Ok(())
}

/// Backfill intraday data for a universe.
pub async fn backfill_intraday(
    db: &Database,
    brapi: &BrapiClient,
    universe: &str,
    interval: &str,
    range: &str,
) -> Result<(), IngestError> {
    let symbols = db.get_universe_symbols(universe).await?;

    if symbols.is_empty() {
        warn!(
            "No symbols in universe '{}'. Run refresh-universe first.",
            universe
        );
        return Ok(());
    }

    info!(
        "Backfilling intraday: {} symbols, interval={}, range={}",
        symbols.len(),
        interval,
        range
    );

    let mut total_bars = 0;
    let mut processed = 0;
    let mut failed = 0;

    // Process one ticker at a time for intraday to avoid overwhelming API
    for symbol in &symbols {
        // GATE CHECK: Validate ticker against provider_universe
        if !db.is_ticker_active(symbol).await.unwrap_or(false) {
            debug!("GATE BLOCKED: {} not ACTIVE in provider_universe", symbol);
            processed += 1;
            continue;
        }

        // Check budget
        let (used, limit) = db.get_budget_status().await?;
        if used >= limit {
            error!(
                "API quota exceeded. Processed {}/{} symbols",
                processed,
                symbols.len()
            );
            return Err(IngestError::QuotaExceeded);
        }

        match brapi.fetch_historical(symbol, range, interval).await {
            Ok((bars, metrics)) => {
                db.log_request("/api/quote", &[symbol.as_str()], &metrics, None)
                    .await?;
                db.increment_budget().await?;

                if bars.is_empty() {
                    warn!("  {} - no intraday data for interval {}", symbol, interval);
                    failed += 1;
                    continue;
                }

                let inserted = db
                    .upsert_ohlcv_intraday_batch(symbol, interval, &bars)
                    .await?;
                total_bars += inserted;

                info!("  {} - {} bars (interval={})", symbol, bars.len(), interval);
            }
            Err(e) => {
                warn!("  {} - failed: {}", symbol, e);
                failed += 1;
            }
        }

        processed += 1;
        if processed % 50 == 0 {
            info!(
                "Progress: {}/{} symbols, {} bars",
                processed,
                symbols.len(),
                total_bars
            );
        }
    }

    info!(
        "Intraday backfill complete: {} symbols, {} bars, {} failed",
        processed, total_bars, failed
    );
    Ok(())
}

/// Determine optimal range for incremental update.
fn optimal_range(last_date: Option<NaiveDate>) -> &'static str {
    let today = Utc::now().date_naive();

    match last_date {
        None => "max",
        Some(date) => {
            let days = (today - date).num_days();
            match days {
                0..=5 => "5d",
                6..=30 => "1mo",
                31..=90 => "3mo",
                91..=180 => "6mo",
                181..=365 => "1y",
                366..=730 => "2y",
                731..=1825 => "5y",
                _ => "10y",
            }
        }
    }
}

/// Incremental update for a universe.
pub async fn update_incremental(
    db: &Database,
    brapi: &BrapiClient,
    universe: &str,
) -> Result<(), IngestError> {
    let symbols = db.get_universe_symbols(universe).await?;

    if symbols.is_empty() {
        warn!(
            "No symbols in universe '{}'. Run refresh-universe first.",
            universe
        );
        return Ok(());
    }

    info!("Incremental update for {} symbols", symbols.len());

    // Group symbols by optimal range to minimize API calls
    let mut range_groups: std::collections::HashMap<&str, Vec<String>> =
        std::collections::HashMap::new();

    for symbol in &symbols {
        let last_date = db.get_last_bar_date(symbol).await?;
        let range = optimal_range(last_date);
        range_groups.entry(range).or_default().push(symbol.clone());
    }

    info!(
        "Range groups: {:?}",
        range_groups.keys().collect::<Vec<_>>()
    );

    let batch_size = brapi.max_tickers();
    let mut total_new_bars = 0;

    for (range, group_symbols) in range_groups {
        info!(
            "Processing {} symbols with range '{}'",
            group_symbols.len(),
            range
        );

        for chunk in group_symbols.chunks(batch_size) {
            // Check budget
            let (used, limit) = db.get_budget_status().await?;
            if used >= limit {
                error!("API quota exceeded during update");
                return Err(IngestError::QuotaExceeded);
            }

            let tickers: Vec<&str> = chunk.iter().map(|s| s.as_str()).collect();

            match brapi.fetch_quotes(&tickers, range, "1d").await {
                Ok((results, metrics)) => {
                    db.log_request("/api/quote", &tickers, &metrics, None)
                        .await?;
                    db.increment_budget().await?;

                    for result in results {
                        let bars = &result.historical_data_price;
                        let inserted = db.upsert_ohlcv_batch(&result.symbol, bars).await?;
                        total_new_bars += inserted;

                        if inserted > 0 {
                            let last_date = bars.last().and_then(|b| b.trading_date());
                            db.update_ingestion_state(
                                &result.symbol,
                                None,
                                last_date,
                                bars.len() as i32,
                                None,
                            )
                            .await?;
                        }
                    }
                }
                Err(e) => {
                    warn!("Update batch failed: {}", e);
                    for ticker in &tickers {
                        db.update_ingestion_state(ticker, None, None, 0, Some(&e.to_string()))
                            .await?;
                    }
                }
            }
        }
    }

    info!("Incremental update complete: {} new bars", total_new_bars);
    Ok(())
}

/// Verify data integrity.
pub async fn verify_integrity(db: &Database) -> Result<(), IngestError> {
    let issues = db.verify_integrity().await?;

    if issues.is_empty() {
        info!("Data integrity check PASSED: no issues found");
    } else {
        warn!("Data integrity check found {} issues:", issues.len());
        for issue in &issues {
            warn!(
                "  {} [{}]: {} occurrences",
                issue.symbol, issue.issue_type, issue.count
            );
        }
    }

    Ok(())
}

/// Show ingestion status.
pub async fn show_status(db: &Database) -> Result<(), IngestError> {
    let stats = db.get_ingestion_stats().await?;
    let (used, limit) = db.get_budget_status().await?;

    println!("\n=== Market Data Ingestion Status ===\n");
    println!("Instruments:      {}", stats.total_instruments);
    println!("Symbols w/ data:  {}", stats.symbols_with_data);
    println!("Total bars:       {}", stats.total_bars);
    println!(
        "Date range:       {:?} to {:?}",
        stats.earliest_date, stats.latest_date
    );
    println!("Failed symbols:   {}", stats.failed_symbols);
    println!(
        "\nAPI Budget:       {}/{} requests ({:.1}%)",
        used,
        limit,
        (used as f64 / limit as f64) * 100.0
    );
    println!();

    Ok(())
}

/// Generate data freshness report.
pub async fn generate_freshness_report(db: &Database, output: &Path) -> Result<(), IngestError> {
    let freshness = db.get_freshness_data().await?;

    // Create output directory if needed
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let mut content =
        String::from("symbol,interval,last_timestamp,last_date,bar_count,backlog_days\n");
    let today = Utc::now().date_naive();

    for (symbol, interval, last_date, bar_count) in freshness {
        let backlog_days = last_date.map(|d| (today - d).num_days()).unwrap_or(-1);
        let last_date_str = last_date.map(|d| d.to_string()).unwrap_or_default();

        content.push_str(&format!(
            "{},{},{},{},{}\n",
            symbol, interval, last_date_str, bar_count, backlog_days
        ));
    }

    std::fs::write(output, content)?;
    info!("Freshness report written to {}", output.display());

    Ok(())
}
