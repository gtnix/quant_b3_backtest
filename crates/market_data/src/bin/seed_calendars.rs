//! Calendar Seeder - Generates and inserts 20 years of market calendar data.
//!
//! Usage:
//!   cargo run --bin seed_calendars

use market_data::calendar::db_provider::SourceLayer;
use market_data::calendar::rules_engine::{B3RulesEngine, NYSERulesEngine, RulesEngine};
use std::env;
use std::sync::Arc;
use tracing::{error, info};
use uuid::Uuid;

const START_YEAR: i32 = 2005;
const END_YEAR: i32 = 2025;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    // Load database URL from environment
    dotenvy::dotenv().ok();
    let database_url = env::var("DATABASE_URL").expect("DATABASE_URL must be set");

    info!("Connecting to database...");
    
    // Connect with TLS for Neon
    let rustls_config = rustls::ClientConfig::builder()
        .with_root_certificates(rustls::RootCertStore {
            roots: webpki_roots::TLS_SERVER_ROOTS.iter().cloned().collect(),
        })
        .with_no_client_auth();
    
    let tls = tokio_postgres_rustls::MakeRustlsConnect::new(rustls_config);
    let (client, connection) = tokio_postgres::connect(&database_url, tls).await?;
    
    // Spawn connection handler
    tokio::spawn(async move {
        if let Err(e) = connection.await {
            error!("Connection error: {}", e);
        }
    });

    let client = Arc::new(client);
    info!("Connected to database");

    // Get version IDs
    let b3_version_id = get_active_version_id(&client, "BR").await?;
    let nyse_version_id = get_active_version_id(&client, "US").await?;

    info!("B3 version ID: {}", b3_version_id);
    info!("NYSE version ID: {}", nyse_version_id);

    // Get source IDs
    let b3_source_id = get_source_id(&client, "B3_RULES_2005_2025").await?;
    let nyse_source_id = get_source_id(&client, "NYSE_RULES_2005_2025").await?;
    let nyse_patch_source_id = get_source_id(&client, "NYSE_OFFICIAL_PATCHES").await?;

    // Seed B3 data
    info!("=== Seeding B3 Calendar (2005-2025) ===");
    seed_b3_calendar(&client, b3_version_id, b3_source_id).await?;

    // Seed NYSE data
    info!("=== Seeding NYSE Calendar (2005-2025) ===");
    seed_nyse_calendar(&client, nyse_version_id, nyse_source_id, nyse_patch_source_id).await?;

    info!("=== Calendar seeding complete! ===");

    // Print summary statistics
    print_summary(&client).await?;

    Ok(())
}

async fn get_active_version_id(
    client: &tokio_postgres::Client,
    market: &str,
) -> Result<Uuid, Box<dyn std::error::Error>> {
    let row = client
        .query_one(
            "SELECT id FROM calendar_versions WHERE market = $1 AND is_active = true",
            &[&market],
        )
        .await?;
    Ok(row.get(0))
}

async fn get_source_id(
    client: &tokio_postgres::Client,
    source_id: &str,
) -> Result<Uuid, Box<dyn std::error::Error>> {
    let row = client
        .query_one(
            "SELECT id FROM calendar_sources WHERE source_id = $1",
            &[&source_id],
        )
        .await?;
    Ok(row.get(0))
}

async fn seed_b3_calendar(
    client: &tokio_postgres::Client,
    version_id: Uuid,
    source_id: Uuid,
) -> Result<(), Box<dyn std::error::Error>> {
    let engine = B3RulesEngine::new(Some(source_id));

    let mut total_holidays = 0;
    let mut total_sessions = 0;

    for year in START_YEAR..=END_YEAR {
        info!("Processing B3 year {}...", year);

        // Generate and insert holidays
        let holidays = engine.generate_holidays(year);
        for holiday in &holidays {
            client
                .execute(
                    "INSERT INTO holidays (version_id, holiday_date, market, name, holiday_type, 
                                          early_close_time, late_open_time, source_layer, source_id)
                     VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                     ON CONFLICT (version_id, holiday_date, market) DO NOTHING",
                    &[
                        &version_id,
                        &holiday.holiday_date,
                        &"BR",
                        &holiday.name,
                        &holiday.holiday_type.as_str(),
                        &holiday.early_close_time,
                        &holiday.late_open_time,
                        &SourceLayer::BRules.as_str(),
                        &source_id,
                    ],
                )
                .await?;
        }
        total_holidays += holidays.len();

        // Generate and insert trading sessions
        let sessions = engine.generate_trading_sessions(year);
        for session in &sessions {
            let session_id: Uuid = client
                .query_one(
                    "INSERT INTO trading_sessions (version_id, session_date, market, day_type, source_layer, source_id)
                     VALUES ($1, $2, $3, $4, $5, $6)
                     ON CONFLICT (version_id, session_date, market) DO UPDATE SET day_type = EXCLUDED.day_type
                     RETURNING id",
                    &[
                        &version_id,
                        &session.session_date,
                        &"BR",
                        &session.day_type.as_str(),
                        &SourceLayer::BRules.as_str(),
                        &source_id,
                    ],
                )
                .await?
                .get(0);

            // Insert session periods for regular and partial trading days
            if session.day_type != market_data::calendar::db_provider::DayType::Closed {
                let periods = engine.generate_regular_session_periods(session_id, session.session_date);
                for period in &periods {
                    client
                        .execute(
                            "INSERT INTO session_periods (session_id, period_type, local_open, local_close, utc_offset_minutes)
                             VALUES ($1, $2, $3, $4, $5)
                             ON CONFLICT DO NOTHING",
                            &[
                                &session_id,
                                &period.period_type,
                                &period.local_open,
                                &period.local_close,
                                &period.utc_offset_minutes,
                            ],
                        )
                        .await?;
                }
            }
        }
        total_sessions += sessions.len();

        info!("  Year {}: {} holidays, {} sessions", year, holidays.len(), sessions.len());
    }

    info!("B3 Total: {} holidays, {} sessions", total_holidays, total_sessions);
    Ok(())
}

async fn seed_nyse_calendar(
    client: &tokio_postgres::Client,
    version_id: Uuid,
    source_id: Uuid,
    patch_source_id: Uuid,
) -> Result<(), Box<dyn std::error::Error>> {
    let engine = NYSERulesEngine::new(Some(source_id));

    let mut total_holidays = 0;
    let mut total_sessions = 0;
    let mut total_closures = 0;

    for year in START_YEAR..=END_YEAR {
        info!("Processing NYSE year {}...", year);

        // Generate and insert holidays
        let holidays = engine.generate_holidays(year);
        for holiday in &holidays {
            client
                .execute(
                    "INSERT INTO holidays (version_id, holiday_date, market, name, holiday_type, 
                                          early_close_time, late_open_time, source_layer, source_id)
                     VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                     ON CONFLICT (version_id, holiday_date, market) DO NOTHING",
                    &[
                        &version_id,
                        &holiday.holiday_date,
                        &"US",
                        &holiday.name,
                        &holiday.holiday_type.as_str(),
                        &holiday.early_close_time,
                        &holiday.late_open_time,
                        &SourceLayer::BRules.as_str(),
                        &source_id,
                    ],
                )
                .await?;
        }
        total_holidays += holidays.len();

        // Generate and insert extraordinary closures
        let closures = engine.get_extraordinary_closures(year);
        for closure in &closures {
            client
                .execute(
                    "INSERT INTO extraordinary_closures (version_id, closure_date, market, reason, 
                                                         legal_reference, source_layer, source_id)
                     VALUES ($1, $2, $3, $4, $5, $6, $7)
                     ON CONFLICT (version_id, closure_date, market) DO NOTHING",
                    &[
                        &version_id,
                        &closure.closure_date,
                        &"US",
                        &closure.reason,
                        &closure.legal_reference,
                        &SourceLayer::CPatch.as_str(),
                        &patch_source_id,
                    ],
                )
                .await?;
        }
        total_closures += closures.len();

        // Generate and insert trading sessions
        let sessions = engine.generate_trading_sessions(year);
        for session in &sessions {
            let session_id: Uuid = client
                .query_one(
                    "INSERT INTO trading_sessions (version_id, session_date, market, day_type, source_layer, source_id)
                     VALUES ($1, $2, $3, $4, $5, $6)
                     ON CONFLICT (version_id, session_date, market) DO UPDATE SET day_type = EXCLUDED.day_type
                     RETURNING id",
                    &[
                        &version_id,
                        &session.session_date,
                        &"US",
                        &session.day_type.as_str(),
                        &SourceLayer::BRules.as_str(),
                        &source_id,
                    ],
                )
                .await?
                .get(0);

            // Insert session periods for regular and partial trading days
            if session.day_type != market_data::calendar::db_provider::DayType::Closed {
                let periods = engine.generate_regular_session_periods(session_id, session.session_date);
                for period in &periods {
                    client
                        .execute(
                            "INSERT INTO session_periods (session_id, period_type, local_open, local_close, utc_offset_minutes)
                             VALUES ($1, $2, $3, $4, $5)
                             ON CONFLICT DO NOTHING",
                            &[
                                &session_id,
                                &period.period_type,
                                &period.local_open,
                                &period.local_close,
                                &period.utc_offset_minutes,
                            ],
                        )
                        .await?;
                }
            }
        }
        total_sessions += sessions.len();

        info!(
            "  Year {}: {} holidays, {} closures, {} sessions",
            year,
            holidays.len(),
            closures.len(),
            sessions.len()
        );
    }

    info!(
        "NYSE Total: {} holidays, {} extraordinary closures, {} sessions",
        total_holidays, total_closures, total_sessions
    );
    Ok(())
}

async fn print_summary(client: &tokio_postgres::Client) -> Result<(), Box<dyn std::error::Error>> {
    info!("\n=== Database Summary ===");

    // Count holidays
    let row = client
        .query_one("SELECT COUNT(*) FROM holidays", &[])
        .await?;
    let holiday_count: i64 = row.get(0);
    info!("Total holidays: {}", holiday_count);

    // Count trading sessions
    let row = client
        .query_one("SELECT COUNT(*) FROM trading_sessions", &[])
        .await?;
    let session_count: i64 = row.get(0);
    info!("Total trading sessions: {}", session_count);

    // Count session periods
    let row = client
        .query_one("SELECT COUNT(*) FROM session_periods", &[])
        .await?;
    let period_count: i64 = row.get(0);
    info!("Total session periods: {}", period_count);

    // Count extraordinary closures
    let row = client
        .query_one("SELECT COUNT(*) FROM extraordinary_closures", &[])
        .await?;
    let closure_count: i64 = row.get(0);
    info!("Total extraordinary closures: {}", closure_count);

    // Count by market
    let rows = client
        .query(
            "SELECT market, COUNT(*) FROM trading_sessions GROUP BY market ORDER BY market",
            &[],
        )
        .await?;
    for row in rows {
        let market: &str = row.get(0);
        let count: i64 = row.get(1);
        info!("  {} trading sessions: {}", market, count);
    }

    // Count by year for validation
    info!("\n=== Trading Days by Year ===");
    let rows = client
        .query(
            "SELECT market, EXTRACT(YEAR FROM session_date)::INT as year, COUNT(*) 
             FROM trading_sessions 
             WHERE day_type != 'CLOSED'
             GROUP BY market, year 
             ORDER BY market, year",
            &[],
        )
        .await?;
    
    let mut current_market = String::new();
    for row in rows {
        let market: &str = row.get(0);
        let year: i32 = row.get(1);
        let count: i64 = row.get(2);
        
        if market != current_market {
            info!("\n{}:", market);
            current_market = market.to_string();
        }
        info!("  {}: {} trading days", year, count);
    }

    Ok(())
}
























