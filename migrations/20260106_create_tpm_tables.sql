-- =============================================================================
-- Trade Parameters Module (TPM) - Database Schema
-- =============================================================================
-- Creates tables for strategy families, templates, and catalogs
-- Project: quant_b3_backtest
-- =============================================================================

-- Famílias de estratégias (15 famílias)
CREATE TABLE IF NOT EXISTS strategy_families (
    id SERIAL PRIMARY KEY,
    slug VARCHAR(50) UNIQUE NOT NULL,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    icon VARCHAR(50),
    color VARCHAR(7),
    hypothesis VARCHAR(30),
    sort_order INT DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Templates de estratégias (116 estratégias)
CREATE TABLE IF NOT EXISTS strategy_templates (
    id SERIAL PRIMARY KEY,
    slug VARCHAR(100) UNIQUE NOT NULL,
    family_id INT REFERENCES strategy_families(id) ON DELETE CASCADE,
    name VARCHAR(200) NOT NULL,
    description TEXT,
    
    -- Classificação
    timeframe VARCHAR(20) NOT NULL CHECK (timeframe IN ('intraday', 'swing', 'position', 'long_term')),
    bar_interval VARCHAR(10) NOT NULL,
    position_type VARCHAR(20) NOT NULL CHECK (position_type IN ('directional', 'pair', 'portfolio', 'multi_strategy')),
    risk_profile VARCHAR(20) NOT NULL CHECK (risk_profile IN ('conservative', 'moderate', 'aggressive', 'very_aggressive')),
    
    -- Configuração TOML completa
    config_toml TEXT NOT NULL,
    config_hash VARCHAR(64),
    
    -- Metadados computacionais
    complexity_tier VARCHAR(20) DEFAULT 'tier2_medium',
    estimated_eval_time_ms INT,
    min_data_years INT DEFAULT 3,
    
    -- Mercados e assets
    markets TEXT[] DEFAULT '{BR,US}',
    asset_classes TEXT[] DEFAULT '{stocks}',
    
    -- UX para leigos
    tooltip_short VARCHAR(500) NOT NULL,
    tooltip_long TEXT,
    recommended_for TEXT,
    difficulty_level INT DEFAULT 2 CHECK (difficulty_level BETWEEN 1 AND 5),
    
    -- Tags para busca
    tags TEXT[],
    
    -- Controle
    enabled BOOLEAN DEFAULT true,
    is_default BOOLEAN DEFAULT false,
    usage_count INT DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Catálogos (grupos de estratégias para seleção rápida)
CREATE TABLE IF NOT EXISTS strategy_catalogs (
    id SERIAL PRIMARY KEY,
    slug VARCHAR(50) UNIQUE NOT NULL,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    icon VARCHAR(50),
    is_system BOOLEAN DEFAULT false,
    is_default BOOLEAN DEFAULT false,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Relação N:N catálogo <-> estratégias
CREATE TABLE IF NOT EXISTS catalog_strategies (
    catalog_id INT REFERENCES strategy_catalogs(id) ON DELETE CASCADE,
    strategy_id INT REFERENCES strategy_templates(id) ON DELETE CASCADE,
    priority INT DEFAULT 0,
    PRIMARY KEY (catalog_id, strategy_id)
);

-- Índices para performance
CREATE INDEX IF NOT EXISTS idx_templates_family ON strategy_templates(family_id);
CREATE INDEX IF NOT EXISTS idx_templates_timeframe ON strategy_templates(timeframe);
CREATE INDEX IF NOT EXISTS idx_templates_risk ON strategy_templates(risk_profile);
CREATE INDEX IF NOT EXISTS idx_templates_enabled ON strategy_templates(enabled);
CREATE INDEX IF NOT EXISTS idx_templates_slug ON strategy_templates(slug);
CREATE INDEX IF NOT EXISTS idx_families_slug ON strategy_families(slug);
CREATE INDEX IF NOT EXISTS idx_catalogs_slug ON strategy_catalogs(slug);

-- Trigger para updated_at
CREATE OR REPLACE FUNCTION update_modified_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

DROP TRIGGER IF EXISTS update_strategy_templates_modtime ON strategy_templates;
CREATE TRIGGER update_strategy_templates_modtime
    BEFORE UPDATE ON strategy_templates
    FOR EACH ROW EXECUTE FUNCTION update_modified_column();

-- =============================================================================
-- SEED DATA: 15 Famílias de Estratégias
-- =============================================================================

INSERT INTO strategy_families (slug, name, description, icon, color, hypothesis, sort_order) VALUES
('intraday', 'Intraday (1h)', 'Estratégias executadas dentro de um dia usando barras de 1 hora', 'Clock', '#00D4FF', 'mixed', 1),
('swing', 'Swing Trading', 'Captura movimentos de preço de 2-10 dias', 'TrendingUp', '#00FF88', 'momentum', 2),
('position', 'Position Trading', 'Posições de semanas a meses baseadas em tendências', 'Target', '#8B5CF6', 'trend', 3),
('pair', 'Pair Trading', 'Explora diferença de preço entre ativos correlacionados', 'GitCompare', '#FF6B6B', 'arbitrage', 4),
('portfolio', 'Portfolio Trading', 'Gestão de múltiplos ativos com regras de alocação', 'Layers', '#FFD93D', 'diversification', 5),
('momentum', 'Momentum', 'Segue tendências estabelecidas que tendem a persistir', 'Zap', '#00FF88', 'momentum', 6),
('mean_reversion', 'Mean Reversion', 'Aposta que preços extremos retornam à média', 'RefreshCw', '#00D4FF', 'mean_reversion', 7),
('breakout', 'Breakout', 'Opera rompimentos de suporte/resistência com volume', 'ArrowUpRight', '#FF9F43', 'breakout', 8),
('sector_rotation', 'Sector Rotation', 'Rotaciona capital entre setores da economia', 'Shuffle', '#A855F7', 'rotation', 9),
('factor', 'Factor Investing', 'Investe baseado em fatores quantificáveis (valor, qualidade)', 'BarChart3', '#6366F1', 'factor', 10),
('seasonal', 'Seasonal Trading', 'Explora padrões que se repetem em certas épocas', 'Calendar', '#EC4899', 'seasonal', 11),
('volatility', 'Volatility Trading', 'Opera mudanças na volatilidade do mercado', 'Activity', '#F97316', 'volatility', 12),
('event_driven', 'Event-Driven', 'Reage a eventos corporativos (balanços, M&A)', 'Bell', '#14B8A6', 'event', 13),
('buy_hold', 'Buy and Hold', 'Estratégia passiva de investimento de longo prazo', 'Wallet', '#84CC16', 'passive', 14),
('multi_strategy', 'Multi-Strategy', 'Combina dinamicamente múltiplas abordagens', 'Boxes', '#F59E0B', 'adaptive', 15)
ON CONFLICT (slug) DO UPDATE SET
    name = EXCLUDED.name,
    description = EXCLUDED.description,
    icon = EXCLUDED.icon,
    color = EXCLUDED.color,
    hypothesis = EXCLUDED.hypothesis,
    sort_order = EXCLUDED.sort_order;

-- =============================================================================
-- SEED DATA: Catálogos Padrão
-- =============================================================================

INSERT INTO strategy_catalogs (slug, name, description, icon, is_system, is_default) VALUES
('all', 'Todas Estratégias', 'Catálogo completo com todas as 116 estratégias', 'Grid', true, true),
('quick_test', 'Teste Rápido', '5 estratégias simples para validação rápida do sistema', 'Zap', true, false),
('institutional', 'Institucional', 'Estratégias com perfil conservador/moderado', 'Building', true, false),
('high_frequency', 'Alta Frequência', 'Estratégias intraday e swing de curto prazo', 'Activity', true, false),
('pairs_only', 'Apenas Pairs', 'Estratégias de pair trading e arbitragem estatística', 'GitCompare', true, false)
ON CONFLICT (slug) DO UPDATE SET
    name = EXCLUDED.name,
    description = EXCLUDED.description,
    icon = EXCLUDED.icon,
    is_system = EXCLUDED.is_system;





