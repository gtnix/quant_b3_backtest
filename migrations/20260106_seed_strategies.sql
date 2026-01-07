-- =============================================================================
-- Trade Parameters Module (TPM) - Strategy Templates Seed
-- =============================================================================
-- Seeds 116 pre-configured strategy templates
-- Project: quant_b3_backtest
-- =============================================================================

-- Helper function to get family_id by slug
CREATE OR REPLACE FUNCTION get_family_id(family_slug VARCHAR) RETURNS INT AS $$
DECLARE
    fid INT;
BEGIN
    SELECT id INTO fid FROM strategy_families WHERE slug = family_slug;
    RETURN fid;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- INTRADAY STRATEGIES (22 strategies)
-- =============================================================================

INSERT INTO strategy_templates (slug, family_id, name, description, timeframe, bar_interval, position_type, risk_profile, config_toml, tooltip_short, difficulty_level, tags) VALUES
('orb_breakout_conservative', get_family_id('intraday'), 'ORB Breakout Conservador', 'Opening Range Breakout com filtros conservadores', 'intraday', '1h', 'directional', 'conservative', '[strategy]\ntype = "orb_breakout"\nrisk_profile = "conservative"', 'Compra quando o preço rompe a máxima dos primeiros 30 minutos do pregão. Versão conservadora com stops apertados.', 2, '{orb,breakout,intraday}'),
('orb_breakout_moderate', get_family_id('intraday'), 'ORB Breakout Moderado', 'Opening Range Breakout com parâmetros balanceados', 'intraday', '1h', 'directional', 'moderate', '[strategy]\ntype = "orb_breakout"\nrisk_profile = "moderate"', 'Compra quando o preço rompe a máxima dos primeiros 30 minutos. Versão equilibrada risco/retorno.', 2, '{orb,breakout,intraday}'),
('orb_breakout_aggressive', get_family_id('intraday'), 'ORB Breakout Agressivo', 'Opening Range Breakout com alvos maiores', 'intraday', '1h', 'directional', 'aggressive', '[strategy]\ntype = "orb_breakout"\nrisk_profile = "aggressive"', 'Compra rompimentos matinais buscando movimentos maiores. Mais arriscado.', 3, '{orb,breakout,intraday}'),
('vwap_mean_reversion_conservative', get_family_id('intraday'), 'VWAP Reversão Conservador', 'Reversão à média usando VWAP como referência', 'intraday', '1h', 'directional', 'conservative', '[strategy]\ntype = "vwap_reversion"\nrisk_profile = "conservative"', 'Compra quando o preço cai muito abaixo da VWAP e vende quando sobe muito acima. Conservador.', 2, '{vwap,mean_reversion,intraday}'),
('vwap_mean_reversion_moderate', get_family_id('intraday'), 'VWAP Reversão Moderado', 'Reversão à média VWAP com parâmetros balanceados', 'intraday', '1h', 'directional', 'moderate', '[strategy]\ntype = "vwap_reversion"\nrisk_profile = "moderate"', 'Opera desvios da VWAP esperando retorno à média. Risco moderado.', 2, '{vwap,mean_reversion,intraday}'),
('vwap_trend_following_moderate', get_family_id('intraday'), 'VWAP Tendência Moderado', 'Seguidor de tendência usando VWAP', 'intraday', '1h', 'directional', 'moderate', '[strategy]\ntype = "vwap_trend"\nrisk_profile = "moderate"', 'Compra quando preço está acima da VWAP e tendência é de alta. Moderado.', 2, '{vwap,trend,intraday}'),
('vwap_trend_following_aggressive', get_family_id('intraday'), 'VWAP Tendência Agressivo', 'Seguidor de tendência VWAP agressivo', 'intraday', '1h', 'directional', 'aggressive', '[strategy]\ntype = "vwap_trend"\nrisk_profile = "aggressive"', 'Segue tendências intraday com posições maiores. Mais arriscado.', 3, '{vwap,trend,intraday}'),
('intraday_mean_reversion_rsi_conservative', get_family_id('intraday'), 'RSI Intraday Conservador', 'Reversão usando RSI sobrecomprado/sobrevendido', 'intraday', '1h', 'directional', 'conservative', '[strategy]\ntype = "rsi_reversion"\nrisk_profile = "conservative"', 'Compra quando RSI indica "muito vendido" e vende quando "muito comprado". Conservador.', 2, '{rsi,mean_reversion,intraday}'),
('intraday_mean_reversion_rsi_moderate', get_family_id('intraday'), 'RSI Intraday Moderado', 'Reversão RSI com parâmetros moderados', 'intraday', '1h', 'directional', 'moderate', '[strategy]\ntype = "rsi_reversion"\nrisk_profile = "moderate"', 'Opera extremos do RSI intraday. Risco equilibrado.', 2, '{rsi,mean_reversion,intraday}'),
('intraday_mean_reversion_bb_conservative', get_family_id('intraday'), 'Bollinger Intraday Conservador', 'Reversão usando Bandas de Bollinger', 'intraday', '1h', 'directional', 'conservative', '[strategy]\ntype = "bb_reversion"\nrisk_profile = "conservative"', 'Compra quando preço toca a banda inferior e vende na superior. Conservador.', 2, '{bollinger,mean_reversion,intraday}'),
('intraday_mean_reversion_bb_moderate', get_family_id('intraday'), 'Bollinger Intraday Moderado', 'Reversão Bollinger moderada', 'intraday', '1h', 'directional', 'moderate', '[strategy]\ntype = "bb_reversion"\nrisk_profile = "moderate"', 'Opera toques nas Bandas de Bollinger. Risco moderado.', 2, '{bollinger,mean_reversion,intraday}'),
('intraday_momentum_macd_moderate', get_family_id('intraday'), 'MACD Intraday Moderado', 'Momentum usando cruzamentos MACD', 'intraday', '1h', 'directional', 'moderate', '[strategy]\ntype = "macd_momentum"\nrisk_profile = "moderate"', 'Compra quando MACD cruza para cima e vende quando cruza para baixo. Moderado.', 2, '{macd,momentum,intraday}'),
('intraday_momentum_macd_aggressive', get_family_id('intraday'), 'MACD Intraday Agressivo', 'Momentum MACD agressivo', 'intraday', '1h', 'directional', 'aggressive', '[strategy]\ntype = "macd_momentum"\nrisk_profile = "aggressive"', 'Segue sinais MACD com posições maiores. Mais arriscado.', 3, '{macd,momentum,intraday}'),
('intraday_momentum_adx_moderate', get_family_id('intraday'), 'ADX Intraday Moderado', 'Momentum usando força da tendência ADX', 'intraday', '1h', 'directional', 'moderate', '[strategy]\ntype = "adx_momentum"\nrisk_profile = "moderate"', 'Opera quando ADX indica tendência forte. Risco moderado.', 3, '{adx,momentum,intraday}'),
('intraday_momentum_adx_aggressive', get_family_id('intraday'), 'ADX Intraday Agressivo', 'Momentum ADX agressivo', 'intraday', '1h', 'directional', 'aggressive', '[strategy]\ntype = "adx_momentum"\nrisk_profile = "aggressive"', 'Segue tendências fortes com posições agressivas.', 3, '{adx,momentum,intraday}'),
('gap_fill_conservative', get_family_id('intraday'), 'Gap Fill Conservador', 'Opera fechamento de gaps de abertura', 'intraday', '1h', 'directional', 'conservative', '[strategy]\ntype = "gap_fill"\nrisk_profile = "conservative"', 'Aposta que gaps de abertura serão fechados durante o dia. Conservador.', 2, '{gap,mean_reversion,intraday}'),
('gap_fill_moderate', get_family_id('intraday'), 'Gap Fill Moderado', 'Gap Fill com parâmetros moderados', 'intraday', '1h', 'directional', 'moderate', '[strategy]\ntype = "gap_fill"\nrisk_profile = "moderate"', 'Opera fechamento de gaps com risco moderado.', 2, '{gap,mean_reversion,intraday}'),
('gap_continuation_moderate', get_family_id('intraday'), 'Gap Continuation Moderado', 'Opera continuação de gaps fortes', 'intraday', '1h', 'directional', 'moderate', '[strategy]\ntype = "gap_continuation"\nrisk_profile = "moderate"', 'Aposta que gaps fortes continuam na mesma direção. Moderado.', 3, '{gap,momentum,intraday}'),
('gap_continuation_aggressive', get_family_id('intraday'), 'Gap Continuation Agressivo', 'Gap Continuation agressivo', 'intraday', '1h', 'directional', 'aggressive', '[strategy]\ntype = "gap_continuation"\nrisk_profile = "aggressive"', 'Segue gaps fortes com posições maiores. Arriscado.', 3, '{gap,momentum,intraday}'),
('volume_profile_poc_moderate', get_family_id('intraday'), 'Volume Profile POC', 'Opera em torno do Point of Control', 'intraday', '1h', 'directional', 'moderate', '[strategy]\ntype = "volume_poc"\nrisk_profile = "moderate"', 'Usa o preço com maior volume negociado como referência.', 4, '{volume,intraday}'),
('volume_profile_vah_val_moderate', get_family_id('intraday'), 'Volume Profile VAH/VAL', 'Opera nas extremidades do perfil de volume', 'intraday', '1h', 'directional', 'moderate', '[strategy]\ntype = "volume_profile"\nrisk_profile = "moderate"', 'Opera suporte/resistência baseado em níveis de volume.', 4, '{volume,intraday}'),
('news_based_volatility_moderate', get_family_id('intraday'), 'News Volatility', 'Opera volatilidade após notícias', 'intraday', '1h', 'directional', 'moderate', '[strategy]\ntype = "news_volatility"\nrisk_profile = "moderate"', 'Posiciona antes ou depois de eventos com impacto no preço.', 4, '{news,volatility,intraday}')
ON CONFLICT (slug) DO UPDATE SET name = EXCLUDED.name, description = EXCLUDED.description, tooltip_short = EXCLUDED.tooltip_short;

-- =============================================================================
-- SWING TRADING STRATEGIES (12 strategies)
-- =============================================================================

INSERT INTO strategy_templates (slug, family_id, name, description, timeframe, bar_interval, position_type, risk_profile, config_toml, tooltip_short, difficulty_level, tags) VALUES
('swing_momentum_ma_crossover_conservative', get_family_id('swing'), 'MA Crossover Conservador', 'Cruzamento de médias móveis conservador', 'swing', '1D', 'directional', 'conservative', '[strategy]\ntype = "ma_crossover"\nrisk_profile = "conservative"', 'Compra quando média curta cruza acima da longa (ex: 20 acima de 50). Conservador.', 2, '{ma,momentum,swing}'),
('swing_momentum_ma_crossover_moderate', get_family_id('swing'), 'MA Crossover Moderado', 'Cruzamento de médias móveis moderado', 'swing', '1D', 'directional', 'moderate', '[strategy]\ntype = "ma_crossover"\nrisk_profile = "moderate"', 'Segue cruzamentos de médias móveis com risco equilibrado.', 2, '{ma,momentum,swing}'),
('swing_momentum_macd_moderate', get_family_id('swing'), 'MACD Swing Moderado', 'MACD para swing trade', 'swing', '1D', 'directional', 'moderate', '[strategy]\ntype = "macd_swing"\nrisk_profile = "moderate"', 'Opera cruzamentos MACD em timeframe diário.', 2, '{macd,momentum,swing}'),
('swing_momentum_macd_aggressive', get_family_id('swing'), 'MACD Swing Agressivo', 'MACD swing agressivo', 'swing', '1D', 'directional', 'aggressive', '[strategy]\ntype = "macd_swing"\nrisk_profile = "aggressive"', 'Segue MACD diário com posições maiores.', 3, '{macd,momentum,swing}'),
('swing_mean_reversion_bb_conservative', get_family_id('swing'), 'Bollinger Swing Conservador', 'Reversão Bollinger para swing', 'swing', '1D', 'directional', 'conservative', '[strategy]\ntype = "bb_swing"\nrisk_profile = "conservative"', 'Compra em toques na banda inferior diária. Conservador.', 2, '{bollinger,mean_reversion,swing}'),
('swing_mean_reversion_bb_moderate', get_family_id('swing'), 'Bollinger Swing Moderado', 'Reversão Bollinger swing moderado', 'swing', '1D', 'directional', 'moderate', '[strategy]\ntype = "bb_swing"\nrisk_profile = "moderate"', 'Opera Bandas de Bollinger no gráfico diário. Moderado.', 2, '{bollinger,mean_reversion,swing}'),
('swing_mean_reversion_rsi_conservative', get_family_id('swing'), 'RSI Swing Conservador', 'RSI para swing trade conservador', 'swing', '1D', 'directional', 'conservative', '[strategy]\ntype = "rsi_swing"\nrisk_profile = "conservative"', 'Compra RSI sobrevendido no diário. Conservador.', 2, '{rsi,mean_reversion,swing}'),
('swing_mean_reversion_rsi_moderate', get_family_id('swing'), 'RSI Swing Moderado', 'RSI swing moderado', 'swing', '1D', 'directional', 'moderate', '[strategy]\ntype = "rsi_swing"\nrisk_profile = "moderate"', 'Opera extremos RSI diários com risco moderado.', 2, '{rsi,mean_reversion,swing}'),
('swing_breakout_channel_moderate', get_family_id('swing'), 'Channel Breakout Moderado', 'Rompimento de canal para swing', 'swing', '1D', 'directional', 'moderate', '[strategy]\ntype = "channel_breakout"\nrisk_profile = "moderate"', 'Compra rompimento de canal de preço no diário.', 3, '{breakout,channel,swing}'),
('swing_breakout_channel_aggressive', get_family_id('swing'), 'Channel Breakout Agressivo', 'Rompimento de canal agressivo', 'swing', '1D', 'directional', 'aggressive', '[strategy]\ntype = "channel_breakout"\nrisk_profile = "aggressive"', 'Opera rompimentos de canal com posições maiores.', 3, '{breakout,channel,swing}'),
('swing_breakout_volume_moderate', get_family_id('swing'), 'Volume Breakout Moderado', 'Rompimento confirmado por volume', 'swing', '1D', 'directional', 'moderate', '[strategy]\ntype = "volume_breakout"\nrisk_profile = "moderate"', 'Compra rompimentos com volume acima da média.', 3, '{breakout,volume,swing}'),
('swing_breakout_volume_aggressive', get_family_id('swing'), 'Volume Breakout Agressivo', 'Volume breakout agressivo', 'swing', '1D', 'directional', 'aggressive', '[strategy]\ntype = "volume_breakout"\nrisk_profile = "aggressive"', 'Opera rompimentos volumosos com posições maiores.', 3, '{breakout,volume,swing}')
ON CONFLICT (slug) DO UPDATE SET name = EXCLUDED.name, description = EXCLUDED.description, tooltip_short = EXCLUDED.tooltip_short;

-- =============================================================================
-- POSITION TRADING STRATEGIES (6 strategies)
-- =============================================================================

INSERT INTO strategy_templates (slug, family_id, name, description, timeframe, bar_interval, position_type, risk_profile, config_toml, tooltip_short, difficulty_level, tags) VALUES
('position_trend_following_ma_conservative', get_family_id('position'), 'Trend Following MA Conservador', 'Seguidor de tendência longo prazo', 'position', '1D', 'directional', 'conservative', '[strategy]\ntype = "trend_ma"\nrisk_profile = "conservative"', 'Segue tendências de longo prazo usando médias de 50/200 dias. Conservador.', 2, '{trend,ma,position}'),
('position_trend_following_ma_moderate', get_family_id('position'), 'Trend Following MA Moderado', 'Seguidor de tendência moderado', 'position', '1D', 'directional', 'moderate', '[strategy]\ntype = "trend_ma"\nrisk_profile = "moderate"', 'Segue tendências de semanas a meses. Risco moderado.', 2, '{trend,ma,position}'),
('position_trend_following_adx_moderate', get_family_id('position'), 'Trend Following ADX', 'Seguidor de tendência com ADX', 'position', '1D', 'directional', 'moderate', '[strategy]\ntype = "trend_adx"\nrisk_profile = "moderate"', 'Opera apenas quando ADX confirma tendência forte.', 3, '{trend,adx,position}'),
('position_fundamental_value_conservative', get_family_id('position'), 'Value Investing Conservador', 'Investimento em valor conservador', 'long_term', '1D', 'directional', 'conservative', '[strategy]\ntype = "value"\nrisk_profile = "conservative"', 'Compra ações baratas pelo P/L e P/VP. Longo prazo.', 3, '{fundamental,value,position}'),
('position_fundamental_value_moderate', get_family_id('position'), 'Value Investing Moderado', 'Value investing moderado', 'long_term', '1D', 'directional', 'moderate', '[strategy]\ntype = "value"\nrisk_profile = "moderate"', 'Investe em ações subvalorizadas com risco moderado.', 3, '{fundamental,value,position}'),
('position_fundamental_quality_moderate', get_family_id('position'), 'Quality Investing', 'Investimento em qualidade', 'long_term', '1D', 'directional', 'moderate', '[strategy]\ntype = "quality"\nrisk_profile = "moderate"', 'Compra empresas com alta qualidade (ROE, margem, baixa dívida).', 3, '{fundamental,quality,position}')
ON CONFLICT (slug) DO UPDATE SET name = EXCLUDED.name, description = EXCLUDED.description, tooltip_short = EXCLUDED.tooltip_short;

-- =============================================================================
-- PAIR TRADING STRATEGIES (12 strategies)
-- =============================================================================

INSERT INTO strategy_templates (slug, family_id, name, description, timeframe, bar_interval, position_type, risk_profile, config_toml, tooltip_short, difficulty_level, tags) VALUES
('pair_trading_cointegration_conservative', get_family_id('pair'), 'Cointegração Conservador', 'Pair trading por cointegração conservador', 'swing', '1D', 'pair', 'conservative', '[strategy]\ntype = "cointegration"\nrisk_profile = "conservative"', 'Compra uma ação e vende outra correlacionada quando divergem. Conservador.', 4, '{pair,cointegration,stat_arb}'),
('pair_trading_cointegration_moderate', get_family_id('pair'), 'Cointegração Moderado', 'Pair trading cointegração moderado', 'swing', '1D', 'pair', 'moderate', '[strategy]\ntype = "cointegration"\nrisk_profile = "moderate"', 'Opera pares cointegrados com risco moderado.', 4, '{pair,cointegration,stat_arb}'),
('pair_trading_cointegration_aggressive', get_family_id('pair'), 'Cointegração Agressivo', 'Pair trading cointegração agressivo', 'swing', '1D', 'pair', 'aggressive', '[strategy]\ntype = "cointegration"\nrisk_profile = "aggressive"', 'Pair trading com desvios menores e posições maiores.', 4, '{pair,cointegration,stat_arb}'),
('pair_trading_cointegration_short_term', get_family_id('pair'), 'Cointegração Curto Prazo', 'Pair trading rápido', 'intraday', '1h', 'pair', 'moderate', '[strategy]\ntype = "cointegration_fast"\nrisk_profile = "moderate"', 'Pair trading intraday para operações rápidas.', 4, '{pair,cointegration,intraday}'),
('pair_trading_distance_conservative', get_family_id('pair'), 'Distance Conservador', 'Pair trading por distância conservador', 'swing', '1D', 'pair', 'conservative', '[strategy]\ntype = "distance"\nrisk_profile = "conservative"', 'Opera quando spread entre pares atinge extremos. Conservador.', 3, '{pair,distance,stat_arb}'),
('pair_trading_distance_moderate', get_family_id('pair'), 'Distance Moderado', 'Distance method moderado', 'swing', '1D', 'pair', 'moderate', '[strategy]\ntype = "distance"\nrisk_profile = "moderate"', 'Pair trading por desvio de distância histórica.', 3, '{pair,distance,stat_arb}'),
('pair_trading_distance_aggressive', get_family_id('pair'), 'Distance Agressivo', 'Distance method agressivo', 'swing', '1D', 'pair', 'aggressive', '[strategy]\ntype = "distance"\nrisk_profile = "aggressive"', 'Opera desvios menores com posições maiores.', 4, '{pair,distance,stat_arb}'),
('pair_trading_distance_short_term', get_family_id('pair'), 'Distance Curto Prazo', 'Distance method rápido', 'intraday', '1h', 'pair', 'moderate', '[strategy]\ntype = "distance_fast"\nrisk_profile = "moderate"', 'Pair trading intraday por distância.', 4, '{pair,distance,intraday}'),
('stat_arb_multi_pair_conservative', get_family_id('pair'), 'Multi-Pair Conservador', 'Arbitragem estatística multi-par', 'swing', '1D', 'portfolio', 'conservative', '[strategy]\ntype = "multi_pair"\nrisk_profile = "conservative"', 'Opera múltiplos pares simultaneamente. Conservador.', 5, '{stat_arb,multi_pair}'),
('stat_arb_multi_pair_moderate', get_family_id('pair'), 'Multi-Pair Moderado', 'Multi-pair moderado', 'swing', '1D', 'portfolio', 'moderate', '[strategy]\ntype = "multi_pair"\nrisk_profile = "moderate"', 'Carteira de pares diversificada.', 5, '{stat_arb,multi_pair}'),
('stat_arb_multi_pair_aggressive', get_family_id('pair'), 'Multi-Pair Agressivo', 'Multi-pair agressivo', 'swing', '1D', 'portfolio', 'aggressive', '[strategy]\ntype = "multi_pair"\nrisk_profile = "aggressive"', 'Multi-pair com alavancagem moderada.', 5, '{stat_arb,multi_pair}'),
('stat_arb_multi_pair_diversified', get_family_id('pair'), 'Multi-Pair Diversificado', 'Multi-pair com diversificação setorial', 'swing', '1D', 'portfolio', 'moderate', '[strategy]\ntype = "multi_pair_div"\nrisk_profile = "moderate"', 'Pares de diferentes setores para reduzir risco.', 5, '{stat_arb,multi_pair,diversification}')
ON CONFLICT (slug) DO UPDATE SET name = EXCLUDED.name, description = EXCLUDED.description, tooltip_short = EXCLUDED.tooltip_short;

-- =============================================================================
-- PORTFOLIO STRATEGIES (14 strategies)
-- =============================================================================

INSERT INTO strategy_templates (slug, family_id, name, description, timeframe, bar_interval, position_type, risk_profile, config_toml, tooltip_short, difficulty_level, tags) VALUES
('portfolio_equal_weight_5_assets', get_family_id('portfolio'), 'Equal Weight 5 Ativos', 'Portfólio com peso igual em 5 ativos', 'position', '1D', 'portfolio', 'conservative', '[strategy]\ntype = "equal_weight"\nsize = 5', 'Divide o capital igualmente entre 5 ações. Simples e diversificado.', 1, '{portfolio,equal_weight}'),
('portfolio_equal_weight_10_assets', get_family_id('portfolio'), 'Equal Weight 10 Ativos', 'Portfólio com peso igual em 10 ativos', 'position', '1D', 'portfolio', 'conservative', '[strategy]\ntype = "equal_weight"\nsize = 10', 'Divide o capital igualmente entre 10 ações.', 1, '{portfolio,equal_weight}'),
('portfolio_equal_weight_20_assets', get_family_id('portfolio'), 'Equal Weight 20 Ativos', 'Portfólio com peso igual em 20 ativos', 'position', '1D', 'portfolio', 'conservative', '[strategy]\ntype = "equal_weight"\nsize = 20', 'Divide o capital igualmente entre 20 ações.', 1, '{portfolio,equal_weight}'),
('portfolio_equal_weight_30_assets', get_family_id('portfolio'), 'Equal Weight 30 Ativos', 'Portfólio com peso igual em 30 ativos', 'position', '1D', 'portfolio', 'conservative', '[strategy]\ntype = "equal_weight"\nsize = 30', 'Divide o capital igualmente entre 30 ações. Máxima diversificação.', 1, '{portfolio,equal_weight}'),
('portfolio_risk_parity_conservative', get_family_id('portfolio'), 'Risk Parity Conservador', 'Paridade de risco conservador', 'position', '1D', 'portfolio', 'conservative', '[strategy]\ntype = "risk_parity"\nrisk_profile = "conservative"', 'Aloca para que cada ativo contribua igual ao risco total. Conservador.', 4, '{portfolio,risk_parity}'),
('portfolio_risk_parity_moderate', get_family_id('portfolio'), 'Risk Parity Moderado', 'Paridade de risco moderado', 'position', '1D', 'portfolio', 'moderate', '[strategy]\ntype = "risk_parity"\nrisk_profile = "moderate"', 'Risk parity com risco moderado.', 4, '{portfolio,risk_parity}'),
('portfolio_risk_parity_aggressive', get_family_id('portfolio'), 'Risk Parity Agressivo', 'Paridade de risco agressivo', 'position', '1D', 'portfolio', 'aggressive', '[strategy]\ntype = "risk_parity"\nrisk_profile = "aggressive"', 'Risk parity buscando maior retorno.', 4, '{portfolio,risk_parity}'),
('portfolio_risk_parity_leveraged', get_family_id('portfolio'), 'Risk Parity Alavancado', 'Paridade de risco com alavancagem', 'position', '1D', 'portfolio', 'very_aggressive', '[strategy]\ntype = "risk_parity"\nleveraged = true', 'Risk parity com alavancagem para amplificar retornos.', 5, '{portfolio,risk_parity,leveraged}'),
('portfolio_min_variance_conservative', get_family_id('portfolio'), 'Mínima Variância Conservador', 'Portfólio de mínima variância', 'position', '1D', 'portfolio', 'conservative', '[strategy]\ntype = "min_variance"\nrisk_profile = "conservative"', 'Busca a combinação de ativos com menor volatilidade possível.', 4, '{portfolio,min_variance}'),
('portfolio_min_variance_moderate', get_family_id('portfolio'), 'Mínima Variância Moderado', 'Min variance moderado', 'position', '1D', 'portfolio', 'moderate', '[strategy]\ntype = "min_variance"\nrisk_profile = "moderate"', 'Minimiza volatilidade com restrições moderadas.', 4, '{portfolio,min_variance}'),
('portfolio_min_variance_long_short', get_family_id('portfolio'), 'Mínima Variância Long/Short', 'Min variance com posições short', 'position', '1D', 'portfolio', 'moderate', '[strategy]\ntype = "min_variance"\nlong_short = true', 'Usa posições vendidas para reduzir ainda mais o risco.', 5, '{portfolio,min_variance,long_short}'),
('portfolio_max_sharpe_conservative', get_family_id('portfolio'), 'Max Sharpe Conservador', 'Maximiza Sharpe ratio conservador', 'position', '1D', 'portfolio', 'conservative', '[strategy]\ntype = "max_sharpe"\nrisk_profile = "conservative"', 'Busca a melhor relação retorno/risco possível. Conservador.', 4, '{portfolio,max_sharpe}'),
('portfolio_max_sharpe_moderate', get_family_id('portfolio'), 'Max Sharpe Moderado', 'Max Sharpe moderado', 'position', '1D', 'portfolio', 'moderate', '[strategy]\ntype = "max_sharpe"\nrisk_profile = "moderate"', 'Otimiza Sharpe com risco moderado.', 4, '{portfolio,max_sharpe}'),
('portfolio_max_sharpe_aggressive', get_family_id('portfolio'), 'Max Sharpe Agressivo', 'Max Sharpe agressivo', 'position', '1D', 'portfolio', 'aggressive', '[strategy]\ntype = "max_sharpe"\nrisk_profile = "aggressive"', 'Maximiza Sharpe aceitando mais concentração.', 4, '{portfolio,max_sharpe}')
ON CONFLICT (slug) DO UPDATE SET name = EXCLUDED.name, description = EXCLUDED.description, tooltip_short = EXCLUDED.tooltip_short;

-- =============================================================================
-- MOMENTUM STRATEGIES (8 strategies)
-- =============================================================================

INSERT INTO strategy_templates (slug, family_id, name, description, timeframe, bar_interval, position_type, risk_profile, config_toml, tooltip_short, difficulty_level, tags) VALUES
('momentum_cross_sectional_3m', get_family_id('momentum'), 'Cross-Sectional 3M', 'Momentum cross-sectional 3 meses', 'position', '1D', 'portfolio', 'moderate', '[strategy]\ntype = "cross_sectional"\nlookback = 63', 'Compra os melhores do ranking de retorno dos últimos 3 meses.', 3, '{momentum,cross_sectional}'),
('momentum_cross_sectional_6m', get_family_id('momentum'), 'Cross-Sectional 6M', 'Momentum cross-sectional 6 meses', 'position', '1D', 'portfolio', 'moderate', '[strategy]\ntype = "cross_sectional"\nlookback = 126', 'Compra winners e vende losers dos últimos 6 meses.', 3, '{momentum,cross_sectional}'),
('momentum_cross_sectional_12m', get_family_id('momentum'), 'Cross-Sectional 12M', 'Momentum cross-sectional 12 meses', 'position', '1D', 'portfolio', 'moderate', '[strategy]\ntype = "cross_sectional"\nlookback = 252', 'Clássico momentum de 12 meses menos 1 mês.', 3, '{momentum,cross_sectional}'),
('momentum_cross_sectional_multi_period', get_family_id('momentum'), 'Cross-Sectional Multi', 'Momentum multi-período', 'position', '1D', 'portfolio', 'moderate', '[strategy]\ntype = "cross_sectional_multi"', 'Combina sinais de momentum de vários períodos.', 4, '{momentum,cross_sectional}'),
('momentum_time_series_50d', get_family_id('momentum'), 'Time Series 50D', 'Momentum time series 50 dias', 'swing', '1D', 'directional', 'moderate', '[strategy]\ntype = "time_series"\nlookback = 50', 'Compra se retorno dos últimos 50 dias foi positivo.', 2, '{momentum,time_series}'),
('momentum_time_series_200d', get_family_id('momentum'), 'Time Series 200D', 'Momentum time series 200 dias', 'position', '1D', 'directional', 'moderate', '[strategy]\ntype = "time_series"\nlookback = 200', 'Segue tendência de longo prazo (200 dias).', 2, '{momentum,time_series}'),
('momentum_time_series_dual_ma', get_family_id('momentum'), 'Dual Moving Average', 'Momentum com duas médias', 'swing', '1D', 'directional', 'moderate', '[strategy]\ntype = "dual_ma"', 'Compra quando MA curta cruza acima da longa.', 2, '{momentum,ma}'),
('momentum_time_series_adaptive', get_family_id('momentum'), 'Adaptive Momentum', 'Momentum adaptativo', 'swing', '1D', 'directional', 'moderate', '[strategy]\ntype = "adaptive_momentum"', 'Ajusta lookback baseado na volatilidade do mercado.', 4, '{momentum,adaptive}')
ON CONFLICT (slug) DO UPDATE SET name = EXCLUDED.name, description = EXCLUDED.description, tooltip_short = EXCLUDED.tooltip_short;

-- =============================================================================
-- MEAN REVERSION STRATEGIES (8 strategies)
-- =============================================================================

INSERT INTO strategy_templates (slug, family_id, name, description, timeframe, bar_interval, position_type, risk_profile, config_toml, tooltip_short, difficulty_level, tags) VALUES
('mean_reversion_bb_conservative', get_family_id('mean_reversion'), 'Bollinger Conservador', 'Reversão Bollinger conservador', 'swing', '1D', 'directional', 'conservative', '[strategy]\ntype = "bb_reversion"\nrisk_profile = "conservative"', 'Compra toques na banda inferior de Bollinger. Conservador.', 2, '{bollinger,mean_reversion}'),
('mean_reversion_bb_moderate', get_family_id('mean_reversion'), 'Bollinger Moderado', 'Reversão Bollinger moderado', 'swing', '1D', 'directional', 'moderate', '[strategy]\ntype = "bb_reversion"\nrisk_profile = "moderate"', 'Opera extremos das Bandas de Bollinger. Moderado.', 2, '{bollinger,mean_reversion}'),
('mean_reversion_bb_aggressive', get_family_id('mean_reversion'), 'Bollinger Agressivo', 'Reversão Bollinger agressivo', 'swing', '1D', 'directional', 'aggressive', '[strategy]\ntype = "bb_reversion"\nrisk_profile = "aggressive"', 'Bollinger com posições maiores e stops mais largos.', 3, '{bollinger,mean_reversion}'),
('mean_reversion_bb_trend_filtered', get_family_id('mean_reversion'), 'Bollinger Trend Filter', 'Bollinger com filtro de tendência', 'swing', '1D', 'directional', 'moderate', '[strategy]\ntype = "bb_filtered"', 'Só opera reversões a favor da tendência maior.', 3, '{bollinger,mean_reversion,trend}'),
('mean_reversion_rsi_conservative', get_family_id('mean_reversion'), 'RSI Conservador', 'Reversão RSI conservador', 'swing', '1D', 'directional', 'conservative', '[strategy]\ntype = "rsi_reversion"\nrisk_profile = "conservative"', 'Compra RSI abaixo de 30, vende acima de 70. Conservador.', 2, '{rsi,mean_reversion}'),
('mean_reversion_rsi_moderate', get_family_id('mean_reversion'), 'RSI Moderado', 'Reversão RSI moderado', 'swing', '1D', 'directional', 'moderate', '[strategy]\ntype = "rsi_reversion"\nrisk_profile = "moderate"', 'Opera extremos RSI com risco moderado.', 2, '{rsi,mean_reversion}'),
('mean_reversion_rsi_aggressive', get_family_id('mean_reversion'), 'RSI Agressivo', 'Reversão RSI agressivo', 'swing', '1D', 'directional', 'aggressive', '[strategy]\ntype = "rsi_reversion"\nrisk_profile = "aggressive"', 'RSI com níveis menos extremos e posições maiores.', 3, '{rsi,mean_reversion}'),
('mean_reversion_rsi_trend_filtered', get_family_id('mean_reversion'), 'RSI Trend Filter', 'RSI com filtro de tendência', 'swing', '1D', 'directional', 'moderate', '[strategy]\ntype = "rsi_filtered"', 'Só opera RSI a favor da tendência de longo prazo.', 3, '{rsi,mean_reversion,trend}')
ON CONFLICT (slug) DO UPDATE SET name = EXCLUDED.name, description = EXCLUDED.description, tooltip_short = EXCLUDED.tooltip_short;

-- =============================================================================
-- BREAKOUT STRATEGIES (6 strategies)
-- =============================================================================

INSERT INTO strategy_templates (slug, family_id, name, description, timeframe, bar_interval, position_type, risk_profile, config_toml, tooltip_short, difficulty_level, tags) VALUES
('breakout_donchian_20d', get_family_id('breakout'), 'Donchian 20D', 'Donchian Channel 20 dias', 'swing', '1D', 'directional', 'moderate', '[strategy]\ntype = "donchian"\nperiod = 20', 'Compra nova máxima de 20 dias, vende nova mínima. Clássico Turtle.', 2, '{breakout,donchian}'),
('breakout_donchian_55d', get_family_id('breakout'), 'Donchian 55D', 'Donchian Channel 55 dias', 'position', '1D', 'directional', 'moderate', '[strategy]\ntype = "donchian"\nperiod = 55', 'Rompimento de 55 dias para movimentos maiores.', 2, '{breakout,donchian}'),
('breakout_donchian_dual_channel', get_family_id('breakout'), 'Donchian Dual', 'Donchian com dois canais', 'swing', '1D', 'directional', 'moderate', '[strategy]\ntype = "donchian_dual"', 'Entrada em 55 dias, saída em 20 dias. Sistema Turtle completo.', 3, '{breakout,donchian}'),
('breakout_volatility_expansion_conservative', get_family_id('breakout'), 'Volatility Expansion Conservador', 'Expansão de volatilidade conservador', 'swing', '1D', 'directional', 'conservative', '[strategy]\ntype = "vol_expansion"\nrisk_profile = "conservative"', 'Compra quando volatilidade expande após contração. Conservador.', 3, '{breakout,volatility}'),
('breakout_volatility_expansion_moderate', get_family_id('breakout'), 'Volatility Expansion Moderado', 'Expansão de volatilidade moderado', 'swing', '1D', 'directional', 'moderate', '[strategy]\ntype = "vol_expansion"\nrisk_profile = "moderate"', 'Opera squeeze de volatilidade com risco moderado.', 3, '{breakout,volatility}'),
('breakout_volatility_expansion_aggressive', get_family_id('breakout'), 'Volatility Expansion Agressivo', 'Expansão de volatilidade agressivo', 'swing', '1D', 'directional', 'aggressive', '[strategy]\ntype = "vol_expansion"\nrisk_profile = "aggressive"', 'Breakout de volatilidade com posições maiores.', 3, '{breakout,volatility}')
ON CONFLICT (slug) DO UPDATE SET name = EXCLUDED.name, description = EXCLUDED.description, tooltip_short = EXCLUDED.tooltip_short;

-- =============================================================================
-- SECTOR ROTATION STRATEGIES (4 strategies)
-- =============================================================================

INSERT INTO strategy_templates (slug, family_id, name, description, timeframe, bar_interval, position_type, risk_profile, config_toml, tooltip_short, difficulty_level, tags) VALUES
('sector_rotation_business_cycle', get_family_id('sector_rotation'), 'Business Cycle', 'Rotação por ciclo econômico', 'long_term', '1D', 'portfolio', 'moderate', '[strategy]\ntype = "business_cycle"', 'Aloca em setores que performam melhor em cada fase do ciclo econômico.', 4, '{sector,rotation,macro}'),
('sector_rotation_business_cycle_defensive', get_family_id('sector_rotation'), 'Business Cycle Defensivo', 'Rotação defensiva', 'long_term', '1D', 'portfolio', 'conservative', '[strategy]\ntype = "business_cycle_def"', 'Versão mais conservadora focando em setores defensivos.', 3, '{sector,rotation,defensive}'),
('sector_rotation_relative_strength_top3', get_family_id('sector_rotation'), 'Relative Strength Top 3', 'Top 3 setores por força relativa', 'position', '1D', 'portfolio', 'moderate', '[strategy]\ntype = "relative_strength"\ntop_n = 3', 'Investe nos 3 setores com melhor momentum recente.', 3, '{sector,rotation,momentum}'),
('sector_rotation_relative_strength_top5', get_family_id('sector_rotation'), 'Relative Strength Top 5', 'Top 5 setores por força relativa', 'position', '1D', 'portfolio', 'moderate', '[strategy]\ntype = "relative_strength"\ntop_n = 5', 'Investe nos 5 setores com melhor performance.', 3, '{sector,rotation,momentum}')
ON CONFLICT (slug) DO UPDATE SET name = EXCLUDED.name, description = EXCLUDED.description, tooltip_short = EXCLUDED.tooltip_short;

-- =============================================================================
-- FACTOR STRATEGIES (8 strategies)
-- =============================================================================

INSERT INTO strategy_templates (slug, family_id, name, description, timeframe, bar_interval, position_type, risk_profile, config_toml, tooltip_short, difficulty_level, tags) VALUES
('factor_value_pe_conservative', get_family_id('factor'), 'Value P/L Conservador', 'Fator valor por P/L', 'long_term', '1D', 'portfolio', 'conservative', '[strategy]\ntype = "value_pe"', 'Compra ações com baixo P/L (preço/lucro). Conservador.', 3, '{factor,value}'),
('factor_value_pb_moderate', get_family_id('factor'), 'Value P/VP Moderado', 'Fator valor por P/VP', 'long_term', '1D', 'portfolio', 'moderate', '[strategy]\ntype = "value_pb"', 'Compra ações com baixo P/VP (preço/valor patrimonial).', 3, '{factor,value}'),
('factor_quality_roe_conservative', get_family_id('factor'), 'Quality ROE Conservador', 'Fator qualidade por ROE', 'long_term', '1D', 'portfolio', 'conservative', '[strategy]\ntype = "quality_roe"', 'Compra ações com alto ROE (retorno sobre patrimônio).', 3, '{factor,quality}'),
('factor_quality_multi_metric_moderate', get_family_id('factor'), 'Quality Multi-Metric', 'Qualidade multi-indicador', 'long_term', '1D', 'portfolio', 'moderate', '[strategy]\ntype = "quality_multi"', 'Combina ROE, margem e baixa dívida para selecionar.', 4, '{factor,quality}'),
('factor_low_volatility_conservative', get_family_id('factor'), 'Low Volatility Conservador', 'Fator baixa volatilidade', 'long_term', '1D', 'portfolio', 'conservative', '[strategy]\ntype = "low_vol"', 'Compra ações que historicamente oscilam menos. Conservador.', 2, '{factor,low_vol}'),
('factor_low_volatility_moderate', get_family_id('factor'), 'Low Volatility Moderado', 'Baixa volatilidade moderado', 'long_term', '1D', 'portfolio', 'moderate', '[strategy]\ntype = "low_vol"\nrisk_profile = "moderate"', 'Fator low-vol com um pouco mais de concentração.', 2, '{factor,low_vol}'),
('factor_multi_factor_balanced', get_family_id('factor'), 'Multi-Factor Balanced', 'Multi-fator balanceado', 'long_term', '1D', 'portfolio', 'moderate', '[strategy]\ntype = "multi_factor"', 'Combina valor, qualidade, momentum e low-vol.', 4, '{factor,multi_factor}'),
('factor_multi_factor_aggressive', get_family_id('factor'), 'Multi-Factor Agressivo', 'Multi-fator agressivo', 'long_term', '1D', 'portfolio', 'aggressive', '[strategy]\ntype = "multi_factor"\nrisk_profile = "aggressive"', 'Multi-factor com maior concentração nos top picks.', 4, '{factor,multi_factor}')
ON CONFLICT (slug) DO UPDATE SET name = EXCLUDED.name, description = EXCLUDED.description, tooltip_short = EXCLUDED.tooltip_short;

-- =============================================================================
-- SEASONAL STRATEGIES (4 strategies)
-- =============================================================================

INSERT INTO strategy_templates (slug, family_id, name, description, timeframe, bar_interval, position_type, risk_profile, config_toml, tooltip_short, difficulty_level, tags) VALUES
('seasonal_calendar_effects_january', get_family_id('seasonal'), 'January Effect', 'Efeito Janeiro', 'long_term', '1D', 'directional', 'moderate', '[strategy]\ntype = "january_effect"', 'Compra small caps em dezembro esperando alta em janeiro.', 3, '{seasonal,calendar}'),
('seasonal_calendar_effects_sell_in_may', get_family_id('seasonal'), 'Sell in May', 'Venda em Maio', 'long_term', '1D', 'directional', 'moderate', '[strategy]\ntype = "sell_in_may"', 'Investe Nov-Abr, fica em caixa Mai-Out. Padrão histórico.', 2, '{seasonal,calendar}'),
('seasonal_commodity_natural_gas', get_family_id('seasonal'), 'Natural Gas Seasonal', 'Sazonalidade gás natural', 'position', '1D', 'directional', 'moderate', '[strategy]\ntype = "natgas_seasonal"', 'Opera padrões sazonais de demanda de gás natural.', 4, '{seasonal,commodity}'),
('seasonal_commodity_grains', get_family_id('seasonal'), 'Grains Seasonal', 'Sazonalidade grãos', 'position', '1D', 'directional', 'moderate', '[strategy]\ntype = "grains_seasonal"', 'Opera padrões sazonais de plantio e colheita.', 4, '{seasonal,commodity}')
ON CONFLICT (slug) DO UPDATE SET name = EXCLUDED.name, description = EXCLUDED.description, tooltip_short = EXCLUDED.tooltip_short;

-- =============================================================================
-- VOLATILITY STRATEGIES (4 strategies)
-- =============================================================================

INSERT INTO strategy_templates (slug, family_id, name, description, timeframe, bar_interval, position_type, risk_profile, config_toml, tooltip_short, difficulty_level, tags) VALUES
('volatility_vix_mean_reversion_conservative', get_family_id('volatility'), 'VIX Mean Reversion Conservador', 'Reversão de VIX conservador', 'swing', '1D', 'directional', 'conservative', '[strategy]\ntype = "vix_reversion"', 'Vende VIX alto, compra VIX baixo. Volatilidade reverte à média.', 4, '{volatility,vix,mean_reversion}'),
('volatility_vix_mean_reversion_moderate', get_family_id('volatility'), 'VIX Mean Reversion Moderado', 'Reversão de VIX moderado', 'swing', '1D', 'directional', 'moderate', '[strategy]\ntype = "vix_reversion"\nrisk_profile = "moderate"', 'Opera VIX com risco moderado.', 4, '{volatility,vix,mean_reversion}'),
('volatility_breakout_atr_moderate', get_family_id('volatility'), 'ATR Breakout', 'Breakout por ATR', 'swing', '1D', 'directional', 'moderate', '[strategy]\ntype = "atr_breakout"', 'Compra quando movimento excede múltiplo do ATR.', 3, '{volatility,atr,breakout}'),
('volatility_breakout_bb_width_moderate', get_family_id('volatility'), 'BB Width Breakout', 'Breakout por squeeze de Bollinger', 'swing', '1D', 'directional', 'moderate', '[strategy]\ntype = "bb_squeeze"', 'Entra quando Bollinger Bands apertam e depois expandem.', 3, '{volatility,bollinger,breakout}')
ON CONFLICT (slug) DO UPDATE SET name = EXCLUDED.name, description = EXCLUDED.description, tooltip_short = EXCLUDED.tooltip_short;

-- =============================================================================
-- EVENT-DRIVEN STRATEGIES (4 strategies)
-- =============================================================================

INSERT INTO strategy_templates (slug, family_id, name, description, timeframe, bar_interval, position_type, risk_profile, config_toml, tooltip_short, difficulty_level, tags) VALUES
('event_driven_earnings_pre_announcement', get_family_id('event_driven'), 'Pre-Earnings', 'Pré-balanço', 'swing', '1D', 'directional', 'moderate', '[strategy]\ntype = "pre_earnings"', 'Posiciona antes da divulgação de resultados esperando volatilidade.', 4, '{event,earnings}'),
('event_driven_earnings_post_surprise', get_family_id('event_driven'), 'Post-Earnings Drift', 'Drift pós-balanço', 'swing', '1D', 'directional', 'moderate', '[strategy]\ntype = "post_earnings"', 'Opera na direção da surpresa após divulgação de resultado.', 4, '{event,earnings}'),
('event_driven_ma_arbitrage_conservative', get_family_id('event_driven'), 'M&A Arbitrage Conservador', 'Arbitragem de M&A conservador', 'position', '1D', 'pair', 'conservative', '[strategy]\ntype = "ma_arb"', 'Compra alvo de aquisição anunciada e vende adquirente.', 5, '{event,merger,arbitrage}'),
('event_driven_ma_arbitrage_moderate', get_family_id('event_driven'), 'M&A Arbitrage Moderado', 'Arbitragem de M&A moderado', 'position', '1D', 'pair', 'moderate', '[strategy]\ntype = "ma_arb"\nrisk_profile = "moderate"', 'Arbitragem de fusões com risco moderado.', 5, '{event,merger,arbitrage}')
ON CONFLICT (slug) DO UPDATE SET name = EXCLUDED.name, description = EXCLUDED.description, tooltip_short = EXCLUDED.tooltip_short;

-- =============================================================================
-- BUY AND HOLD STRATEGIES (4 strategies)
-- =============================================================================

INSERT INTO strategy_templates (slug, family_id, name, description, timeframe, bar_interval, position_type, risk_profile, config_toml, tooltip_short, difficulty_level, tags) VALUES
('buy_hold_index_ibov', get_family_id('buy_hold'), 'Buy & Hold IBOV', 'Comprar e segurar índice IBOV', 'long_term', '1D', 'directional', 'moderate', '[strategy]\ntype = "buy_hold"\nindex = "IBOV"', 'Compra e mantém exposição ao Ibovespa. Benchmark Brasil.', 1, '{passive,index,brazil}'),
('buy_hold_index_sp500', get_family_id('buy_hold'), 'Buy & Hold S&P 500', 'Comprar e segurar S&P 500', 'long_term', '1D', 'directional', 'moderate', '[strategy]\ntype = "buy_hold"\nindex = "SPX"', 'Compra e mantém exposição ao S&P 500. Benchmark EUA.', 1, '{passive,index,usa}'),
('buy_hold_dividend_growth_conservative', get_family_id('buy_hold'), 'Dividend Growth Conservador', 'Crescimento de dividendos conservador', 'long_term', '1D', 'portfolio', 'conservative', '[strategy]\ntype = "dividend_growth"', 'Compra empresas com histórico de crescimento de dividendos.', 2, '{passive,dividend}'),
('buy_hold_dividend_growth_moderate', get_family_id('buy_hold'), 'Dividend Growth Moderado', 'Crescimento de dividendos moderado', 'long_term', '1D', 'portfolio', 'moderate', '[strategy]\ntype = "dividend_growth"\nrisk_profile = "moderate"', 'Dividend growth com um pouco mais de concentração.', 2, '{passive,dividend}')
ON CONFLICT (slug) DO UPDATE SET name = EXCLUDED.name, description = EXCLUDED.description, tooltip_short = EXCLUDED.tooltip_short;

-- =============================================================================
-- POPULATE CATALOGS
-- =============================================================================

-- Quick Test catalog (5 simples strategies)
INSERT INTO catalog_strategies (catalog_id, strategy_id, priority)
SELECT c.id, s.id, 1
FROM strategy_catalogs c, strategy_templates s
WHERE c.slug = 'quick_test' 
  AND s.slug IN ('swing_momentum_ma_crossover_moderate', 'mean_reversion_bb_moderate', 'breakout_donchian_20d', 'portfolio_equal_weight_10_assets', 'buy_hold_index_ibov')
ON CONFLICT DO NOTHING;

-- Institutional catalog (conservative/moderate strategies)
INSERT INTO catalog_strategies (catalog_id, strategy_id, priority)
SELECT c.id, s.id, 1
FROM strategy_catalogs c, strategy_templates s
WHERE c.slug = 'institutional' 
  AND s.risk_profile IN ('conservative', 'moderate')
ON CONFLICT DO NOTHING;

-- High Frequency catalog (intraday + swing)
INSERT INTO catalog_strategies (catalog_id, strategy_id, priority)
SELECT c.id, s.id, 1
FROM strategy_catalogs c, strategy_templates s
WHERE c.slug = 'high_frequency' 
  AND s.timeframe IN ('intraday', 'swing')
ON CONFLICT DO NOTHING;

-- Pairs Only catalog
INSERT INTO catalog_strategies (catalog_id, strategy_id, priority)
SELECT c.id, s.id, 1
FROM strategy_catalogs c, strategy_templates s
WHERE c.slug = 'pairs_only' 
  AND s.family_id = get_family_id('pair')
ON CONFLICT DO NOTHING;

-- All strategies catalog
INSERT INTO catalog_strategies (catalog_id, strategy_id, priority)
SELECT c.id, s.id, s.id
FROM strategy_catalogs c, strategy_templates s
WHERE c.slug = 'all'
ON CONFLICT DO NOTHING;

-- Cleanup helper function
DROP FUNCTION IF EXISTS get_family_id(VARCHAR);




