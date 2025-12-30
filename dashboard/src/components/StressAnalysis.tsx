import { useState, useEffect } from 'react';
import { config } from '../lib/platform';
import { QuickTooltip } from './ui/TooltipInfo';
import { 
  RefreshCw,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Zap,
  TrendingDown,
  Shield,
  Activity
} from 'lucide-react';

interface StressScenario {
  scenario_id: string;
  scenario_name: string;
  description: string;
  base_sharpe: number;
  stressed_sharpe: number;
  degradation_pct: number;
  threshold: number;
  status: 'PASS' | 'FAIL';
  severity: 'low' | 'medium' | 'high';
}

interface StressData {
  candidate_id: string;
  stress_config: {
    min_sharpe_threshold: number;
    pass_ratio_required: number;
    scenarios_tested: number;
  };
  scenarios: StressScenario[];
  summary: {
    total_scenarios: number;
    passed: number;
    failed: number;
    pass_rate: number;
    overall_status: 'PASS' | 'FAIL';
    worst_scenario: string;
  };
}

interface Props {
  candidateId: string;
}

export function StressAnalysis({ candidateId }: Props) {
  const [data, setData] = useState<StressData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadStress();
  }, [candidateId]);

  const loadStress = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${config.apiBase}/candidate/${candidateId}/stress`);
      if (response.ok) {
        const result = await response.json();
        setData(result);
      } else {
        setError('Failed to load stress data');
      }
    } catch (err) {
      setError('Connection error');
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <RefreshCw className="w-8 h-8 animate-spin text-terminal-muted" />
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="flex items-center justify-center h-64 text-terminal-muted">
        {error || 'No stress data'}
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Summary Header */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
        <SummaryCard
          icon={<Shield className="w-5 h-5" />}
          label={<span className="inline-flex items-center">Stress Test<QuickTooltip termKey="stress_test" /></span>}
          value={data.summary.overall_status}
          passed={data.summary.overall_status === 'PASS'}
          isStatus
        />
        <SummaryCard
          icon={<CheckCircle className="w-5 h-5" />}
          label="Passed"
          value={`${data.summary.passed}`}
          passed={true}
          suffix={`/${data.summary.total_scenarios}`}
        />
        <SummaryCard
          icon={<XCircle className="w-5 h-5" />}
          label="Failed"
          value={`${data.summary.failed}`}
          passed={data.summary.failed === 0}
        />
        <SummaryCard
          icon={<Activity className="w-5 h-5" />}
          label="Pass Rate"
          value={`${data.summary.pass_rate}%`}
          passed={data.summary.pass_rate >= 62.5}
        />
        <SummaryCard
          icon={<TrendingDown className="w-5 h-5" />}
          label={<span className="inline-flex items-center">Worst<QuickTooltip termKey="stress_scenario" /></span>}
          value={data.summary.worst_scenario.split(' ')[0]}
          passed={false}
          small
        />
      </div>

      {/* Config Info */}
      <div className="flex items-center gap-6 text-xs text-terminal-muted">
        <span>Min Sharpe Threshold: {data.stress_config.min_sharpe_threshold}</span>
        <span>Required Pass Rate: {(data.stress_config.pass_ratio_required * 100).toFixed(1)}%</span>
        <span>Scenarios: {data.stress_config.scenarios_tested}</span>
      </div>

      {/* Scenarios Table */}
      <div className="rounded-xl border border-terminal-border overflow-hidden">
        <table className="w-full">
          <thead className="bg-terminal-surface">
            <tr>
              <th className="text-left text-xs font-medium text-terminal-muted uppercase tracking-wider px-4 py-3">Scenario</th>
              <th className="text-center text-xs font-medium text-terminal-muted uppercase tracking-wider px-4 py-3">Severity</th>
              <th className="text-right text-xs font-medium text-terminal-muted uppercase tracking-wider px-4 py-3"><span className="inline-flex items-center">Base Sharpe<QuickTooltip termKey="sharpe" /></span></th>
              <th className="text-right text-xs font-medium text-terminal-muted uppercase tracking-wider px-4 py-3">Stressed</th>
              <th className="text-right text-xs font-medium text-terminal-muted uppercase tracking-wider px-4 py-3"><span className="inline-flex items-center">Degradation<QuickTooltip termKey="stress_degradation" /></span></th>
              <th className="text-center text-xs font-medium text-terminal-muted uppercase tracking-wider px-4 py-3">Status</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-terminal-border">
            {data.scenarios.map((scenario) => (
              <ScenarioRow key={scenario.scenario_id} scenario={scenario} />
            ))}
          </tbody>
        </table>
      </div>

      {/* Sharpe Degradation Visual */}
      <div className="p-4 rounded-xl bg-terminal-surface border border-terminal-border">
        <h4 className="text-sm font-medium text-terminal-muted mb-4">Sharpe Impact by Scenario</h4>
        <div className="space-y-3">
          {data.scenarios.map((scenario) => (
            <div key={scenario.scenario_id} className="space-y-1">
              <div className="flex items-center justify-between text-xs">
                <span className="text-terminal-muted truncate max-w-[200px]">{scenario.scenario_name}</span>
                <span className={scenario.status === 'PASS' ? 'text-profit' : 'text-loss'}>
                  {scenario.stressed_sharpe.toFixed(2)}
                </span>
              </div>
              <div className="h-2 bg-terminal-bg rounded-full overflow-hidden">
                <div 
                  className={`h-full rounded-full transition-all ${
                    scenario.status === 'PASS' ? 'bg-profit' : 'bg-loss'
                  }`}
                  style={{ 
                    width: `${Math.min(100, (scenario.stressed_sharpe / scenario.base_sharpe) * 100)}%` 
                  }}
                />
              </div>
            </div>
          ))}
        </div>
        <div className="mt-4 pt-4 border-t border-terminal-border flex items-center justify-center gap-2 text-xs text-terminal-muted">
          <span>Threshold: Sharpe &gt; {data.stress_config.min_sharpe_threshold}</span>
        </div>
      </div>

      {/* Risk Matrix */}
      <div className="grid grid-cols-3 gap-4">
        <RiskCategory
          title="Low Severity"
          scenarios={data.scenarios.filter(s => s.severity === 'low')}
          color="profit"
        />
        <RiskCategory
          title="Medium Severity"
          scenarios={data.scenarios.filter(s => s.severity === 'medium')}
          color="warning"
        />
        <RiskCategory
          title="High Severity"
          scenarios={data.scenarios.filter(s => s.severity === 'high')}
          color="loss"
        />
      </div>
    </div>
  );
}

function SummaryCard({ 
  icon, 
  label, 
  value, 
  passed, 
  isStatus,
  suffix,
  small 
}: { 
  icon: React.ReactNode;
  label: React.ReactNode; 
  value: string; 
  passed: boolean; 
  isStatus?: boolean;
  suffix?: string;
  small?: boolean;
}) {
  return (
    <div className={`p-4 rounded-xl border ${passed ? 'bg-profit/5 border-profit/30' : 'bg-loss/5 border-loss/30'}`}>
      <div className="flex items-center gap-2 mb-2">
        <span className={passed ? 'text-profit' : 'text-loss'}>{icon}</span>
        <span className="text-xs text-terminal-muted uppercase tracking-wider">{label}</span>
      </div>
      <div className="flex items-baseline gap-1">
        <span className={`font-bold font-mono ${passed ? 'text-profit' : 'text-loss'} ${small ? 'text-sm' : 'text-xl'}`}>
          {value}
        </span>
        {suffix && <span className="text-xs text-terminal-muted">{suffix}</span>}
      </div>
    </div>
  );
}

function ScenarioRow({ scenario }: { scenario: StressScenario }) {
  const severityColors = {
    low: 'bg-profit/20 text-profit',
    medium: 'bg-accent-yellow/20 text-accent-yellow',
    high: 'bg-loss/20 text-loss'
  };

  return (
    <tr className="hover:bg-terminal-surface/50 transition-colors">
      <td className="px-4 py-3">
        <div className="font-medium text-sm">{scenario.scenario_name}</div>
        <div className="text-xs text-terminal-muted">{scenario.description}</div>
      </td>
      <td className="px-4 py-3 text-center">
        <span className={`text-xs font-medium px-2 py-1 rounded ${severityColors[scenario.severity]}`}>
          {scenario.severity.toUpperCase()}
        </span>
      </td>
      <td className="px-4 py-3 text-right font-mono text-sm">
        {scenario.base_sharpe.toFixed(3)}
      </td>
      <td className="px-4 py-3 text-right font-mono text-sm">
        <span className={scenario.status === 'PASS' ? 'text-profit' : 'text-loss'}>
          {scenario.stressed_sharpe.toFixed(3)}
        </span>
      </td>
      <td className="px-4 py-3 text-right font-mono text-sm text-loss">
        -{scenario.degradation_pct}%
      </td>
      <td className="px-4 py-3 text-center">
        {scenario.status === 'PASS' ? (
          <CheckCircle className="w-5 h-5 text-profit inline" />
        ) : (
          <XCircle className="w-5 h-5 text-loss inline" />
        )}
      </td>
    </tr>
  );
}

function RiskCategory({ 
  title, 
  scenarios, 
  color 
}: { 
  title: string; 
  scenarios: StressScenario[]; 
  color: 'profit' | 'warning' | 'loss'; 
}) {
  const colorClasses = {
    profit: 'border-profit/30 bg-profit/5',
    warning: 'border-accent-yellow/30 bg-accent-yellow/5',
    loss: 'border-loss/30 bg-loss/5'
  };

  const textColors = {
    profit: 'text-profit',
    warning: 'text-accent-yellow',
    loss: 'text-loss'
  };

  const passed = scenarios.filter(s => s.status === 'PASS').length;
  const total = scenarios.length;

  return (
    <div className={`p-4 rounded-xl border ${colorClasses[color]}`}>
      <div className="flex items-center justify-between mb-3">
        <h4 className={`text-sm font-medium ${textColors[color]}`}>{title}</h4>
        <span className="text-xs text-terminal-muted">
          {passed}/{total} passed
        </span>
      </div>
      <div className="space-y-2">
        {scenarios.map(s => (
          <div key={s.scenario_id} className="flex items-center justify-between text-xs">
            <span className="text-terminal-muted truncate max-w-[120px]">{s.scenario_name.split(' ')[0]}</span>
            {s.status === 'PASS' ? (
              <CheckCircle className="w-3 h-3 text-profit" />
            ) : (
              <XCircle className="w-3 h-3 text-loss" />
            )}
          </div>
        ))}
        {scenarios.length === 0 && (
          <div className="text-xs text-terminal-muted text-center py-2">No scenarios</div>
        )}
      </div>
    </div>
  );
}

