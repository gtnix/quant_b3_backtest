import { useEffect, useRef, useMemo, useState } from 'react';
import { createChart, ColorType, LineStyle } from 'lightweight-charts';

interface EquityChartProps {
  data: Array<{
    time: string;
    value: number;
  }>;
  logScale?: boolean;
  showBenchmark?: boolean;
  benchmarkRate?: number; // Annual CDI rate (e.g., 0.1075 for 10.75%)
}

export function EquityChart({ 
  data, 
  logScale = false, 
  showBenchmark = true,
  benchmarkRate = 0.1075 // CDI rate
}: EquityChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<ReturnType<typeof createChart> | null>(null);
  const [normalized, setNormalized] = useState(true); // Normalize to 100 for fair comparison

  // Filter out invalid data points (null/undefined values)
  const validData = useMemo(() => {
    return data.filter(point => 
      point && 
      point.time && 
      typeof point.value === 'number' && 
      !isNaN(point.value) &&
      isFinite(point.value)
    );
  }, [data]);

  // Normalize data to base 100 for fair comparison
  const normalizedData = useMemo(() => {
    if (!normalized || validData.length === 0) return validData;
    const startValue = validData[0].value;
    return validData.map(point => ({
      time: point.time,
      value: (point.value / startValue) * 100
    }));
  }, [validData, normalized]);

  // Generate benchmark data (CDI cumulative return)
  const benchmarkData = useMemo(() => {
    if (!showBenchmark || validData.length === 0) return [];
    
    const dailyRate = Math.pow(1 + benchmarkRate, 1/252) - 1;
    const startValue = normalized ? 100 : validData[0].value;
    
    return validData.map((point, i) => ({
      time: point.time,
      value: startValue * Math.pow(1 + dailyRate, i)
    }));
  }, [validData, showBenchmark, benchmarkRate, normalized]);

  // Compute final values for legend
  const finalStrategyValue = useMemo(() => {
    if (validData.length === 0) return 0;
    return validData[validData.length - 1].value;
  }, [validData]);

  const finalCDIValue = useMemo(() => {
    if (benchmarkData.length === 0) return 0;
    return benchmarkData[benchmarkData.length - 1].value;
  }, [benchmarkData]);

  const strategyReturn = useMemo(() => {
    if (validData.length < 2) return 0;
    return ((validData[validData.length - 1].value / validData[0].value) - 1) * 100;
  }, [validData]);

  const cdiReturn = useMemo(() => {
    if (benchmarkData.length < 2) return 0;
    return ((benchmarkData[benchmarkData.length - 1].value / benchmarkData[0].value) - 1) * 100;
  }, [benchmarkData]);

  useEffect(() => {
    if (!chartContainerRef.current || validData.length === 0) return;

    const chart = createChart(chartContainerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: 'transparent' },
        textColor: '#71717a',
        fontFamily: 'JetBrains Mono, monospace',
      },
      grid: {
        vertLines: { color: '#1e1e2e', style: LineStyle.Dotted },
        horzLines: { color: '#1e1e2e', style: LineStyle.Dotted },
      },
      crosshair: {
        vertLine: {
          color: '#00ff88',
          width: 1,
          style: LineStyle.Dashed,
          labelBackgroundColor: '#12121a',
        },
        horzLine: {
          color: '#00ff88',
          width: 1,
          style: LineStyle.Dashed,
          labelBackgroundColor: '#12121a',
        },
      },
      rightPriceScale: {
        borderColor: '#1e1e2e',
        scaleMargins: {
          top: 0.15,
          bottom: 0.15,
        },
        mode: logScale ? 1 : 0, // 1 = logarithmic, 0 = normal
        minimumWidth: 80, // Ensure enough space for labels
      },
      timeScale: {
        borderColor: '#1e1e2e',
        timeVisible: true,
      },
      handleScale: {
        axisPressedMouseMove: true,
      },
      handleScroll: {
        vertTouchDrag: true,
      },
    });

    chartRef.current = chart;

    // Benchmark (CDI) line - draw first so it's behind
    if (showBenchmark && benchmarkData.length > 0) {
      const benchmarkSeries = chart.addLineSeries({
        color: '#fbbf24',
        lineWidth: 2,
        lineStyle: LineStyle.Solid,
        priceFormat: {
          type: 'custom',
          formatter: (price: number) => normalized ? price.toFixed(0) : '$' + price.toLocaleString(undefined, { maximumFractionDigits: 0 }),
        },
        title: `CDI`,
      });
      benchmarkSeries.setData(benchmarkData);
    }

    // Strategy equity curve
    const strategySeries = chart.addAreaSeries({
      lineColor: '#00ff88',
      topColor: 'rgba(0, 255, 136, 0.3)',
      bottomColor: 'rgba(0, 255, 136, 0.0)',
      lineWidth: 2,
      priceFormat: {
        type: 'custom',
        formatter: (price: number) => normalized ? price.toFixed(0) : '$' + price.toLocaleString(undefined, { maximumFractionDigits: 0 }),
      },
      title: 'Strategy',
    });

    strategySeries.setData(normalized ? normalizedData : validData);
    chart.timeScale().fitContent();

    const handleResize = () => {
      if (chartContainerRef.current) {
        chart.applyOptions({
          width: chartContainerRef.current.clientWidth,
          height: chartContainerRef.current.clientHeight,
        });
      }
    };

    window.addEventListener('resize', handleResize);
    handleResize();

    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
    };
  }, [validData, normalizedData, logScale, showBenchmark, benchmarkData, benchmarkRate, normalized]);

  if (validData.length === 0) {
    return (
      <div className="w-full h-full flex items-center justify-center text-terminal-muted">
        No data available
      </div>
    );
  }

  return (
    <div className="w-full h-full relative">
      <div ref={chartContainerRef} className="w-full h-full" />
      
      {/* Legend with actual returns */}
      <div className="absolute bottom-2 left-2 flex items-center gap-4 bg-terminal-bg/90 backdrop-blur-sm border border-terminal-border rounded-lg px-3 py-2 z-10">
        <div className="flex items-center gap-2">
          <div className="w-3 h-0.5 bg-profit rounded" />
          <span className="text-xs text-terminal-muted">Strategy</span>
          <span className="text-xs font-mono text-profit">+{strategyReturn.toFixed(1)}%</span>
        </div>
        {showBenchmark && (
          <div className="flex items-center gap-2">
            <div className="w-3 h-0.5 bg-amber-400 rounded" />
            <span className="text-xs text-terminal-muted">CDI ({(benchmarkRate * 100).toFixed(1)}% a.a.)</span>
            <span className="text-xs font-mono text-amber-400">+{cdiReturn.toFixed(1)}%</span>
          </div>
        )}
      </div>
    </div>
  );
}
