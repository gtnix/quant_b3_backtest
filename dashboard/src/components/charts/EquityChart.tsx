import { useEffect, useRef, useMemo } from 'react';
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

  // Generate benchmark data (CDI cumulative return)
  const benchmarkData = useMemo(() => {
    if (!showBenchmark || validData.length === 0) return [];
    
    const startValue = validData[0].value;
    const dailyRate = Math.pow(1 + benchmarkRate, 1/252) - 1;
    
    return validData.map((point, i) => ({
      time: point.time,
      value: startValue * Math.pow(1 + dailyRate, i)
    }));
  }, [validData, showBenchmark, benchmarkRate]);

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
          top: 0.1,
          bottom: 0.1,
        },
        mode: logScale ? 1 : 0, // 1 = logarithmic, 0 = normal
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

    // Strategy equity curve
    const strategySeries = chart.addAreaSeries({
      lineColor: '#00ff88',
      topColor: 'rgba(0, 255, 136, 0.3)',
      bottomColor: 'rgba(0, 255, 136, 0.0)',
      lineWidth: 2,
      priceFormat: {
        type: 'custom',
        formatter: (price: number) => '$' + price.toLocaleString(undefined, { maximumFractionDigits: 0 }),
      },
      title: 'Strategy',
    });

    // Benchmark (CDI) line
    if (showBenchmark && benchmarkData.length > 0) {
      const benchmarkSeries = chart.addLineSeries({
        color: '#71717a',
        lineWidth: 1,
        lineStyle: LineStyle.Dashed,
        priceFormat: {
          type: 'custom',
          formatter: (price: number) => '$' + price.toLocaleString(undefined, { maximumFractionDigits: 0 }),
        },
        title: `CDI (${(benchmarkRate * 100).toFixed(1)}% a.a.)`,
      });
      benchmarkSeries.setData(benchmarkData);
    }

    strategySeries.setData(validData);
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
  }, [validData, logScale, showBenchmark, benchmarkData, benchmarkRate]);

  if (validData.length === 0) {
    return (
      <div className="w-full h-full flex items-center justify-center text-terminal-muted">
        No data available
      </div>
    );
  }

  return <div ref={chartContainerRef} className="w-full h-full" />;
}
