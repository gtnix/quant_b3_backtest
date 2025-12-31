import { useEffect, useRef } from 'react';
import { createChart, ColorType, LineStyle } from 'lightweight-charts';

interface GenerationChartProps {
  data: Array<{
    generation: number;
    bestSharpe: number;
    meanSharpe: number;
    paretoSize: number;
  }>;
}

export function GenerationChart({ data }: GenerationChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!chartContainerRef.current) return;

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
          color: '#00d4ff',
          width: 1,
          style: LineStyle.Dashed,
          labelBackgroundColor: '#12121a',
        },
        horzLine: {
          color: '#00d4ff',
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
      },
      timeScale: {
        borderColor: '#1e1e2e',
        tickMarkFormatter: (time: number) => `G${time}`,
      },
    });

    // Best Sharpe line
    const bestSharpeSeries = chart.addLineSeries({
      color: '#00ff88',
      lineWidth: 2,
      title: 'Best Sharpe',
      priceFormat: {
        type: 'custom',
        formatter: (price: number) => price.toFixed(3),
      },
    });

    // Mean Sharpe line
    const meanSharpeSeries = chart.addLineSeries({
      color: '#8b5cf6',
      lineWidth: 2,
      lineStyle: LineStyle.Dashed,
      title: 'Mean Sharpe',
      priceFormat: {
        type: 'custom',
        formatter: (price: number) => price.toFixed(3),
      },
    });

    // Format data for lightweight-charts (use generation as time index)
    const bestData = data.map(d => ({
      time: d.generation as unknown as string,
      value: d.bestSharpe,
    }));

    const meanData = data.map(d => ({
      time: d.generation as unknown as string,
      value: d.meanSharpe,
    }));

    bestSharpeSeries.setData(bestData);
    meanSharpeSeries.setData(meanData);
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
  }, [data]);

  return (
    <div className="relative w-full h-full">
      <div ref={chartContainerRef} className="w-full h-full" />
      {/* Legend */}
      <div className="absolute top-2 left-2 flex items-center gap-4 text-xs">
        <div className="flex items-center gap-1">
          <div className="w-3 h-0.5 bg-profit" />
          <span className="text-terminal-muted">Best Sharpe</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-0.5 bg-accent-purple" style={{ borderStyle: 'dashed' }} />
          <span className="text-terminal-muted">Mean Sharpe</span>
        </div>
      </div>
    </div>
  );
}









