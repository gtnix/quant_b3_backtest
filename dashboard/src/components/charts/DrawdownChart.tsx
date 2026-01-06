import { useEffect, useRef } from 'react';
import { createChart, ColorType, LineStyle } from 'lightweight-charts';

interface DrawdownChartProps {
  data: Array<{
    time: string;
    value: number;
  }>;
}

export function DrawdownChart({ data }: DrawdownChartProps) {
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
          color: '#ff3366',
          width: 1,
          style: LineStyle.Dashed,
          labelBackgroundColor: '#12121a',
        },
        horzLine: {
          color: '#ff3366',
          width: 1,
          style: LineStyle.Dashed,
          labelBackgroundColor: '#12121a',
        },
      },
      rightPriceScale: {
        borderColor: '#1e1e2e',
        scaleMargins: {
          top: 0.05,
          bottom: 0.1,
        },
      },
      timeScale: {
        borderColor: '#1e1e2e',
        timeVisible: true,
      },
    });

    const areaSeries = chart.addAreaSeries({
      lineColor: '#ff3366',
      topColor: 'rgba(255, 51, 102, 0.0)',
      bottomColor: 'rgba(255, 51, 102, 0.4)',
      lineWidth: 2,
      priceFormat: {
        type: 'custom',
        formatter: (price: number) => (price * 100).toFixed(2) + '%',
      },
    });

    // Add zero line
    const zeroLine = chart.addLineSeries({
      color: '#3a3a4a',
      lineWidth: 1,
      lineStyle: LineStyle.Dashed,
      priceLineVisible: false,
      lastValueVisible: false,
    });

    const zeroData = data.map(d => ({
      time: d.time,
      value: 0,
    }));

    areaSeries.setData(data);
    zeroLine.setData(zeroData);
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

  return <div ref={chartContainerRef} className="w-full h-full" />;
}























