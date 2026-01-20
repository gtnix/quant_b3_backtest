/**
 * CountUp - Animated number counter with formatting
 */
import { useEffect, useState, useRef } from 'react';
import { motion, useSpring, useTransform } from 'framer-motion';

interface CountUpProps {
  value: number;
  duration?: number;
  decimals?: number;
  prefix?: string;
  suffix?: string;
  className?: string;
  colorize?: boolean; // Green for positive, red for negative
}

export function CountUp({
  value,
  duration = 1,
  decimals = 2,
  prefix = '',
  suffix = '',
  className = '',
  colorize = false,
}: CountUpProps) {
  const spring = useSpring(0, { duration: duration * 1000, bounce: 0 });
  const display = useTransform(spring, (current) => {
    const formatted = current.toFixed(decimals);
    return `${prefix}${formatted}${suffix}`;
  });
  
  const [displayValue, setDisplayValue] = useState(`${prefix}${(0).toFixed(decimals)}${suffix}`);
  
  useEffect(() => {
    spring.set(value);
    const unsubscribe = display.on('change', (v) => setDisplayValue(v));
    return () => unsubscribe();
  }, [value, spring, display]);
  
  const colorClass = colorize
    ? value >= 0
      ? 'text-emerald-400'
      : 'text-red-400'
    : '';
  
  return (
    <motion.span
      className={`${className} ${colorClass} tabular-nums`}
      initial={{ opacity: 0, scale: 0.8 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.2 }}
    >
      {displayValue}
    </motion.span>
  );
}

/**
 * CountUpInteger - For whole numbers
 */
export function CountUpInteger({
  value,
  duration = 0.8,
  prefix = '',
  suffix = '',
  className = '',
}: Omit<CountUpProps, 'decimals'>) {
  return (
    <CountUp
      value={value}
      duration={duration}
      decimals={0}
      prefix={prefix}
      suffix={suffix}
      className={className}
    />
  );
}

export default CountUp;
