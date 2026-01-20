/**
 * PulseIndicator - Animated status indicator with pulse effect
 */
import { motion } from 'framer-motion';
import { clsx } from 'clsx';

interface PulseIndicatorProps {
  status: 'online' | 'offline' | 'running' | 'warning' | 'idle';
  size?: 'sm' | 'md' | 'lg';
  label?: string;
  className?: string;
}

const statusColors = {
  online: 'bg-emerald-500',
  offline: 'bg-slate-500',
  running: 'bg-cyan-500',
  warning: 'bg-amber-500',
  idle: 'bg-slate-400',
};

const pulseColors = {
  online: 'bg-emerald-400',
  offline: 'bg-slate-400',
  running: 'bg-cyan-400',
  warning: 'bg-amber-400',
  idle: 'bg-slate-300',
};

const sizes = {
  sm: 'w-2 h-2',
  md: 'w-3 h-3',
  lg: 'w-4 h-4',
};

export function PulseIndicator({
  status,
  size = 'md',
  label,
  className,
}: PulseIndicatorProps) {
  const shouldPulse = status === 'running' || status === 'online';
  
  return (
    <div className={clsx('flex items-center gap-2', className)}>
      <div className="relative">
        {/* Pulse ring */}
        {shouldPulse && (
          <motion.div
            className={clsx(
              'absolute inset-0 rounded-full',
              pulseColors[status]
            )}
            animate={{
              scale: [1, 1.8, 1.8],
              opacity: [0.6, 0, 0],
            }}
            transition={{
              duration: 1.5,
              repeat: Infinity,
              ease: 'easeOut',
            }}
          />
        )}
        {/* Core dot */}
        <div
          className={clsx(
            'rounded-full relative',
            sizes[size],
            statusColors[status]
          )}
        />
      </div>
      {label && (
        <span className="text-sm text-slate-400">{label}</span>
      )}
    </div>
  );
}

export default PulseIndicator;
