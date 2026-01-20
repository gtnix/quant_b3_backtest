/**
 * GlassPanel - Premium glassmorphism panel with optional neon glow
 */
import { motion, HTMLMotionProps } from 'framer-motion';
import { clsx } from 'clsx';

interface GlassPanelProps extends HTMLMotionProps<'div'> {
  children: React.ReactNode;
  glow?: 'none' | 'cyan' | 'green' | 'gold' | 'red';
  intensity?: 'low' | 'medium' | 'high';
  className?: string;
}

const glowColors = {
  none: '',
  cyan: 'shadow-[0_0_20px_rgba(0,212,255,0.15)] border-cyan-500/30',
  green: 'shadow-[0_0_20px_rgba(0,255,136,0.15)] border-emerald-500/30',
  gold: 'shadow-[0_0_20px_rgba(255,215,0,0.15)] border-amber-500/30',
  red: 'shadow-[0_0_20px_rgba(255,71,87,0.15)] border-red-500/30',
};

const intensityStyles = {
  low: 'backdrop-blur-sm bg-slate-900/40',
  medium: 'backdrop-blur-md bg-slate-900/60',
  high: 'backdrop-blur-lg bg-slate-900/80',
};

export function GlassPanel({
  children,
  glow = 'none',
  intensity = 'medium',
  className,
  ...motionProps
}: GlassPanelProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, ease: 'easeOut' }}
      className={clsx(
        'rounded-xl border',
        intensityStyles[intensity],
        glowColors[glow],
        glow === 'none' && 'border-slate-700/50',
        className
      )}
      {...motionProps}
    >
      {children}
    </motion.div>
  );
}

export default GlassPanel;
