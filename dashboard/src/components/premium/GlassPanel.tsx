/**
 * GlassPanel - Ultra-minimal transparent panel with subtle glow
 * Designed to blend with grid background
 */
import { motion } from 'framer-motion';
import { clsx } from 'clsx';
import { ReactNode } from 'react';

interface GlassPanelProps {
  children: ReactNode;
  glow?: 'none' | 'cyan' | 'green' | 'gold' | 'red';
  intensity?: 'low' | 'medium' | 'high';
  className?: string;
}

const glowStyles = {
  none: {
    border: 'border-slate-700/20',
    shadow: '',
    bg: 'bg-transparent',
  },
  cyan: {
    border: 'border-cyan-500/20',
    shadow: 'shadow-[0_0_30px_rgba(0,212,255,0.08)]',
    bg: 'bg-cyan-950/10',
  },
  green: {
    border: 'border-emerald-500/20',
    shadow: 'shadow-[0_0_30px_rgba(0,255,136,0.08)]',
    bg: 'bg-emerald-950/10',
  },
  gold: {
    border: 'border-amber-500/25',
    shadow: 'shadow-[0_0_30px_rgba(255,215,0,0.1)]',
    bg: 'bg-amber-950/10',
  },
  red: {
    border: 'border-red-500/20',
    shadow: 'shadow-[0_0_30px_rgba(255,71,87,0.08)]',
    bg: 'bg-red-950/10',
  },
};

export function GlassPanel({
  children,
  glow = 'none',
  intensity = 'medium',
  className,
}: GlassPanelProps) {
  const style = glowStyles[glow];
  
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, ease: 'easeOut' }}
      className={clsx(
        'rounded-xl border backdrop-blur-[2px]',
        style.border,
        style.shadow,
        style.bg,
        className
      )}
    >
      {children}
    </motion.div>
  );
}

export default GlassPanel;
