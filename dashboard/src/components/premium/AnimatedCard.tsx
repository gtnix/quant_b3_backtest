/**
 * AnimatedCard - Card with hover animations and optional glow
 */
import { motion } from 'framer-motion';
import { clsx } from 'clsx';

interface AnimatedCardProps {
  children: React.ReactNode;
  onClick?: () => void;
  selected?: boolean;
  variant?: 'default' | 'elite' | 'success' | 'warning';
  className?: string;
}

const variantStyles = {
  default: {
    base: 'border-slate-700/50 hover:border-slate-600',
    glow: '',
    selected: 'border-cyan-500/50 bg-cyan-500/5',
  },
  elite: {
    base: 'border-amber-500/30 hover:border-amber-400/50',
    glow: 'hover:shadow-[0_0_30px_rgba(255,215,0,0.1)]',
    selected: 'border-amber-400 bg-amber-500/10 shadow-[0_0_30px_rgba(255,215,0,0.15)]',
  },
  success: {
    base: 'border-emerald-500/30 hover:border-emerald-400/50',
    glow: 'hover:shadow-[0_0_30px_rgba(0,255,136,0.1)]',
    selected: 'border-emerald-400 bg-emerald-500/10',
  },
  warning: {
    base: 'border-amber-500/30 hover:border-amber-400/50',
    glow: 'hover:shadow-[0_0_30px_rgba(255,165,0,0.1)]',
    selected: 'border-amber-400 bg-amber-500/10',
  },
};

export function AnimatedCard({
  children,
  onClick,
  selected = false,
  variant = 'default',
  className,
}: AnimatedCardProps) {
  const styles = variantStyles[variant];
  
  return (
    <motion.div
      whileHover={{ scale: 1.01, y: -2 }}
      whileTap={{ scale: 0.99 }}
      transition={{ type: 'spring', stiffness: 400, damping: 25 }}
      onClick={onClick}
      className={clsx(
        'rounded-xl border bg-slate-800/50 backdrop-blur-sm p-4 cursor-pointer transition-all duration-200',
        selected ? styles.selected : styles.base,
        styles.glow,
        className
      )}
    >
      {children}
    </motion.div>
  );
}

export default AnimatedCard;
