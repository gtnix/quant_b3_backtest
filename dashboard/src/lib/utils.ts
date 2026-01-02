import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

/**
 * Merge Tailwind classes with clsx
 */
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/**
 * Format a number as percentage
 */
export function formatPercent(value: number, decimals = 2): string {
  return `${(value * 100).toFixed(decimals)}%`;
}

/**
 * Format a number as currency
 */
export function formatCurrency(value: number, currency = 'USD'): string {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency,
    minimumFractionDigits: 0,
    maximumFractionDigits: 0,
  }).format(value);
}

/**
 * Format a number with specified decimals
 */
export function formatNumber(value: number, decimals = 2): string {
  return value.toFixed(decimals);
}

/**
 * Format a ratio (e.g., Sharpe)
 */
export function formatRatio(value: number, decimals = 3): string {
  return value.toFixed(decimals);
}

/**
 * Get color class based on value sign
 */
export function getValueColor(value: number): string {
  if (value > 0) return 'text-profit';
  if (value < 0) return 'text-loss';
  return 'text-terminal-muted';
}

/**
 * Format duration in seconds to human readable
 */
export function formatDuration(seconds: number): string {
  if (seconds < 60) return `${seconds}s`;
  if (seconds < 3600) {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}m ${secs}s`;
  }
  const hours = Math.floor(seconds / 3600);
  const mins = Math.floor((seconds % 3600) / 60);
  return `${hours}h ${mins}m`;
}

/**
 * Format timestamp to relative time
 */
export function formatRelativeTime(timestamp: string): string {
  const date = new Date(timestamp);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffMins = Math.floor(diffMs / 60000);
  
  if (diffMins < 1) return 'just now';
  if (diffMins < 60) return `${diffMins} min ago`;
  
  const diffHours = Math.floor(diffMins / 60);
  if (diffHours < 24) return `${diffHours}h ago`;
  
  const diffDays = Math.floor(diffHours / 24);
  return `${diffDays}d ago`;
}

/**
 * Truncate string with ellipsis
 */
export function truncate(str: string, maxLength: number): string {
  if (str.length <= maxLength) return str;
  return str.slice(0, maxLength - 3) + '...';
}

/**
 * Generate gradient color between two colors based on value
 */
export function getGradientColor(value: number, min: number, max: number): string {
  const ratio = Math.max(0, Math.min(1, (value - min) / (max - min)));
  
  // Green to Yellow to Red
  if (ratio < 0.5) {
    // Yellow to Red
    const r = 255;
    const g = Math.floor(255 * ratio * 2);
    return `rgb(${r}, ${g}, 0)`;
  } else {
    // Green to Yellow
    const r = Math.floor(255 * (1 - ratio) * 2);
    const g = 255;
    return `rgb(${r}, ${g}, 0)`;
  }
}

/**
 * Debounce function
 */
export function debounce<T extends (...args: Parameters<T>) => ReturnType<T>>(
  func: T,
  wait: number
): (...args: Parameters<T>) => void {
  let timeout: ReturnType<typeof setTimeout> | null = null;
  
  return (...args: Parameters<T>) => {
    if (timeout) clearTimeout(timeout);
    timeout = setTimeout(() => func(...args), wait);
  };
}














