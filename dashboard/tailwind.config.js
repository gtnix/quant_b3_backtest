/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Terminal NYC Dark Theme
        terminal: {
          bg: '#0a0a0f',
          surface: '#12121a',
          border: '#1e1e2e',
          muted: '#3a3a4a',
        },
        profit: {
          DEFAULT: '#00ff88',
          dim: '#00cc6a',
          glow: '#00ff8840',
        },
        loss: {
          DEFAULT: '#ff3366',
          dim: '#cc2952',
          glow: '#ff336640',
        },
        accent: {
          cyan: '#00d4ff',
          purple: '#8b5cf6',
          orange: '#ff9500',
          yellow: '#ffd700',
        },
      },
      fontFamily: {
        mono: ['JetBrains Mono', 'IBM Plex Mono', 'monospace'],
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'glow': 'glow 2s ease-in-out infinite alternate',
        'ticker': 'ticker 0.3s ease-out',
      },
      keyframes: {
        glow: {
          '0%': { boxShadow: '0 0 5px var(--glow-color)' },
          '100%': { boxShadow: '0 0 20px var(--glow-color), 0 0 40px var(--glow-color)' },
        },
        ticker: {
          '0%': { transform: 'translateY(-10px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
      },
      backgroundImage: {
        'grid-pattern': 'linear-gradient(to right, #1e1e2e 1px, transparent 1px), linear-gradient(to bottom, #1e1e2e 1px, transparent 1px)',
      },
    },
  },
  plugins: [],
}

