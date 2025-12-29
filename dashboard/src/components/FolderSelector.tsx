/**
 * FolderSelector Component
 * 
 * Unified folder selection that works in both Tauri and Browser modes.
 * - Tauri: Uses native file dialog
 * - Browser: Text input with path validation
 */

import { useState, useEffect } from 'react';
import { platform, capabilities } from '../lib/platform';
import { cmd } from '../lib/commands';
import type { PathInfo } from '../lib/commands';
import { 
  FolderOpen, 
  Check, 
  AlertCircle, 
  RefreshCw, 
  Server,
  HardDrive
} from 'lucide-react';

interface FolderSelectorProps {
  type: 'artifacts' | 'workspace';
  label: string;
  description?: string;
  onPathChange?: (path: string) => void;
  className?: string;
}

export function FolderSelector({ 
  type, 
  label, 
  description, 
  onPathChange,
  className = ''
}: FolderSelectorProps) {
  const [pathInfo, setPathInfo] = useState<PathInfo | null>(null);
  const [inputPath, setInputPath] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showInput, setShowInput] = useState(!capabilities.nativeDialog);

  // Load current path on mount
  useEffect(() => {
    loadCurrentPath();
  }, []);

  const loadCurrentPath = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const info = type === 'artifacts' 
        ? await cmd.getArtifactsRoot()
        : await cmd.getWorkspaceRoot();
      
      setPathInfo(info);
      setInputPath(info.path);
      
      if (info.valid && onPathChange) {
        onPathChange(info.path);
      }
    } catch (err) {
      setError(String(err));
    } finally {
      setIsLoading(false);
    }
  };

  const handleNativeSelect = async () => {
    if (!capabilities.nativeDialog) {
      setShowInput(true);
      return;
    }

    try {
      const { open } = await import('@tauri-apps/plugin-dialog');
      const selected = await open({
        directory: true,
        multiple: false,
        title: `Select ${label}`,
      });

      if (selected && typeof selected === 'string') {
        await setPath(selected);
      }
    } catch (err) {
      setError(`Failed to open folder dialog: ${err}`);
      // Fallback to input mode
      setShowInput(true);
    }
  };

  const setPath = async (path: string) => {
    setIsLoading(true);
    setError(null);
    
    try {
      const newPath = type === 'artifacts'
        ? await cmd.setArtifactsRoot(path)
        : await cmd.setWorkspaceRoot(path);
      
      // Reload path info
      const info = type === 'artifacts'
        ? await cmd.getArtifactsRoot()
        : await cmd.getWorkspaceRoot();
      
      setPathInfo(info);
      setInputPath(newPath);
      
      if (onPathChange) {
        onPathChange(newPath);
      }
    } catch (err) {
      setError(String(err));
    } finally {
      setIsLoading(false);
    }
  };

  const handleInputSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (inputPath.trim()) {
      await setPath(inputPath.trim());
    }
  };

  const getStatusIcon = () => {
    if (isLoading) {
      return <RefreshCw className="w-5 h-5 animate-spin text-terminal-muted" />;
    }
    if (pathInfo?.valid) {
      return <Check className="w-5 h-5 text-profit" />;
    }
    if (error || (pathInfo && !pathInfo.valid)) {
      return <AlertCircle className="w-5 h-5 text-loss" />;
    }
    return <FolderOpen className="w-5 h-5 text-terminal-muted" />;
  };

  const getStatusText = () => {
    if (isLoading) return 'Loading...';
    if (error) return error;
    if (!pathInfo) return 'Not configured';
    if (!pathInfo.valid) return 'Invalid path';
    
    if (type === 'artifacts') {
      if (pathInfo.has_index) {
        return 'Connected - index.json found';
      }
      return 'Path exists but no index.json';
    }
    
    if (type === 'workspace') {
      if (pathInfo.combiner_exists) {
        return 'Connected - combiner CLI found';
      }
      if (pathInfo.is_rust_project) {
        return 'Rust project found (build combiner first)';
      }
      return 'Path exists but not a valid workspace';
    }
    
    return 'Connected';
  };

  const isValid = pathInfo?.valid && (
    type === 'artifacts' ? pathInfo.has_index : pathInfo.combiner_exists || pathInfo.is_rust_project
  );

  return (
    <div className={`bg-terminal-surface rounded-xl border border-terminal-border p-4 ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-3">
          {type === 'artifacts' ? (
            <Server className="w-5 h-5 text-accent-cyan" />
          ) : (
            <HardDrive className="w-5 h-5 text-accent-purple" />
          )}
          <div>
            <h3 className="font-medium">{label}</h3>
            {description && (
              <p className="text-xs text-terminal-muted">{description}</p>
            )}
          </div>
        </div>
        {getStatusIcon()}
      </div>

      {/* Path Display / Input */}
      {showInput || !capabilities.nativeDialog ? (
        <form onSubmit={handleInputSubmit} className="space-y-3">
          <div className="flex gap-2">
            <input
              type="text"
              value={inputPath}
              onChange={(e) => setInputPath(e.target.value)}
              placeholder={type === 'artifacts' 
                ? '/path/to/project/artifacts'
                : '/path/to/project'
              }
              className="flex-1 px-3 py-2 bg-terminal-bg border border-terminal-border rounded-lg text-sm font-mono focus:outline-none focus:border-profit"
              disabled={isLoading}
            />
            <button
              type="submit"
              disabled={isLoading || !inputPath.trim()}
              className="px-4 py-2 bg-profit text-black font-medium rounded-lg hover:bg-profit/90 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isLoading ? (
                <RefreshCw className="w-4 h-4 animate-spin" />
              ) : (
                'Connect'
              )}
            </button>
          </div>
          
          {/* Status */}
          <div className={`text-xs flex items-center gap-2 ${
            isValid ? 'text-profit' : error ? 'text-loss' : 'text-terminal-muted'
          }`}>
            {getStatusIcon()}
            <span>{getStatusText()}</span>
          </div>
          
          {/* Common paths suggestion */}
          {!isValid && !isLoading && (
            <div className="text-xs text-terminal-muted">
              <span className="block mb-1">Common paths:</span>
              <div className="flex flex-wrap gap-2">
                {type === 'artifacts' ? (
                  <>
                    <button
                      type="button"
                      onClick={() => setInputPath('../artifacts')}
                      className="px-2 py-1 bg-terminal-bg rounded hover:bg-terminal-border transition-colors"
                    >
                      ../artifacts
                    </button>
                    <button
                      type="button"
                      onClick={() => setInputPath('./artifacts')}
                      className="px-2 py-1 bg-terminal-bg rounded hover:bg-terminal-border transition-colors"
                    >
                      ./artifacts
                    </button>
                  </>
                ) : (
                  <>
                    <button
                      type="button"
                      onClick={() => setInputPath('..')}
                      className="px-2 py-1 bg-terminal-bg rounded hover:bg-terminal-border transition-colors"
                    >
                      ..
                    </button>
                    <button
                      type="button"
                      onClick={() => setInputPath('.')}
                      className="px-2 py-1 bg-terminal-bg rounded hover:bg-terminal-border transition-colors"
                    >
                      .
                    </button>
                  </>
                )}
              </div>
            </div>
          )}
        </form>
      ) : (
        /* Native dialog mode (Tauri) */
        <div className="space-y-3">
          {pathInfo?.path ? (
            <div className="flex items-center gap-2">
              <div className="flex-1 px-3 py-2 bg-terminal-bg border border-terminal-border rounded-lg text-sm font-mono truncate">
                {pathInfo.path}
              </div>
              <button
                onClick={handleNativeSelect}
                className="px-4 py-2 bg-terminal-surface border border-terminal-border rounded-lg hover:border-profit transition-colors"
              >
                Change
              </button>
            </div>
          ) : (
            <button
              onClick={handleNativeSelect}
              className="w-full px-4 py-3 bg-gradient-to-r from-profit/20 to-accent-cyan/20 border border-profit/30 rounded-lg hover:border-profit transition-all flex items-center justify-center gap-2"
            >
              <FolderOpen className="w-5 h-5" />
              <span>Select {label}</span>
            </button>
          )}
          
          {/* Status */}
          <div className={`text-xs flex items-center gap-2 ${
            isValid ? 'text-profit' : error ? 'text-loss' : 'text-terminal-muted'
          }`}>
            {getStatusIcon()}
            <span>{getStatusText()}</span>
          </div>
          
          {/* Toggle to manual input */}
          <button
            onClick={() => setShowInput(true)}
            className="text-xs text-terminal-muted hover:text-terminal-text transition-colors"
          >
            Enter path manually →
          </button>
        </div>
      )}

      {/* Platform indicator */}
      <div className="mt-3 pt-3 border-t border-terminal-border/50 flex items-center justify-between text-xs text-terminal-muted">
        <span>
          {platform.isTauri ? '🖥️ Desktop Mode' : '🌐 Browser Mode'}
        </span>
        <button
          onClick={loadCurrentPath}
          className="flex items-center gap-1 hover:text-terminal-text transition-colors"
          disabled={isLoading}
        >
          <RefreshCw className={`w-3 h-3 ${isLoading ? 'animate-spin' : ''}`} />
          Refresh
        </button>
      </div>
    </div>
  );
}

export default FolderSelector;

