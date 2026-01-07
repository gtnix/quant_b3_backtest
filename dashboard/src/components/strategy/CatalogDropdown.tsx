import { ChevronDown, Folder, Check } from 'lucide-react';
import { useState, useRef, useEffect } from 'react';
import type { StrategyCatalog } from '../../stores/strategyStore';

interface CatalogDropdownProps {
  catalogs: StrategyCatalog[];
  activeCatalog: string | null;
  onSelect: (slug: string) => void;
}

export function CatalogDropdown({ catalogs, activeCatalog, onSelect }: CatalogDropdownProps) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  const active = catalogs.find((c) => c.slug === activeCatalog);

  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  return (
    <div ref={ref} className="relative">
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-2 px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-sm hover:border-slate-600 transition-colors"
      >
        <Folder className="w-4 h-4 text-amber-400" />
        <span className="text-white">{active?.name || 'Selecionar Catálogo'}</span>
        <ChevronDown className={`w-4 h-4 text-slate-400 transition-transform ${open ? 'rotate-180' : ''}`} />
      </button>

      {open && (
        <div className="absolute top-full left-0 mt-2 w-72 bg-slate-900 border border-slate-700 rounded-lg shadow-xl z-50 overflow-hidden animate-in fade-in slide-in-from-top-2 duration-150">
          <div className="p-2 border-b border-slate-800">
            <span className="text-xs font-semibold text-slate-500 uppercase px-2">Catálogos</span>
          </div>
          <div className="max-h-64 overflow-y-auto">
            {catalogs.map((catalog) => (
              <button
                key={catalog.slug}
                onClick={() => {
                  onSelect(catalog.slug);
                  setOpen(false);
                }}
                className={`w-full flex items-start gap-3 p-3 hover:bg-slate-800 transition-colors text-left ${
                  activeCatalog === catalog.slug ? 'bg-slate-800/50' : ''
                }`}
              >
                <div className="flex-1">
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-medium text-white">{catalog.name}</span>
                    {catalog.is_system && (
                      <span className="px-1.5 py-0.5 text-[10px] bg-cyan-500/20 text-cyan-400 rounded">
                        Sistema
                      </span>
                    )}
                  </div>
                  <p className="text-xs text-slate-400 mt-0.5">{catalog.description}</p>
                </div>
                {activeCatalog === catalog.slug && (
                  <Check className="w-4 h-4 text-cyan-400 flex-shrink-0" />
                )}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}


