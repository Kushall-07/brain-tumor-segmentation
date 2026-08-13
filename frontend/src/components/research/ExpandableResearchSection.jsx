import { useState } from 'react';
import { ChevronDown, ChevronRight } from 'lucide-react';

export default function ExpandableResearchSection({ title, subtitle, children, defaultOpen = false }) {
  const [open, setOpen] = useState(defaultOpen);

  return (
    <div className="border border-sepia-border rounded-sm bg-parchment overflow-hidden">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center justify-between px-5 py-4 text-left hover:bg-parchment-dark transition-colors"
      >
        <div>
          <h3 className="font-serif text-lg font-semibold text-ink">{title}</h3>
          {subtitle && <p className="text-sm text-ink-body mt-0.5">{subtitle}</p>}
        </div>
        {open ? (
          <ChevronDown className="text-sepia-muted shrink-0" size={20} />
        ) : (
          <ChevronRight className="text-sepia-muted shrink-0" size={20} />
        )}
      </button>
      {open && (
        <div className="px-5 pb-5 pt-0 border-t border-sepia-border bg-parchment-dark/40">
          {children}
        </div>
      )}
    </div>
  );
}
