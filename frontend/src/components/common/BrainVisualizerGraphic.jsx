import { Brain } from 'lucide-react';

export default function BrainVisualizerGraphic() {
  return (
    <div className="relative w-full max-w-md mx-auto">
      <div className="relative border border-sepia-border rounded-sm p-8 bg-parchment-dark">
        {/* Corner marks */}
        <span className="absolute top-2 left-2 w-3 h-3 border-t border-l border-brass" />
        <span className="absolute top-2 right-2 w-3 h-3 border-t border-r border-brass" />
        <span className="absolute bottom-2 left-2 w-3 h-3 border-b border-l border-brass" />
        <span className="absolute bottom-2 right-2 w-3 h-3 border-b border-r border-brass" />

        <div className="relative flex items-center justify-center py-4">
          <Brain
            className="text-annotation w-48 h-48 sm:w-56 sm:h-56"
            strokeWidth={0.75}
          />
        </div>

        <p className="text-center atlas-label mt-4">Fig. 1 — Anatomical Reference</p>
      </div>
    </div>
  );
}
