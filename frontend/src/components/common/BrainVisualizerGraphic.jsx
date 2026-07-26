import { Brain } from 'lucide-react';

export default function BrainVisualizerGraphic() {
  return (
    <div className="relative w-full max-w-md mx-auto">
      {/* Glow effect */}
      <div className="absolute inset-0 bg-cyan-500/20 rounded-full blur-3xl animate-pulse" />
      
      {/* Brain icon with animation */}
      <div className="relative flex items-center justify-center">
        <Brain 
          className="text-cyan-400 w-64 h-64 animate-[pulse_3s_ease-in-out_infinite]" 
          strokeWidth={1}
        />
        
        {/* Scanning line effect */}
        <div className="absolute inset-0 overflow-hidden rounded-full">
          <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-transparent via-cyan-400 to-transparent animate-[scan_2s_linear_infinite]" />
        </div>
      </div>
    </div>
  );
}
