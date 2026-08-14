export default function BrainVisualizerGraphic() {
  return (
    <div className="relative w-full max-w-md mx-auto">
      <div className="relative border border-sepia-border rounded-sm p-4 bg-parchment-dark">
        {/* Corner marks */}
        <span className="absolute top-2 left-2 w-3 h-3 border-t border-l border-brass" />
        <span className="absolute top-2 right-2 w-3 h-3 border-t border-r border-brass" />
        <span className="absolute bottom-2 left-2 w-3 h-3 border-b border-l border-brass" />
        <span className="absolute bottom-2 right-2 w-3 h-3 border-b border-r border-brass" />

        <div className="relative flex items-center justify-center">
          <img
            src="/brain-tumor-ai-fig1.png"
            alt="Brain tumor segmentation anatomical reference"
            className="w-full h-auto object-contain"
          />
        </div>
      </div>
    </div>
  );
}
