function PipelineStep({ label, detail, isLast = false }) {
  return (
    <div className="flex flex-col items-center">
      <div className="w-full max-w-xs border border-sepia-border bg-parchment-dark rounded-sm px-4 py-3 text-center">
        <p className="atlas-label text-ink">{label}</p>
        {detail && <p className="text-xs text-ink-body mt-1">{detail}</p>}
      </div>
      {!isLast && (
        <div className="flex flex-col items-center py-2">
          <div className="w-px h-4 bg-brass" />
          <span className="text-brass text-xs">▼</span>
        </div>
      )}
    </div>
  );
}

export default function PipelineDiagram() {
  return (
    <div className="bg-parchment border border-sepia-border rounded-sm p-6 sm:p-8">
      <h2 className="font-serif text-xl font-semibold text-ink mb-2 pb-3 border-b border-sepia-border">
        Architecture / Pipeline
      </h2>
      <p className="text-xs text-sepia-muted mb-6 italic">
        Visual summary only — does not alter the inference pipeline.
      </p>

      <div className="flex flex-col items-center max-w-lg mx-auto">
        <PipelineStep label="MRI Input" detail="T1 · T1ce · T2 · FLAIR" />
        <PipelineStep label="Multi-Modal Preprocessing" detail="Z-score normalization · canonical orientation" />
        <PipelineStep label="3D SwinUNETR" detail="Sliding-window inference" />
        <PipelineStep label="4-Class Segmentation" detail="Background · NCR/NET · Edema · ET" />
        <PipelineStep label="Post-Processing / Analysis" detail="Volume · dimensions · WT/TC/ET" />
        <PipelineStep label="Clinical Visualization" detail="NiiVue multi-planar viewer" isLast />
      </div>
    </div>
  );
}
