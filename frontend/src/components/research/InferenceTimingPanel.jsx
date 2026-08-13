export default function InferenceTimingPanel({ timing, jobTiming }) {
  const totalPipeline = jobTiming?.total_s ?? timing?.total_s ?? null;
  const modelInference = timing?.inference_s ?? null;
  const modelLoad = timing?.model_load_s ?? null;
  const mriLoad = timing?.mri_load_s ?? null;

  const hasAnyTiming = totalPipeline != null || modelInference != null;

  return (
    <div className="bg-parchment border border-sepia-border rounded-sm p-6 sm:p-8">
      <div className="atlas-section-header">
        <h2 className="font-serif text-2xl font-semibold text-ink">Inference Performance</h2>
      </div>

      {!hasAnyTiming ? (
        <p className="text-sepia-muted text-sm">Inference timing not available for this run.</p>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          {totalPipeline != null && (
            <div className="atlas-metadata-field">
              <p className="atlas-label mb-1">Total Pipeline Time</p>
              <p className="text-3xl font-mono font-medium text-ink-mono">
                {Number(totalPipeline).toFixed(2)} s
              </p>
            </div>
          )}
          {modelInference != null && (
            <div className="atlas-metadata-field">
              <p className="atlas-label mb-1">Model Inference Time</p>
              <p className="text-2xl font-mono font-medium text-ink-mono">
                {Number(modelInference).toFixed(2)} s
              </p>
            </div>
          )}
          {(modelLoad != null || mriLoad != null) && (
            <div className="atlas-metadata-field sm:col-span-2">
              <p className="atlas-label mb-2">Additional Timing</p>
              <div className="flex flex-wrap gap-x-6 gap-y-1 text-sm font-mono text-ink-mono">
                {modelLoad != null && (
                  <span>Model Load Time: {Number(modelLoad).toFixed(3)} s</span>
                )}
                {mriLoad != null && (
                  <span>MRI Load Time: {Number(mriLoad).toFixed(3)} s</span>
                )}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
