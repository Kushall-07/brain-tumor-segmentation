import { useEffect, useState } from 'react';
import predictionService from '../../services/predictionService';
import ExpandableResearchSection from './ExpandableResearchSection';

function InfoRow({ label, value }) {
  return (
    <div className="flex flex-col sm:flex-row sm:items-start gap-1 sm:gap-4 py-2 border-b border-sepia-border/40 last:border-0">
      <span className="atlas-label sm:w-40 shrink-0">{label}</span>
      <span className="text-sm text-ink-body font-mono break-all">{value ?? 'Not specified'}</span>
    </div>
  );
}

export default function ModelInfoPanel({ checkpointPath }) {
  const [modelInfo, setModelInfo] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    predictionService.getModelInfo(checkpointPath)
      .then((data) => setModelInfo(data.model))
      .catch(() => setModelInfo(null))
      .finally(() => setLoading(false));
  }, [checkpointPath]);

  return (
    <ExpandableResearchSection title="Model Information" subtitle="Architecture and checkpoint details">
      {loading ? (
        <p className="text-sm text-sepia-muted py-2">Loading model information...</p>
      ) : !modelInfo ? (
        <p className="text-sm text-sepia-muted py-2">Model information unavailable.</p>
      ) : (
        <div className="pt-4 space-y-1">
          <InfoRow label="Architecture" value={modelInfo.architecture ?? '3D SwinUNETR'} />
          <InfoRow label="Task" value={modelInfo.task ?? 'Multi-modal brain tumor segmentation'} />
          <InfoRow label="Input Modalities" value={modelInfo.input_modalities?.join(', ') ?? 'T1, T1ce, T2, FLAIR'} />
          <InfoRow label="Output" value={`${modelInfo.output_classes ?? 4}-class segmentation`} />
          <InfoRow
            label="Classes"
            value={modelInfo.classes
              ? Object.entries(modelInfo.classes).map(([k, v]) => `${k}: ${v}`).join(' · ')
              : '0: Background · 1: NCR/NET · 2: Edema · 3: ET'}
          />
          <InfoRow label="Framework" value="PyTorch + MONAI" />
          <InfoRow label="Inference" value="Sliding-window 3D inference" />
          <InfoRow label="Dataset" value={modelInfo.dataset} />
          <InfoRow label="Checkpoint" value={modelInfo.checkpoint} />
          {modelInfo.parameter_count != null && (
            <InfoRow
              label="Parameters"
              value={modelInfo.parameter_count.toLocaleString()}
            />
          )}
          {modelInfo.training_validation_scores && (
            <div className="mt-4 pt-3 border-t border-sepia-border">
              <p className="atlas-label mb-2">Training Validation Scores (held-out set)</p>
              <p className="text-xs text-sepia-muted italic mb-2">{modelInfo.training_validation_note}</p>
              <div className="font-mono text-sm text-ink space-y-1">
                {Object.entries(modelInfo.training_validation_scores).map(([k, v]) => (
                  <div key={k}>{k}: {Number(v).toFixed(4)}</div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </ExpandableResearchSection>
  );
}
