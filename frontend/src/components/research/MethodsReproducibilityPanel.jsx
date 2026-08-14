import { useEffect, useState } from 'react';
import predictionService from '../../services/predictionService';
import ExpandableResearchSection from './ExpandableResearchSection';

function SectionBlock({ title, children }) {
  return (
    <div className="mb-4">
      <p className="atlas-label mb-2">{title}</p>
      <div className="text-sm text-ink-body space-y-1">{children}</div>
    </div>
  );
}

export default function MethodsReproducibilityPanel() {
  const [methods, setMethods] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    predictionService.getMethodsSummary()
      .then((data) => setMethods(data.methods))
      .catch(() => setMethods(null))
      .finally(() => setLoading(false));
  }, []);

  return (
    <ExpandableResearchSection title="Methods & Reproducibility" subtitle="Dataset, preprocessing, and training configuration">
      {loading ? (
        <p className="text-sm text-sepia-muted py-2">Loading methods summary...</p>
      ) : !methods ? (
        <p className="text-sm text-sepia-muted py-2">Methods summary unavailable.</p>
      ) : (
        <div className="pt-4">
          <SectionBlock title="Dataset">
            <p>{methods.dataset?.name}</p>
            <p className="font-mono text-xs">{methods.dataset?.subset_split}</p>
          </SectionBlock>

          <SectionBlock title="Input Modalities">
            <p>{methods.input_modalities?.join(', ')}</p>
          </SectionBlock>

          <SectionBlock title="Preprocessing">
            <p>Normalization: {methods.preprocessing?.normalization}</p>
            <p>Resampling: {methods.preprocessing?.resampling}</p>
            <p>Patch extraction: {methods.preprocessing?.patch_extraction}</p>
            <p>Orientation: {methods.preprocessing?.orientation}</p>
          </SectionBlock>

          <SectionBlock title="Architecture">
            <p>{methods.architecture?.name} — {methods.architecture?.type}</p>
            <p className="font-mono text-xs">
              Patch: {methods.architecture?.patch_size?.join('×')} ·
              Classes: {methods.architecture?.output_classes} ·
              Feature size: {methods.architecture?.swin_feature_size}
            </p>
          </SectionBlock>

          <SectionBlock title="Training">
            <p>Optimizer: {methods.training?.optimizer} · Loss: {methods.training?.loss_function}</p>
            <p className="font-mono text-xs">
              Epochs: {methods.training?.epochs} · Batch: {methods.training?.batch_size}
              (effective {methods.training?.effective_batch_size}) · LR: {methods.training?.learning_rate}
            </p>
            <p>Scheduler: {methods.training?.scheduler} · EMA: {methods.training?.ema ? 'Yes' : 'No'}</p>
            <p>Augmentation: {methods.training?.augmentation}</p>
          </SectionBlock>

          <SectionBlock title="Hardware">
            <p>GPU: {methods.hardware?.gpu}</p>
            <p>CPU: {methods.hardware?.cpu}</p>
            <p>RAM: {methods.hardware?.ram}</p>
          </SectionBlock>
        </div>
      )}
    </ExpandableResearchSection>
  );
}
