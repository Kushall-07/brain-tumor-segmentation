import { useEffect, useRef, useState } from 'react';
import { Niivue } from '@niivue/niivue';
import predictionService from '../../services/predictionService';

const TABS = [
  { id: 'gt', label: 'Ground Truth' },
  { id: 'pred', label: 'Prediction' },
  { id: 'comparison', label: 'Comparison' },
];

function toRelativePredictionPath(filePath) {
  if (!filePath) return '';
  return filePath.replace(/\\/g, '/').replace(/.*outputs\/predictions\//, '');
}

function filenameFromPath(filePath, fallback) {
  if (!filePath) return fallback;
  const normalized = filePath.replace(/\\/g, '/');
  return normalized.split('/').pop() || fallback;
}

function validateVolumes(volumes) {
  volumes.forEach((volume, index) => {
    if (!volume?.url || typeof volume.url !== 'string' || !volume.url.trim()) {
      throw new Error(`Volume ${index} is missing a valid URL`);
    }
    if (!volume?.name || typeof volume.name !== 'string' || !volume.name.trim()) {
      throw new Error(`Volume ${index} is missing a valid filename`);
    }
  });
}

async function downloadVolumeBlob(filePath) {
  const relativePath = toRelativePredictionPath(filePath);
  return predictionService.downloadPrediction(relativePath);
}

export default function GroundTruthComparisonViewer({
  mriPath,
  predMaskPath,
  gtMaskPath,
  comparisonMaskPath,
}) {
  const canvasRef = useRef(null);
  const nvRef = useRef(null);
  const [activeTab, setActiveTab] = useState('gt');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const blobUrlsRef = useRef([]);

  const mriFileName = filenameFromPath(mriPath, 'BraTS-Patient_flair.nii.gz');
  const gtFileName = filenameFromPath(gtMaskPath, 'BraTS-Patient_gt.nii.gz');
  const predFileName = filenameFromPath(predMaskPath, 'BraTS-Patient_pred.nii.gz');
  const comparisonFileName = filenameFromPath(
    comparisonMaskPath,
    'BraTS-Patient_comparison.nii.gz'
  );

  const comparisonReady = Boolean(mriPath && gtMaskPath && predMaskPath);

  useEffect(() => {
    let cancelled = false;

    const loadViewer = async () => {
      try {
        setLoading(true);
        setError(null);

        blobUrlsRef.current.forEach((url) => URL.revokeObjectURL(url));
        blobUrlsRef.current = [];

        console.log('[GroundTruthComparisonViewer] Ground Truth URL path:', gtMaskPath);
        console.log('[GroundTruthComparisonViewer] Ground Truth filename:', gtFileName);
        console.log('[GroundTruthComparisonViewer] Prediction URL path:', predMaskPath);
        console.log('[GroundTruthComparisonViewer] Prediction filename:', predFileName);

        const mriBlob = await downloadVolumeBlob(mriPath);
        const mriUrl = URL.createObjectURL(mriBlob);
        blobUrlsRef.current.push(mriUrl);

        const volumes = [
          {
            url: mriUrl,
            name: mriFileName,
            colormap: 'gray',
            opacity: 1,
            visible: true,
          },
        ];

        if (activeTab === 'gt') {
          const gtBlob = await downloadVolumeBlob(gtMaskPath);
          const gtUrl = URL.createObjectURL(gtBlob);
          blobUrlsRef.current.push(gtUrl);
          console.log('[GroundTruthComparisonViewer] Ground Truth URL:', gtUrl);
          volumes.push({
            url: gtUrl,
            name: gtFileName,
            colormap: 'green',
            opacity: 0.55,
            visible: true,
            isLabelMap: true,
          });
        } else if (activeTab === 'pred') {
          const predBlob = await downloadVolumeBlob(predMaskPath);
          const predUrl = URL.createObjectURL(predBlob);
          blobUrlsRef.current.push(predUrl);
          console.log('[GroundTruthComparisonViewer] Prediction URL:', predUrl);
          volumes.push({
            url: predUrl,
            name: predFileName,
            colormap: 'red',
            opacity: 0.55,
            visible: true,
            isLabelMap: true,
          });
        } else if (activeTab === 'comparison') {
          if (comparisonMaskPath) {
            const comparisonBlob = await downloadVolumeBlob(comparisonMaskPath);
            const comparisonUrl = URL.createObjectURL(comparisonBlob);
            blobUrlsRef.current.push(comparisonUrl);
            console.log('[GroundTruthComparisonViewer] Comparison URL:', comparisonUrl);
            console.log('[GroundTruthComparisonViewer] Comparison filename:', comparisonFileName);
            volumes.push({
              url: comparisonUrl,
              name: comparisonFileName,
              colormap: 'hot',
              opacity: 0.65,
              visible: true,
              isLabelMap: true,
            });
          } else {
            const [gtBlob, predBlob] = await Promise.all([
              downloadVolumeBlob(gtMaskPath),
              downloadVolumeBlob(predMaskPath),
            ]);
            const gtUrl = URL.createObjectURL(gtBlob);
            const predUrl = URL.createObjectURL(predBlob);
            blobUrlsRef.current.push(gtUrl, predUrl);
            console.log('[GroundTruthComparisonViewer] Comparison fallback GT URL:', gtUrl);
            console.log('[GroundTruthComparisonViewer] Comparison fallback Prediction URL:', predUrl);
            volumes.push(
              {
                url: gtUrl,
                name: gtFileName,
                colormap: 'green',
                opacity: 0.45,
                visible: true,
                isLabelMap: true,
              },
              {
                url: predUrl,
                name: predFileName,
                colormap: 'red',
                opacity: 0.45,
                visible: true,
                isLabelMap: true,
              }
            );
          }
        }

        if (cancelled) return;

        if (nvRef.current) {
          nvRef.current = null;
        }

        const nv = new Niivue({
          show3Dcrosshair: true,
          isColorbar: true,
        });
        nvRef.current = nv;
        nv.attachToCanvas(canvasRef.current);

        volumes.forEach((volume, index) => {
          console.log(`[GroundTruthComparisonViewer] volume ${index}`, {
            url: volume?.url,
            name: volume?.name,
            colormap: volume?.colormap,
          });
        });

        const validVolumes = volumes.filter((volume) => volume?.url && volume?.name);
        console.log('[GroundTruthComparisonViewer] Valid volumes:', validVolumes);

        if (!validVolumes.length) {
          throw new Error('No valid volumes available for comparison viewer');
        }

        validateVolumes(validVolumes);
        console.log('[GroundTruthComparisonViewer] Loading volumes:', validVolumes);

        await nv.loadVolumes(validVolumes);
        if (!cancelled) setLoading(false);
      } catch (err) {
        console.error('[GroundTruthComparisonViewer]', err);
        if (!cancelled) {
          setError(err?.message || 'Failed to load comparison visualization');
          setLoading(false);
        }
      }
    };

    if (comparisonReady) {
      loadViewer();
    }

    return () => {
      cancelled = true;
      blobUrlsRef.current.forEach((url) => URL.revokeObjectURL(url));
      blobUrlsRef.current = [];
      nvRef.current = null;
    };
  }, [
    mriPath,
    predMaskPath,
    gtMaskPath,
    comparisonMaskPath,
    activeTab,
    mriFileName,
    gtFileName,
    predFileName,
    comparisonFileName,
    comparisonReady,
  ]);

  return (
    <div className="w-full overflow-hidden bg-parchment border border-sepia-border">
      <div className="flex flex-wrap items-center gap-2 border-b border-sepia-border bg-parchment-dark px-4 py-3">
        <span className="atlas-label mr-2">View</span>
        {TABS.map((tab) => (
          <button
            key={tab.id}
            type="button"
            onClick={() => setActiveTab(tab.id)}
            disabled={tab.id === 'comparison' && !comparisonReady}
            className={`rounded-sm border px-3 py-1.5 text-xs font-medium uppercase tracking-wide transition-colors ${
              activeTab === tab.id
                ? 'border-annotation bg-parchment text-annotation'
                : 'border-sepia-border bg-parchment text-sepia-muted hover:text-ink disabled:opacity-40'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {activeTab === 'comparison' && (
        <div className="px-4 py-2 border-b border-sepia-border bg-parchment text-xs text-sepia-muted">
          {comparisonMaskPath
            ? 'Comparison (WT region): overlap · prediction-only · ground-truth-only'
            : 'Overlay comparison: green = ground truth · red = prediction · overlap regions appear blended'}
        </div>
      )}

      <div className="relative h-[420px] w-full bg-slate-900">
        <canvas ref={canvasRef} className="h-full w-full" />
        {loading && (
          <div className="absolute inset-0 flex items-center justify-center bg-slate-900/90">
            <p className="text-stone-400 text-sm">Loading comparison view...</p>
          </div>
        )}
        {error && (
          <div className="absolute inset-0 flex items-center justify-center bg-slate-900/90">
            <p className="text-red-400 text-sm">{error}</p>
          </div>
        )}
      </div>
    </div>
  );
}
