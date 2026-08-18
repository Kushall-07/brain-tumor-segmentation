function formatMetric(value) {
  if (value === null || value === undefined) return '—';
  if (typeof value === 'number') return value.toFixed(4);
  return String(value);
}

function normalizeValidationMetrics(validationMetrics) {
  if (!validationMetrics) return null;

  if (validationMetrics.available === false) {
    return null;
  }

  if (validationMetrics.available === true && validationMetrics.metrics) {
    const { metrics } = validationMetrics;
    return {
      wt: metrics.WT,
      tc: metrics.TC,
      et: metrics.ET,
    };
  }

  if (validationMetrics.metrics) {
    const { metrics } = validationMetrics;
    return {
      wt: metrics.WT,
      tc: metrics.TC,
      et: metrics.ET,
    };
  }

  return null;
}

function MetricsTable({ metrics }) {
  const regions = ['wt', 'tc', 'et'];
  const regionLabels = { wt: 'WT', tc: 'TC', et: 'ET' };
  const rows = [
    { key: 'dice', label: 'Dice' },
    { key: 'hd95_mm', altKey: 'hd95', label: 'HD95 (mm)' },
    { key: 'sensitivity', label: 'Sensitivity' },
    { key: 'specificity', label: 'Specificity' },
  ];

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm border-collapse">
        <thead>
          <tr className="border-b border-sepia-border">
            <th className="text-left py-2 pr-4 atlas-label">Metric</th>
            {regions.map((r) => (
              <th key={r} className="text-center py-2 px-3 atlas-label">{regionLabels[r]}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.key} className="border-b border-sepia-border/50">
              <td className="py-2.5 pr-4 text-ink font-medium">{row.label}</td>
              {regions.map((r) => (
                <td key={r} className="text-center py-2.5 px-3 font-mono text-ink-mono">
                  {formatMetric(metrics?.[r]?.[row.key] ?? metrics?.[r]?.[row.altKey])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function ValidationMetricsPanel({
  validationMetrics,
  groundTruthAvailable,
  validationError,
  validationErrorType,
  loading = false,
}) {
  const hasGroundTruth = Boolean(groundTruthAvailable);
  const normalizedMetrics = normalizeValidationMetrics(validationMetrics);
  const failureReason =
    validationError ||
    validationMetrics?.reason ||
    null;

  if (validationMetrics) {
    console.log('[VALIDATION] Panel metrics payload:', validationMetrics);
  }

  return (
    <div className="bg-parchment border border-sepia-border rounded-sm p-6 sm:p-8">
      <div className="atlas-section-header">
        <h2 className="font-serif text-2xl font-semibold text-ink">Model Validation</h2>
      </div>

      {loading && (
        <p className="text-sm text-ink-body mb-4">Computing validation metrics...</p>
      )}

      {!hasGroundTruth ? (
        <div className="bg-parchment-dark border border-sepia-border rounded-sm p-5">
          <p className="text-sepia-muted text-sm leading-relaxed">
            Ground truth unavailable for this case. Validation metrics (Dice, HD95, sensitivity,
            specificity) are computed only when a labeled ground-truth segmentation mask is provided
            with the upload.
          </p>
        </div>
      ) : !normalizedMetrics ? (
        <div className="bg-parchment-dark border border-sepia-border rounded-sm p-5">
          <p className="text-sm text-ink-body mb-2">
            A ground-truth segmentation mask was provided for this case, but validation metrics could
            not be computed.
          </p>
          {failureReason && (
            <div className="space-y-2">
              {validationErrorType && (
                <p className="text-xs text-arterial font-mono">{validationErrorType}</p>
              )}
              <p className="text-sm text-sepia-muted font-mono leading-relaxed">{failureReason}</p>
            </div>
          )}
        </div>
      ) : (
        <>
          <p className="text-sm text-ink-body mb-4">
            Segmentation performance for this case against the provided ground-truth mask.
          </p>
          <MetricsTable metrics={normalizedMetrics} />
          <p className="text-xs text-sepia-muted mt-4 italic">
            WT = NCR/NET + Edema + ET · TC = NCR/NET + ET · ET = enhancing tumor.
            HD95 is reported in millimeters and shown as — when undefined
            (e.g., empty region in prediction or ground truth).
          </p>
        </>
      )}
    </div>
  );
}
