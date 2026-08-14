import GroundTruthComparisonViewer from './GroundTruthComparisonViewer';

export default function GroundTruthComparisonPanel({ result }) {
  const hasGt = Boolean(
    result?.ground_truth_available || result?.ground_truth_path
  );

  return (
    <div className="bg-parchment border border-sepia-border rounded-sm p-6 sm:p-8">
      <div className="atlas-section-header">
        <h2 className="font-serif text-2xl font-semibold text-ink">Ground Truth vs Prediction</h2>
      </div>

      {!hasGt ? (
        <div className="bg-parchment-dark border border-sepia-border rounded-sm p-5">
          <p className="text-sepia-muted text-sm">
            Ground-truth segmentation is not available for this case. Upload an optional
            segmentation mask on the Predict page to enable comparison.
          </p>
        </div>
      ) : (
        <GroundTruthComparisonViewer
          mriPath={result.mri_path}
          predMaskPath={result.mask_path}
          gtMaskPath={result.ground_truth_path}
          comparisonMaskPath={result.comparison_mask_path}
        />
      )}
    </div>
  );
}
