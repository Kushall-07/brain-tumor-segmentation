import { useEffect, useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import ValidationMetricsPanel from '../components/research/ValidationMetricsPanel';
import GroundTruthComparisonPanel from '../components/research/GroundTruthComparisonPanel';
import InferenceTimingPanel from '../components/research/InferenceTimingPanel';
import ModelInfoPanel from '../components/research/ModelInfoPanel';
import MethodsReproducibilityPanel from '../components/research/MethodsReproducibilityPanel';
import LimitationsStatement from '../components/research/LimitationsStatement';
import PipelineDiagram from '../components/research/PipelineDiagram';
import predictionService from '../services/predictionService';

const CHECKPOINT_PATH = 'outputs/exp_swinunetr_4class_et_fixed/checkpoints/best_mean_dice.pt';

export default function ModelEvaluation() {
  const navigate = useNavigate();
  const [result, setResult] = useState(null);
  const [validationLoading, setValidationLoading] = useState(false);
  const validationAttemptKeyRef = useRef(null);

  useEffect(() => {
    try {
      const stored = sessionStorage.getItem('brainTumorLatestPrediction');
      if (stored) {
        const parsed = JSON.parse(stored);
        console.log('[VALIDATION] Stored prediction result:', parsed);
        setResult(parsed);
      }
    } catch (error) {
      console.error('Failed to parse stored prediction:', error);
    }
  }, []);

  useEffect(() => {
    if (!result?.ground_truth_path || !result?.mask_path) {
      return;
    }

    const attemptKey = `${result.mask_path}|${result.ground_truth_path}`;
    const hasMetrics = Boolean(
      result.validation_metrics?.available === true ||
      (result.validation_metrics?.metrics && result.validation_metrics?.available !== false)
    );

    if (hasMetrics || validationLoading || validationAttemptKeyRef.current === attemptKey) {
      return;
    }

    validationAttemptKeyRef.current = attemptKey;
    let cancelled = false;
    setValidationLoading(true);

    predictionService
      .validateCase(result.mask_path, result.ground_truth_path)
      .then((validation) => {
        if (cancelled) return;

        console.log('[VALIDATION] API RESPONSE:', validation);

        setResult((prev) => {
          const next = {
            ...prev,
            validation_metrics: validation?.available ? validation : prev?.validation_metrics,
            validation_error: validation?.available
              ? null
              : validation?.reason || prev?.validation_error,
            validation_error_type: validation?.available
              ? null
              : validation?.error_type || prev?.validation_error_type,
          };

          try {
            sessionStorage.setItem('brainTumorLatestPrediction', JSON.stringify(next));
          } catch (storageError) {
            console.error('Failed to update stored prediction validation:', storageError);
          }

          return next;
        });
      })
      .catch((error) => {
        if (cancelled) return;
        console.error('[VALIDATION] API call failed:', error);
        const detail = error.response?.data?.detail;
        setResult((prev) => ({
          ...prev,
          validation_error: detail || error.message || 'Validation request failed',
          validation_error_type: 'RequestError',
        }));
      })
      .finally(() => {
        if (!cancelled) {
          setValidationLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [result?.ground_truth_path, result?.mask_path, result?.validation_metrics, validationLoading]);

  const groundTruthAvailable = Boolean(
    result?.ground_truth_available || result?.ground_truth_path
  );

  return (
    <div className="min-h-screen bg-parchment py-12 px-4 sm:px-6 lg:px-8 pt-28">
      <div className="max-w-4xl mx-auto">
        {/* Page header */}
        <div className="text-center mb-10 pb-8 border-b border-sepia-border">
          <p className="atlas-label mb-2">Research & Technical Report</p>
          <h1 className="font-serif text-4xl font-semibold text-ink mb-3">
            Model &amp; Evaluation
          </h1>
          <p className="text-ink-body text-lg max-w-2xl mx-auto">
            Model performance, reproducibility, inference analysis, and system architecture.
          </p>
          {result?.case_id && (
            <p className="text-sm text-sepia-muted mt-4">
              Current case: <span className="font-mono text-ink">{result.case_id}</span>
            </p>
          )}
        </div>

        {!result && (
          <div className="mb-8 bg-parchment-dark border border-sepia-border rounded-sm p-5">
            <p className="text-sm text-ink-body leading-relaxed">
              No prediction data available. Run a segmentation from the{' '}
              <button
                type="button"
                onClick={() => navigate('/predict')}
                className="text-annotation underline hover:text-ink"
              >
                Predict
              </button>{' '}
              page to populate case-specific validation metrics and inference timing.
              Model information, methods, limitations, and pipeline details are shown below.
            </p>
          </div>
        )}

        <div className="space-y-6">
          {/* 1. Model Validation */}
          <ValidationMetricsPanel
            validationMetrics={result?.validation_metrics}
            groundTruthAvailable={groundTruthAvailable}
            validationError={result?.validation_error}
            validationErrorType={result?.validation_error_type}
            loading={validationLoading}
          />

          {/* 2. Ground Truth vs Prediction */}
          {result ? (
            <GroundTruthComparisonPanel result={result} />
          ) : (
            <div className="bg-parchment border border-sepia-border rounded-sm p-6 sm:p-8">
              <div className="atlas-section-header">
                <h2 className="font-serif text-2xl font-semibold text-ink">Ground Truth vs Prediction</h2>
              </div>
              <div className="bg-parchment-dark border border-sepia-border rounded-sm p-5">
                <p className="text-sepia-muted text-sm">
                  Ground-truth segmentation is not available for this case. Upload an optional
                  segmentation mask on the Predict page to enable comparison.
                </p>
              </div>
            </div>
          )}

          {/* 3. Inference Performance */}
          <InferenceTimingPanel timing={result?.timing} jobTiming={result?.job_timing} />

          {/* 4. Model Information */}
          <ModelInfoPanel checkpointPath={CHECKPOINT_PATH} />

          {/* 5. Methods & Reproducibility */}
          <MethodsReproducibilityPanel />

          {/* 6. Limitations */}
          <LimitationsStatement />

          {/* 7. Architecture / Pipeline */}
          <PipelineDiagram />
        </div>

        <div className="mt-10 pt-6 border-t border-sepia-border text-center text-sm text-sepia-muted">
          <p className="font-serif text-ink">Brain Tumor Segmentation AI System</p>
          <p className="mt-1">Powered by 3D U-Net and MONAI</p>
        </div>
      </div>
    </div>
  );
}
