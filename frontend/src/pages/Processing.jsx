import { useState, useEffect, useRef, useCallback } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { Loader2, AlertCircle, CheckCircle2 } from 'lucide-react';
import predictionService from '../services/predictionService';

const POLL_INTERVAL_MS = 1000;

// Module-level guard: ensures only one job-creation request is in flight
// even when React Strict Mode remounts the component during development.
let activeJobCreation = null;

const pipelineSteps = [
  { id: 'upload_received', label: 'MRI Input', description: 'Multi-modal MRI files received' },
  { id: 'validation', label: 'Input Validation', description: 'Validating NIfTI volumes' },
  { id: 'preprocessing', label: 'MRI Preprocessing', description: 'Preparing MRI volumes for inference' },
  { id: 'model_inference', label: 'Model Inference', description: 'Running 3D U-Net segmentation' },
  { id: 'segmentation_generation', label: 'Segmentation Generation', description: 'Generating segmentation mask' },
  { id: 'volume_analysis', label: 'Volumetric Analysis', description: 'Calculating estimated tumor volume' },
  { id: 'preparing_results', label: 'Preparing Results', description: 'Preparing visualization and analysis results' },
];

const stageOrder = pipelineSteps.map((step) => step.id);

function getStageIndex(stage) {
  if (stage === 'completed') {
    return stageOrder.length;
  }
  const index = stageOrder.indexOf(stage);
  return index >= 0 ? index : 0;
}

export default function Processing() {
  const navigate = useNavigate();
  const location = useLocation();
  const [status, setStatus] = useState('idle'); // idle, processing, error
  const [error, setError] = useState(null);
  const [backendStage, setBackendStage] = useState('upload_received');
  const [backendMessage, setBackendMessage] = useState('');
  const [files, setFiles] = useState(null);
  const requestStartedRef = useRef(false);
  const pollIntervalRef = useRef(null);
  const jobIdRef = useRef(null);

  const stopPolling = useCallback(() => {
    if (pollIntervalRef.current !== null) {
      clearInterval(pollIntervalRef.current);
      pollIntervalRef.current = null;
    }
  }, []);

  const handleJobComplete = useCallback((result, uploadedFiles) => {
    stopPolling();

    try {
      sessionStorage.setItem('brainTumorLatestPrediction', JSON.stringify(result));
    } catch (storageError) {
      console.error('Failed to store prediction in sessionStorage:', storageError);
    }

    navigate('/results', {
      state: {
        result,
        files: uploadedFiles,
      },
    });
  }, [navigate, stopPolling]);

  const pollJobStatus = useCallback(async (jobId, uploadedFiles) => {
    try {
      const jobStatus = await predictionService.getPredictionStatus(jobId);

      if (jobStatus.stage) {
        setBackendStage(jobStatus.stage);
      }
      if (jobStatus.message) {
        setBackendMessage(jobStatus.message);
      }

      if (jobStatus.status === 'completed' && jobStatus.result) {
        handleJobComplete(jobStatus.result, uploadedFiles);
        return;
      }

      if (jobStatus.status === 'failed') {
        stopPolling();
        setStatus('error');
        setError({
          status: 500,
          message: jobStatus.error || jobStatus.message || 'Analysis failed',
        });
      }
    } catch (pollError) {
      console.error('Status polling error:', pollError);
      stopPolling();
      setStatus('error');
      setError({
        status: pollError.status || 0,
        message: pollError.message || 'Failed to retrieve analysis status',
      });
    }
  }, [handleJobComplete, stopPolling]);

  const beginPolling = useCallback((jobId, uploadedFiles) => {
    stopPolling();

    pollJobStatus(jobId, uploadedFiles);

    pollIntervalRef.current = setInterval(() => {
      pollJobStatus(jobId, uploadedFiles);
    }, POLL_INTERVAL_MS);
  }, [pollJobStatus, stopPolling]);

  const startPrediction = useCallback(async (uploadedFiles) => {
    if (requestStartedRef.current && jobIdRef.current) {
      beginPolling(jobIdRef.current, uploadedFiles);
      return;
    }
    if (requestStartedRef.current) return;
    requestStartedRef.current = true;

    setStatus('processing');
    setError(null);
    setBackendStage('upload_received');
    setBackendMessage('MRI volumes received');

    const formData = new FormData();
    const modalities = ['t1', 't1ce', 't2', 'flair'];
    modalities.forEach((modality) => {
      if (uploadedFiles[modality]) {
        formData.append(modality, uploadedFiles[modality]);
      }
    });

    try {
      if (!activeJobCreation) {
        activeJobCreation = predictionService.startPrediction(formData, {
          checkpoint_path: 'outputs/checkpoints/best.pt',
          save_probabilities: false,
        });
      }

      const response = await activeJobCreation;
      activeJobCreation = null;

      const jobId = response.job_id;
      jobIdRef.current = jobId;

      beginPolling(jobId, uploadedFiles);
    } catch (startError) {
      activeJobCreation = null;
      console.error('Prediction start error:', startError);
      stopPolling();
      setStatus('error');
      setError({
        status: startError.status,
        message: startError.message,
      });
    }
  }, [beginPolling, pollJobStatus, stopPolling]);

  useEffect(() => {
    const filesFromState = location.state?.files;
    if (filesFromState) {
      setFiles(filesFromState);
      if (!requestStartedRef.current) {
        startPrediction(filesFromState);
      }
    } else {
      setStatus('idle');
    }

    return () => {
      stopPolling();
    };
  }, [location.state, startPrediction, stopPolling]);

  const handleStartNew = () => {
    stopPolling();
    requestStartedRef.current = false;
    jobIdRef.current = null;
    activeJobCreation = null;
    navigate('/predict');
  };

  const handleRetry = () => {
    if (files) {
      stopPolling();
      requestStartedRef.current = false;
      jobIdRef.current = null;
      activeJobCreation = null;
      startPrediction(files);
    }
  };

  const currentStageIndex = getStageIndex(backendStage);

  // Idle state - no active prediction
  if (status === 'idle') {
    return (
      <div className="min-h-screen bg-stone-50 flex items-center justify-center px-4">
        <div className="max-w-md w-full">
          <div className="bg-white rounded-xl border border-stone-200 p-8 text-center shadow-sm">
            <div className="w-16 h-16 bg-stone-100 rounded-full flex items-center justify-center mx-auto mb-6">
              <Loader2 className="text-stone-400" size={32} />
            </div>
            <h2 className="text-2xl font-semibold text-stone-900 mb-2">
              No analysis currently running
            </h2>
            <p className="text-stone-600 mb-6">
              Start a new segmentation to view processing status.
            </p>
            <button
              onClick={handleStartNew}
              className="w-full px-6 py-3 bg-teal-600 hover:bg-teal-700 text-white font-medium rounded-lg transition-colors"
            >
              Start New Analysis
            </button>
          </div>
        </div>
      </div>
    );
  }

  // Error state
  if (status === 'error') {
    return (
      <div className="min-h-screen bg-stone-50 flex items-center justify-center px-4">
        <div className="max-w-md w-full">
          <div className="bg-white rounded-xl border border-stone-200 p-8 shadow-sm">
            <div className="w-16 h-16 bg-red-50 rounded-full flex items-center justify-center mx-auto mb-6">
              <AlertCircle className="text-red-500" size={32} />
            </div>
            <h2 className="text-2xl font-semibold text-stone-900 mb-2">
              Analysis could not be completed
            </h2>
            <p className="text-stone-600 mb-6">
              {error?.message || 'An unexpected error occurred during processing.'}
            </p>
            <div className="flex gap-3">
              <button
                onClick={handleRetry}
                className="flex-1 px-6 py-3 bg-teal-600 hover:bg-teal-700 text-white font-medium rounded-lg transition-colors"
              >
                Try Again
              </button>
              <button
                onClick={handleStartNew}
                className="flex-1 px-6 py-3 bg-stone-200 hover:bg-stone-300 text-stone-700 font-medium rounded-lg transition-colors"
              >
                Return to MRI Upload
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Processing state
  return (
    <div className="min-h-screen bg-stone-50 py-12 px-4">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-3xl font-semibold text-stone-900 mb-2">
            MRI Analysis
          </h1>
          <p className="text-stone-600">
            Processing multi-modal MRI volumes
          </p>
          <p className="text-sm text-stone-500 mt-1">
            This may take a moment depending on the available GPU.
          </p>
        </div>

        {/* Processing Indicator */}
        <div className="bg-white rounded-xl border border-stone-200 p-8 mb-8 shadow-sm">
          <div className="flex items-center justify-center">
            <Loader2 className="text-teal-600 animate-spin" size={48} />
          </div>
          {backendMessage && (
            <p className="text-center text-sm text-stone-500 mt-4">
              {backendMessage}
            </p>
          )}
        </div>

        {/* Analysis Pipeline */}
        <div className="bg-white rounded-xl border border-stone-200 p-8 mb-8 shadow-sm">
          <h2 className="text-xl font-semibold text-stone-900 mb-6">
            Analysis Pipeline
          </h2>

          <div className="space-y-4">
            {pipelineSteps.map((step, index) => {
              const isCompleted = index < currentStageIndex;
              const isActive = index === currentStageIndex && backendStage !== 'completed';
              const isPending = index > currentStageIndex;

              return (
                <div
                  key={step.id}
                  className="flex items-start gap-4"
                >
                  <div className="flex-shrink-0">
                    <div className={`
                      w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium
                      ${isCompleted ? 'bg-teal-600 text-white' : ''}
                      ${isActive ? 'bg-teal-100 text-teal-700 ring-2 ring-teal-600' : ''}
                      ${isPending ? 'bg-stone-100 text-stone-400' : ''}
                    `}>
                      {isCompleted ? (
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                        </svg>
                      ) : isActive ? (
                        <Loader2 className="w-4 h-4 animate-spin" />
                      ) : (
                        <span>{String(index + 1).padStart(2, '0')}</span>
                      )}
                    </div>
                  </div>

                  <div className="flex-1 pt-1">
                    <h3 className={`
                      font-medium mb-1
                      ${isCompleted ? 'text-stone-900' : ''}
                      ${isActive ? 'text-teal-700' : ''}
                      ${isPending ? 'text-stone-400' : ''}
                    `}>
                      {step.label}
                      {isActive && (
                        <span className="ml-2 text-sm font-normal text-teal-600">
                          Running...
                        </span>
                      )}
                      {isCompleted && (
                        <span className="ml-2 text-sm font-normal text-teal-600">
                          Complete
                        </span>
                      )}
                      {isPending && (
                        <span className="ml-2 text-sm font-normal text-stone-400">
                          Waiting
                        </span>
                      )}
                    </h3>
                    <p className={`
                      text-sm
                      ${isCompleted ? 'text-stone-600' : ''}
                      ${isActive ? 'text-stone-700' : ''}
                      ${isPending ? 'text-stone-400' : ''}
                    `}>
                      {step.description}
                    </p>
                  </div>

                  {isActive && (
                    <div className="flex-shrink-0 pt-1">
                      <div className="w-2 h-2 bg-teal-600 rounded-full animate-pulse" />
                    </div>
                  )}
                </div>
              );
            })}

            {/* Completed step */}
            {backendStage === 'completed' && (
              <div className="flex items-start gap-4">
                <div className="flex-shrink-0">
                  <div className="w-8 h-8 rounded-full flex items-center justify-center bg-teal-600 text-white">
                    <CheckCircle2 className="w-4 h-4" />
                  </div>
                </div>
                <div className="flex-1 pt-1">
                  <h3 className="font-medium mb-1 text-stone-900">
                    Analysis Complete
                    <span className="ml-2 text-sm font-normal text-teal-600">
                      Complete
                    </span>
                  </h3>
                  <p className="text-sm text-stone-600">
                    Redirecting to results...
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Input Modalities */}
        {files && (
          <div className="bg-white rounded-xl border border-stone-200 p-8 shadow-sm">
            <h2 className="text-xl font-semibold text-stone-900 mb-6">
              Input Modalities
            </h2>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {[
                { key: 'flair', label: 'FLAIR' },
                { key: 't1', label: 'T1' },
                { key: 't1ce', label: 'T1ce' },
                { key: 't2', label: 'T2' },
              ].map((modality) => (
                <div
                  key={modality.key}
                  className="bg-stone-50 rounded-lg p-4 border border-stone-200"
                >
                  <p className="text-sm font-medium text-stone-900 mb-1">
                    {modality.label}
                  </p>
                  <p className="text-sm text-teal-600">
                    Ready
                  </p>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
