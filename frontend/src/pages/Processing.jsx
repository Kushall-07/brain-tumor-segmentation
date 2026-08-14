import { useState, useEffect, useRef, useCallback } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { Loader2, AlertCircle, CheckCircle2 } from 'lucide-react';
import predictionService from '../services/predictionService';
import { consumePendingUploadFiles } from '../utils/pendingUpload';

const POLL_INTERVAL_MS = 1000;

function buildUploadSessionKey(uploadedFiles) {
  if (!uploadedFiles) return '';
  return ['t1', 't1ce', 't2', 'flair', 'seg']
    .map((key) => {
      const file = uploadedFiles[key];
      return file ? `${key}:${file.name}:${file.size}:${file.lastModified}` : `${key}:`;
    })
    .join('|');
}

function appendFormDataFiles(formData, uploadedFiles) {
  const modalities = ['t1', 't1ce', 't2', 'flair'];
  modalities.forEach((modality) => {
    const file = uploadedFiles[modality];
    if (file) {
      formData.append(modality, file, file.name);
    }
  });

  const segFile = uploadedFiles.seg;
  if (segFile) {
    formData.append('seg', segFile, segFile.name);
  }
}

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
  const [status, setStatus] = useState('idle');
  const [error, setError] = useState(null);
  const [backendStage, setBackendStage] = useState('upload_received');
  const [backendMessage, setBackendMessage] = useState('');
  const [files, setFiles] = useState(null);
  const requestStartedRef = useRef(false);
  const pollIntervalRef = useRef(null);
  const jobIdRef = useRef(null);
  const uploadSessionKeyRef = useRef('');
  const activeJobCreationRef = useRef(null);

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
    const sessionKey = buildUploadSessionKey(uploadedFiles);
    if (!sessionKey) return;

    if (requestStartedRef.current && uploadSessionKeyRef.current === sessionKey && jobIdRef.current) {
      beginPolling(jobIdRef.current, uploadedFiles);
      return;
    }
    if (requestStartedRef.current && uploadSessionKeyRef.current === sessionKey) {
      return;
    }

    requestStartedRef.current = true;
    uploadSessionKeyRef.current = sessionKey;
    activeJobCreationRef.current = null;

    setStatus('processing');
    setError(null);
    setBackendStage('upload_received');
    setBackendMessage('MRI volumes received');

    const formData = new FormData();
    appendFormDataFiles(formData, uploadedFiles);

    try {
      const jobPromise = predictionService.startPrediction(formData, {
        checkpoint_path: 'outputs/exp_swinunetr_4class_et_fixed/checkpoints/best_mean_dice.pt',
        save_probabilities: false,
      });
      activeJobCreationRef.current = jobPromise;

      const response = await jobPromise;
      activeJobCreationRef.current = null;

      const jobId = response.job_id;
      jobIdRef.current = jobId;

      beginPolling(jobId, uploadedFiles);
    } catch (startError) {
      activeJobCreationRef.current = null;
      console.error('Prediction start error:', startError);
      stopPolling();
      setStatus('error');
      setError({
        status: startError.status,
        message: startError.message,
      });
    }
  }, [beginPolling, stopPolling]);

  useEffect(() => {
    const filesFromStore = consumePendingUploadFiles();
    const filesFromState = location.state?.files;
    const filesToProcess = filesFromStore || filesFromState;

    if (filesToProcess) {
      setFiles(filesToProcess);
      startPrediction(filesToProcess);
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
    uploadSessionKeyRef.current = '';
    activeJobCreationRef.current = null;
    navigate('/predict');
  };

  const handleRetry = () => {
    if (files) {
      stopPolling();
      requestStartedRef.current = false;
      jobIdRef.current = null;
      uploadSessionKeyRef.current = '';
      activeJobCreationRef.current = null;
      startPrediction(files);
    }
  };

  const currentStageIndex = getStageIndex(backendStage);

  if (status === 'idle') {
    return (
      <div className="min-h-screen bg-parchment flex items-center justify-center px-4 pt-28">
        <div className="max-w-md w-full">
          <div className="bg-parchment border border-sepia-border rounded-sm p-8 text-center">
            <div className="w-16 h-16 border border-sepia-border rounded-sm flex items-center justify-center mx-auto mb-6">
              <Loader2 className="text-sepia-muted" size={28} />
            </div>
            <h2 className="font-serif text-2xl font-semibold text-ink mb-2">
              No analysis currently running
            </h2>
            <p className="text-sepia-muted mb-6">
              Start a new segmentation to view processing status.
            </p>
            <button
              onClick={handleStartNew}
              className="w-full px-6 py-3 bg-arterial hover:bg-arterial-light text-parchment font-medium rounded-sm transition-colors"
            >
              Start New Analysis
            </button>
          </div>
        </div>
      </div>
    );
  }

  if (status === 'error') {
    return (
      <div className="min-h-screen bg-parchment flex items-center justify-center px-4 pt-28">
        <div className="max-w-md w-full">
          <div className="bg-parchment border border-sepia-border rounded-sm p-8">
            <div className="w-16 h-16 border border-arterial/30 rounded-sm flex items-center justify-center mx-auto mb-6">
              <AlertCircle className="text-arterial" size={28} />
            </div>
            <h2 className="font-serif text-2xl font-semibold text-ink mb-2">
              Analysis could not be completed
            </h2>
            <p className="text-sepia-muted mb-6">
              {error?.message || 'An unexpected error occurred during processing.'}
            </p>
            <div className="flex gap-3">
              <button
                onClick={handleRetry}
                className="flex-1 px-6 py-3 bg-arterial hover:bg-arterial-light text-parchment font-medium rounded-sm transition-colors"
              >
                Try Again
              </button>
              <button
                onClick={handleStartNew}
                className="flex-1 px-6 py-3 bg-parchment-dark hover:bg-sepia-border/30 text-ink font-medium rounded-sm border border-sepia-border transition-colors"
              >
                Return to MRI Upload
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-parchment py-12 px-4 pt-28">
      <div className="max-w-4xl mx-auto">
        <div className="text-center mb-12">
          <p className="atlas-label mb-2">Clinical Processing</p>
          <h1 className="font-serif text-3xl font-semibold text-ink mb-2">
            MRI Analysis
          </h1>
          <p className="text-ink-body">
            Processing multi-modal MRI volumes
          </p>
          <p className="text-sm text-sepia-muted mt-1">
            This may take a moment depending on the available GPU.
          </p>
        </div>

        <div className="bg-parchment border border-sepia-border rounded-sm p-8 mb-8">
          <div className="flex items-center justify-center">
            <Loader2 className="text-arterial animate-spin" size={48} strokeWidth={1.5} />
          </div>
          {backendMessage && (
            <p className="text-center text-sm text-sepia-muted mt-4 font-mono">
              {backendMessage}
            </p>
          )}
        </div>

        <div className="bg-parchment border border-sepia-border rounded-sm p-8 mb-8">
          <h2 className="font-serif text-xl font-semibold text-ink mb-6 pb-3 border-b border-sepia-border">
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
                      w-8 h-8 rounded-sm flex items-center justify-center text-sm font-mono font-medium border
                      ${isCompleted ? 'bg-annotation text-parchment border-annotation' : ''}
                      ${isActive ? 'bg-parchment-dark text-arterial border-arterial' : ''}
                      ${isPending ? 'bg-parchment-dark text-sepia-muted border-sepia-border' : ''}
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
                      ${isCompleted ? 'text-ink' : ''}
                      ${isActive ? 'text-arterial' : ''}
                      ${isPending ? 'text-sepia-muted' : ''}
                    `}>
                      {step.label}
                      {isActive && (
                        <span className="ml-2 text-sm font-normal text-sepia-muted">
                          Running...
                        </span>
                      )}
                      {isCompleted && (
                        <span className="ml-2 text-sm font-normal text-annotation">
                          Complete
                        </span>
                      )}
                      {isPending && (
                        <span className="ml-2 text-sm font-normal text-sepia-muted">
                          Waiting
                        </span>
                      )}
                    </h3>
                    <p className={`
                      text-sm
                      ${isCompleted ? 'text-ink-body' : ''}
                      ${isActive ? 'text-ink-body' : ''}
                      ${isPending ? 'text-sepia-muted' : ''}
                    `}>
                      {step.description}
                    </p>
                  </div>

                  {isActive && (
                    <div className="flex-shrink-0 pt-1">
                      <div className="w-2 h-2 bg-arterial rounded-full animate-pulse" />
                    </div>
                  )}
                </div>
              );
            })}

            {backendStage === 'completed' && (
              <div className="flex items-start gap-4">
                <div className="flex-shrink-0">
                  <div className="w-8 h-8 rounded-sm flex items-center justify-center bg-annotation text-parchment border border-annotation">
                    <CheckCircle2 className="w-4 h-4" />
                  </div>
                </div>
                <div className="flex-1 pt-1">
                  <h3 className="font-medium mb-1 text-ink">
                    Analysis Complete
                    <span className="ml-2 text-sm font-normal text-annotation">
                      Complete
                    </span>
                  </h3>
                  <p className="text-sm text-sepia-muted">
                    Redirecting to results...
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>

        {files && (
          <div className="bg-parchment border border-sepia-border rounded-sm p-8">
            <h2 className="font-serif text-xl font-semibold text-ink mb-6 pb-3 border-b border-sepia-border">
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
                  className="bg-parchment-dark rounded-sm p-4 border border-sepia-border"
                >
                  <p className="atlas-label text-ink mb-1">
                    {modality.label}
                  </p>
                  <p className="text-sm font-mono text-annotation">
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
