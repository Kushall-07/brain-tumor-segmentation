import { useState, useEffect } from 'react';

const processingSteps = [
  { id: 'upload', label: 'Uploading Files', description: 'Uploading MRI scans to server' },
  { id: 'preprocess', label: 'Preprocessing', description: 'Normalizing, resizing, and skull-stripping' },
  { id: 'inference', label: 'Model Inference', description: 'Running SwinUNETR model inference' },
  { id: 'postprocess', label: 'Post-processing', description: 'Applying CRF refinement and cleanup' },
  { id: 'visualize', label: 'Generating Visualizations', description: 'Creating 3D views and overlays' },
  { id: 'complete', label: 'Complete', description: 'Results ready for viewing' }
];

export default function ProcessingProgress({ jobId, onComplete }) {
  const [currentStep, setCurrentStep] = useState(0);
  const [progress, setProgress] = useState(0);
  const [logs, setLogs] = useState([]);
  const [isComplete, setIsComplete] = useState(false);
  const [error, setError] = useState(null);
  const [results, setResults] = useState(null);

  useEffect(() => {
    if (!jobId) return;

    const eventSource = new EventSource(`/api/jobs/${jobId}/stream`);

    eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data);

      if (data.type === 'progress') {
        setProgress(data.progress);
        setCurrentStep(data.stepIndex);
        if (data.log) {
          setLogs(prev => [...prev, { ...data.log, timestamp: new Date() }]);
        }
      } else if (data.type === 'complete') {
        setIsComplete(true);
        setProgress(100);
        setCurrentStep(processingSteps.length - 1);
        setResults(data.results);
        eventSource.close();
        if (onComplete) onComplete(data.results);
      } else if (data.type === 'error') {
        setError(data.message);
        eventSource.close();
      }
    };

    eventSource.onerror = () => {
      setError('Connection lost. Please refresh the page.');
      eventSource.close();
    };

    return () => eventSource.close();
  }, [jobId, onComplete]);

  // Simulated progress for demo when no jobId
  useEffect(() => {
    if (jobId) return;

    const steps = [
      { step: 0, progress: 10, delay: 500 },
      { step: 1, progress: 30, delay: 2000 },
      { step: 2, progress: 60, delay: 4000 },
      { step: 3, progress: 80, delay: 2000 },
      { step: 4, progress: 95, delay: 2000 },
      { step: 5, progress: 100, delay: 1000 }
    ];

    let currentIndex = 0;
    const interval = setInterval(() => {
      if (currentIndex >= steps.length) {
        setIsComplete(true);
        clearInterval(interval);
        return;
      }

      const step = steps[currentIndex];
      setCurrentStep(step.step);
      setProgress(step.progress);
      setLogs(prev => [...prev, {
        message: processingSteps[step.step].description,
        type: 'info',
        timestamp: new Date()
      }]);
      currentIndex++;
    }, steps[currentIndex]?.delay || 1000);

    return () => clearInterval(interval);
  }, [jobId]);

  const getStepStatus = (index) => {
    if (index < currentStep) return 'completed';
    if (index === currentStep) return 'active';
    return 'pending';
  };

  const formatTime = (date) => {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  };

  return (
    <div className="space-y-6">
      <div className="bg-slate-800/50 rounded-xl border border-slate-700 p-6">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-xl font-semibold text-white">Processing Pipeline</h2>
          <div className="text-right">
            <div className="text-3xl font-bold text-primary-500">{progress}%</div>
            <div className="text-sm text-slate-400">Overall Progress</div>
          </div>
        </div>

        <div className="h-3 bg-slate-700 rounded-full overflow-hidden mb-6">
          <div
            className="h-full bg-gradient-to-r from-primary-500 to-primary-400 transition-all duration-500 ease-out"
            style={{ width: `${progress}%` }}
          />
        </div>

        <div className="space-y-4">
          {processingSteps.map((step, index) => {
            const status = getStepStatus(index);
            const isActive = status === 'active';
            const isCompleted = status === 'completed';

            return (
              <div
                key={step.id}
                className={`flex items-center gap-4 p-4 rounded-xl transition-all duration-300 ${
                  isActive ? 'bg-primary-500/10 border border-primary-500/30' : 'bg-slate-800/50 border border-slate-700'
                }`}
              >
                <div
                  className={`flex-shrink-0 w-10 h-10 rounded-full flex items-center justify-center transition-all duration-300 ${
                    isCompleted
                      ? 'bg-primary-500 text-white'
                      : isActive
                        ? 'bg-primary-500 text-white animate-pulse ring-4 ring-primary-500/30'
                        : 'bg-slate-700 text-slate-500'
                  }`}
                >
                  {isCompleted ? (
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                    </svg>
                  ) : (
                    <span className="font-medium">{index + 1}</span>
                  )}
                </div>

                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <h3 className={`font-medium ${isCompleted ? 'text-white' : isActive ? 'text-primary-400' : 'text-slate-300'}`}>
                      {step.label}
                    </h3>
                    {isActive && (
                      <span className="px-2 py-0.5 text-xs font-medium bg-primary-500/20 text-primary-400 rounded-full animate-pulse">
                        Processing
                      </span>
                    )}
                  </div>
                  <p className="text-sm text-slate-500 mt-1">{step.description}</p>
                </div>

                {isCompleted && (
                  <svg className="w-5 h-5 text-primary-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {logs.length > 0 && (
        <div className="bg-slate-800/50 rounded-xl border border-slate-700 p-6">
          <h3 className="text-lg font-semibold text-white mb-4">Processing Logs</h3>
          <div className="space-y-2 max-h-64 overflow-y-auto font-mono text-sm">
            {logs.map((log, index) => (
              <div
                key={index}
                className={`flex items-start gap-2 p-2 rounded-lg ${
                  log.type === 'error' ? 'bg-red-500/10 text-red-400' :
                  log.type === 'warning' ? 'bg-yellow-500/10 text-yellow-400' :
                  'bg-slate-700/50 text-slate-300'
                }`}
              >
                <span className="text-slate-500 whitespace-nowrap">{formatTime(log.timestamp)}</span>
                <span className="flex-1 break-all">{log.message}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {error && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-6 text-red-400">
          <div className="flex items-start gap-3">
            <svg className="w-5 h-5 mt-0.5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
            <div>
              <h3 className="font-semibold">Processing Error</h3>
              <p className="mt-1">{error}</p>
            </div>
          </div>
        </div>
      )}

      {isComplete && !error && (
        <div className="bg-primary-500/10 border border-primary-500/30 rounded-xl p-6 text-center">
          <div className="flex items-center justify-center gap-3 mb-4">
            <div className="w-16 h-16 rounded-full bg-primary-500/20 flex items-center justify-center">
              <svg className="w-8 h-8 text-primary-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
              </svg>
            </div>
          </div>
          <h3 className="text-2xl font-bold text-white mb-2">Processing Complete!</h3>
          <p className="text-slate-300 mb-6">Your brain tumor segmentation results are ready.</p>
          <button
            onClick={() => onComplete?.(results)}
            className="px-8 py-3 bg-primary-500 hover:bg-primary-600 text-white font-medium rounded-lg transition-colors"
          >
            View Results
          </button>
        </div>
      )}
    </div>
  );
}