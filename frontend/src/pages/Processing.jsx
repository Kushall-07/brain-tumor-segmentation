import { useState, useEffect, useRef } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import ProcessingProgress from '../components/ProcessingProgress';
import { api } from '../services/api';

const processingSteps = [
  { id: 'upload', label: 'Uploading Files', description: 'Uploading MRI scans to server' },
  { id: 'preprocess', label: 'Preprocessing', description: 'Normalizing, resizing, and skull-stripping' },
  { id: 'inference', label: 'Model Inference', description: 'Running SwinUNETR model inference' },
  { id: 'postprocess', label: 'Post-processing', description: 'Applying CRF refinement and cleanup' },
  { id: 'visualize', label: 'Generating Visualizations', description: 'Creating 3D views and overlays' },
  { id: 'complete', label: 'Complete', description: 'Results ready for viewing' }
];

export default function Processing() {
  const navigate = useNavigate();
  const location = useLocation();
  const [jobId, setJobId] = useState(null);
  const [error, setError] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const filesRef = useRef(location.state?.files || null);

  useEffect(() => {
    if (filesRef.current && !jobId && !isUploading) {
      startProcessing();
    }
  }, [jobId, isUploading]);

  const startProcessing = async () => {
    setIsUploading(true);
    setError(null);

    try {
      const response = await api.createJob(filesRef.current);
      setJobId(response.jobId);
    } catch (err) {
      setError(err.message || 'Failed to start processing');
      setIsUploading(false);
    }
  };

  const handleComplete = (results) => {
    navigate('/results', { state: { jobId, results } });
  };

  const handleRetry = () => {
    setError(null);
    setJobId(null);
    setIsUploading(false);
    startProcessing();
  };

  const handleCancel = () => {
    navigate('/');
  };

  if (isUploading && !jobId) {
    return (
      <div className="max-w-2xl mx-auto px-4 py-12 text-center">
        <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-primary-500/20 mb-6">
          <svg className="w-8 h-8 text-primary-500 animate-spin" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
          </svg>
        </div>
        <h2 className="text-2xl font-bold text-white mb-2">Uploading Files...</h2>
        <p className="text-slate-400">Please wait while we upload your MRI scans</p>
      </div>
    );
  }

  if (error && !jobId) {
    return (
      <div className="max-w-2xl mx-auto px-4 py-12">
        <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-6">
          <div className="flex items-start gap-3">
            <svg className="w-5 h-5 mt-0.5 flex-shrink-0 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
            <div>
              <h3 className="font-semibold text-red-400 mb-1">Upload Failed</h3>
              <p className="text-slate-300 mb-4">{error}</p>
              <div className="flex gap-3">
                <button
                  onClick={handleRetry}
                  className="px-4 py-2 bg-primary-500 hover:bg-primary-600 text-white font-medium rounded-lg transition-colors"
                >
                  Try Again
                </button>
                <button
                  onClick={handleCancel}
                  className="px-4 py-2 bg-slate-700 hover:bg-slate-600 text-white font-medium rounded-lg transition-colors"
                >
                  Go Back
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto px-4 py-12">
      <div className="text-center mb-8">
        <h1 className="text-3xl md:text-4xl font-bold text-white mb-2">Processing Pipeline</h1>
        <p className="text-slate-400">Job ID: <span className="font-mono text-cyan-400">{jobId?.slice(0, 8)}...</span></p>
      </div>

      <ProcessingProgress jobId={jobId} onComplete={handleComplete} />
    </div>
  );
}