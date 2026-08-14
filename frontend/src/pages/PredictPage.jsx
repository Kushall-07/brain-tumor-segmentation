import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Brain, Loader2, AlertCircle, CheckCircle } from 'lucide-react';
import UploadCard from '../components/UploadCard';
import predictionService from '../services/predictionService';
import { setPendingUploadFiles } from '../utils/pendingUpload';

const MODALITIES = ['t1', 't1ce', 't2', 'flair'];

const PredictPage = () => {
  const navigate = useNavigate();

  const [files, setFiles] = useState({
    t1: null,
    t1ce: null,
    t2: null,
    flair: null,
    seg: null,
  });

  const [errors, setErrors] = useState({});
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [uploadStatus, setUploadStatus] = useState('');
  const [apiError, setApiError] = useState(null);

  const clearModalityError = (modality) => {
    setErrors((prev) => {
      if (!(modality in prev)) {
        return prev;
      }
      const next = { ...prev };
      delete next[modality];
      return next;
    });
  };

  const handleFileSelect = (modality, file) => {
    setFiles(prev => ({ ...prev, [modality]: file }));
    clearModalityError(modality);
    setApiError(null);
  };

  const handleFileRemove = (modality) => {
    setFiles(prev => ({ ...prev, [modality]: null }));
    clearModalityError(modality);
  };

  const validateFiles = () => {
    const newErrors = {};
    let hasError = false;

    MODALITIES.forEach(modality => {
      if (!files[modality]) {
        newErrors[modality] = `${modality.toUpperCase()} is required`;
        hasError = true;
      }
    });

    const fileHashes = new Map();
    MODALITIES.forEach(modality => {
      if (files[modality]) {
        const file = files[modality];
        const hash = `${file.name}-${file.size}`;
        if (fileHashes.has(hash)) {
          const duplicateModality = fileHashes.get(hash);
          newErrors[modality] = `Duplicate of ${duplicateModality.toUpperCase()}`;
          newErrors[duplicateModality] = `Duplicate of ${modality.toUpperCase()}`;
          hasError = true;
        } else {
          fileHashes.set(hash, modality);
        }
      }
    });

    setErrors(newErrors);
    return !hasError;
  };

  const handleSubmit = async () => {
    if (!validateFiles()) {
      return;
    }

    setIsSubmitting(true);
    setApiError(null);
    setUploadStatus('Preparing files...');

    setPendingUploadFiles(files);

    navigate('/processing', {
      state: {
        files,
      },
    });
  };

  const allFilesSelected = MODALITIES.every(modality => files[modality] !== null);
  const visibleErrors = Object.entries(errors).filter(
    ([, message]) => typeof message === 'string' && message.trim().length > 0
  );
  const hasErrors = visibleErrors.length > 0;

  const getErrorMessage = (status) => {
    switch (status) {
      case 400:
        return 'Invalid request - please check your files';
      case 404:
        return 'Server resource not found - checkpoint may be missing';
      case 422:
        return 'Validation failed - file format or content issue';
      case 500:
        return 'Server error - please try again later';
      case 0:
        return 'Network error - unable to connect to server';
      default:
        return 'An unexpected error occurred';
    }
  };

  return (
    <div className="min-h-screen bg-parchment py-12 px-4 sm:px-6 lg:px-8 pt-28">
      <div className="max-w-4xl mx-auto">
        <div className="text-center mb-10">
          <div className="flex items-center justify-center mb-4">
            <Brain className="text-arterial" size={40} strokeWidth={1.5} />
          </div>
          <h1 className="font-serif text-3xl font-semibold text-ink mb-2">
            Brain Tumor Segmentation
          </h1>
          <p className="text-ink-body">
            Upload MRI modalities for AI-powered tumor segmentation
          </p>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
          {MODALITIES.map(modality => (
            <UploadCard
              key={modality}
              modality={modality}
              selectedFile={files[modality]}
              onFileSelect={(file) => handleFileSelect(modality, file)}
              onRemove={() => handleFileRemove(modality)}
              disabled={isSubmitting}
              error={errors[modality]}
            />
          ))}
        </div>

        {/* Optional Ground Truth */}
        <div className="mb-8">
          <p className="atlas-label mb-3 text-center">Optional — Ground Truth Segmentation</p>
          <div className="max-w-xs mx-auto">
            <UploadCard
              modality="seg"
              selectedFile={files.seg}
              onFileSelect={(file) => handleFileSelect('seg', file)}
              onRemove={() => handleFileRemove('seg')}
              disabled={isSubmitting}
              error={errors.seg}
            />
          </div>
          <p className="text-xs text-sepia-muted text-center mt-2 max-w-md mx-auto">
            Provide a labeled BraTS-format segmentation mask to enable validation metrics and
            ground-truth comparison on the Results page.
          </p>
        </div>

        {hasErrors && !isSubmitting && (
          <div className="mb-6 p-4 bg-parchment-dark border border-arterial/30 rounded-sm">
            <div className="flex items-start">
              <AlertCircle className="text-arterial mt-0.5 mr-2 flex-shrink-0" size={20} />
              <div>
                <p className="font-medium text-arterial">Please fix the following errors:</p>
                <ul className="mt-2 text-sm text-sepia-muted list-disc list-inside">
                  {visibleErrors.map(([modality, error]) => (
                    <li key={modality}>{error}</li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        )}

        {apiError && (
          <div className="mb-6 p-4 bg-parchment-dark border border-arterial/30 rounded-sm">
            <div className="flex items-start">
              <AlertCircle className="text-arterial mt-0.5 mr-2 flex-shrink-0" size={20} />
              <div>
                <p className="font-medium text-arterial">
                  {getErrorMessage(apiError.status)}
                </p>
                <p className="mt-1 text-sm text-sepia-muted">{apiError.message}</p>
              </div>
            </div>
          </div>
        )}

        {isSubmitting && (
          <div className="mb-6 p-6 bg-parchment-dark border border-sepia-border rounded-sm">
            <div className="flex items-center justify-center space-x-3">
              <Loader2 className="animate-spin text-annotation" size={24} />
              <p className="text-ink font-medium">{uploadStatus}</p>
            </div>
          </div>
        )}

        <div className="flex justify-center">
          <button
            onClick={handleSubmit}
            disabled={!allFilesSelected || isSubmitting}
            className={`
              px-8 py-3 rounded-sm font-medium transition-colors duration-200
              ${allFilesSelected && !isSubmitting
                ? 'bg-arterial hover:bg-arterial-light text-parchment'
                : 'bg-sepia-border text-sepia-muted cursor-not-allowed'
              }
              ${isSubmitting ? 'opacity-75' : ''}
            `}
          >
            {isSubmitting ? (
              <span className="flex items-center space-x-2">
                <Loader2 className="animate-spin" size={20} />
                <span>Processing...</span>
              </span>
            ) : (
              <span className="flex items-center space-x-2">
                <CheckCircle size={20} strokeWidth={1.5} />
                <span>Run Segmentation</span>
              </span>
            )}
          </button>
        </div>

        <div className="mt-8 p-4 bg-parchment-dark border border-sepia-border rounded-sm">
          <div className="flex items-start">
            <CheckCircle className="text-annotation mt-0.5 mr-2 flex-shrink-0" size={20} strokeWidth={1.5} />
            <div className="text-sm text-ink-body">
              <p className="font-medium text-ink mb-1">Supported file formats:</p>
              <p className="font-mono text-xs">.nii (uncompressed) and .nii.gz (compressed)</p>
              <p className="mt-2 font-medium text-ink mb-1">Required modalities:</p>
              <p>T1, T1ce, T2, and FLAIR MRI scans</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PredictPage;
