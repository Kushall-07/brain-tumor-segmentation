import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Brain, Loader2, AlertCircle, CheckCircle } from 'lucide-react';
import UploadCard from '../components/UploadCard';
import predictionService from '../services/predictionService';

const MODALITIES = ['t1', 't1ce', 't2', 'flair'];

const PredictPage = () => {
  const navigate = useNavigate();
  
  const [files, setFiles] = useState({
    t1: null,
    t1ce: null,
    t2: null,
    flair: null,
  });
  
  const [errors, setErrors] = useState({});
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [uploadStatus, setUploadStatus] = useState('');
  const [apiError, setApiError] = useState(null);

  const handleFileSelect = (modality, file) => {
    setFiles(prev => ({ ...prev, [modality]: file }));
    setErrors(prev => ({ ...prev, [modality]: null }));
    setApiError(null);
  };

  const handleFileRemove = (modality) => {
    setFiles(prev => ({ ...prev, [modality]: null }));
    setErrors(prev => ({ ...prev, [modality]: null }));
  };

  const validateFiles = () => {
    const newErrors = {};
    let hasError = false;

    // Check if all modalities are selected
    MODALITIES.forEach(modality => {
      if (!files[modality]) {
        newErrors[modality] = `${modality.toUpperCase()} is required`;
        hasError = true;
      }
    });

    // Check for duplicate files
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
    setUploadStatus('Uploading MRI files...');

    const formData = new FormData();
    MODALITIES.forEach(modality => {
      formData.append(modality, files[modality]);
    });

    try {
      setUploadStatus('Running segmentation model...');
      
      const response = await predictionService.uploadPrediction(formData, {
        checkpoint_path: 'outputs/checkpoints/best.pt',
        save_probabilities: false,
      });

      setUploadStatus('Receiving prediction...');
      
      // Navigate to results page with response data
      navigate('/results', { 
        state: { 
          result: response.result,
          files: files 
        } 
      });

    } catch (error) {
      console.error('Prediction error:', error);
      setApiError({
        status: error.status,
        message: error.message,
      });
      setUploadStatus('');
    } finally {
      setIsSubmitting(false);
    }
  };

  const allFilesSelected = MODALITIES.every(modality => files[modality] !== null);
  const hasErrors = Object.keys(errors).length > 0;

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
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="text-center mb-10">
          <div className="flex items-center justify-center mb-4">
            <Brain className="text-indigo-600" size={48} />
          </div>
          <h1 className="text-3xl font-bold text-gray-900 mb-2">
            Brain Tumor Segmentation
          </h1>
          <p className="text-gray-600">
            Upload MRI modalities for AI-powered tumor segmentation
          </p>
        </div>

        {/* Upload Cards Grid */}
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

        {/* Validation Error Summary */}
        {hasErrors && !isSubmitting && (
          <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg">
            <div className="flex items-start">
              <AlertCircle className="text-red-500 mt-0.5 mr-2" size={20} />
              <div>
                <p className="font-medium text-red-800">Please fix the following errors:</p>
                <ul className="mt-2 text-sm text-red-700 list-disc list-inside">
                  {Object.entries(errors).map(([modality, error]) => (
                    <li key={modality}>{error}</li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        )}

        {/* API Error */}
        {apiError && (
          <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg">
            <div className="flex items-start">
              <AlertCircle className="text-red-500 mt-0.5 mr-2" size={20} />
              <div>
                <p className="font-medium text-red-800">
                  {getErrorMessage(apiError.status)}
                </p>
                <p className="mt-1 text-sm text-red-700">{apiError.message}</p>
              </div>
            </div>
          </div>
        )}

        {/* Loading Status */}
        {isSubmitting && (
          <div className="mb-6 p-6 bg-white border border-gray-200 rounded-lg shadow-sm">
            <div className="flex items-center justify-center space-x-3">
              <Loader2 className="animate-spin text-indigo-600" size={24} />
              <p className="text-gray-700 font-medium">{uploadStatus}</p>
            </div>
          </div>
        )}

        {/* Predict Button */}
        <div className="flex justify-center">
          <button
            onClick={handleSubmit}
            disabled={!allFilesSelected || isSubmitting}
            className={`
              px-8 py-3 rounded-lg font-semibold text-white transition-all duration-200
              ${allFilesSelected && !isSubmitting
                ? 'bg-indigo-600 hover:bg-indigo-700 shadow-md hover:shadow-lg'
                : 'bg-gray-400 cursor-not-allowed'
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
                <CheckCircle size={20} />
                <span>Run Segmentation</span>
              </span>
            )}
          </button>
        </div>

        {/* Info Card */}
        <div className="mt-8 p-4 bg-blue-50 border border-blue-200 rounded-lg">
          <div className="flex items-start">
            <CheckCircle className="text-blue-500 mt-0.5 mr-2" size={20} />
            <div className="text-sm text-blue-800">
              <p className="font-medium mb-1">Supported file formats:</p>
              <p>.nii (uncompressed) and .nii.gz (compressed)</p>
              <p className="mt-2 font-medium mb-1">Required modalities:</p>
              <p>T1, T1ce, T2, and FLAIR MRI scans</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PredictPage;
