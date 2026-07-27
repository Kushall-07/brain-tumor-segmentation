import { useEffect, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import toast from 'react-hot-toast';
import { CheckCircle, Download, ArrowLeft, FileText, AlertCircle } from 'lucide-react';
import predictionService from '../services/predictionService';
import NiiVueViewer from '../components/NiiVueViewer';

export default function Results() {
  const navigate = useNavigate();
  const location = useLocation();
  const predictionState = location.state;
  const [isDownloading, setIsDownloading] = useState(false);

  useEffect(() => {
    if (!predictionState?.result) {
      toast.error('No prediction data found', {
        position: 'top-center',
      });
      navigate('/predict', { replace: true });
    } else {
      toast.success('Prediction completed successfully!', {
        position: 'top-center',
        duration: 4000,
      });
    }
  }, [navigate, predictionState]);

  if (!predictionState?.result) {
    return null;
  }

  const { result } = predictionState;

  const handleDownload = async () => {
    setIsDownloading(true);
    try {
      // Extract relative path from the full mask_path
      // Backend returns full path like "outputs/predictions/session_id/case_id_pred.nii.gz"
      // We need to extract the part after "outputs/predictions/"
      const relativePath = result.mask_path.replace('outputs/predictions/', '');
      
      // Download the file as a blob
      const blob = await predictionService.downloadPrediction(relativePath);
      
      // Create a download link and trigger it
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      
      // Extract filename from the path
      const filename = result.mask_path.split('/').pop();
      link.download = filename;
      
      document.body.appendChild(link);
      link.click();
      
      // Cleanup
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);
      
      toast.success('Download completed successfully', {
        position: 'top-center',
      });
      
    } catch (error) {
      console.error('Download error:', error);
      let errorMessage = 'Download failed';
      
      if (error.status === 404) {
        errorMessage = 'File not found on server';
      } else if (error.status === 403) {
        errorMessage = 'Access denied';
      } else if (error.status === 0) {
        errorMessage = 'Network error - unable to connect to server';
      } else {
        errorMessage = error.message || 'Download failed';
      }
      
      toast.error(errorMessage, {
        position: 'top-center',
      });
    } finally {
      setIsDownloading(false);
    }
  };

  const handlePredictAnother = () => {
    navigate('/predict');
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 to-emerald-100 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="text-center mb-10">
          <div className="flex items-center justify-center mb-4">
            <CheckCircle className="text-green-600" size={64} />
          </div>
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
            Prediction Successful
          </h1>
          <p className="text-gray-600 text-lg">
            Brain tumor segmentation completed
          </p>
        </div>

        {/* Results Card */}
        <div className="bg-white rounded-xl shadow-lg p-6 sm:p-8 mb-6">
          <h2 className="text-2xl font-semibold text-gray-900 mb-6 flex items-center">
            <FileText className="mr-2 text-indigo-600" size={24} />
            Prediction Details
          </h2>

          <div className="space-y-4">
            {/* Case ID */}
            <div className="bg-gray-50 rounded-lg p-4">
              <p className="text-sm font-medium text-gray-500 mb-1">Case ID</p>
              <p className="text-lg font-semibold text-gray-900">{result.case_id}</p>
            </div>

            {/* Mask Path */}
            <div className="bg-gray-50 rounded-lg p-4">
              <p className="text-sm font-medium text-gray-500 mb-1">Segmentation Mask</p>
              <p className="text-sm text-gray-900 break-all font-mono">{result.mask_path}</p>
            </div>

            {/* Probability Path */}
            {result.probability_path && (
              <div className="bg-gray-50 rounded-lg p-4">
                <p className="text-sm font-medium text-gray-500 mb-1">Probability Map</p>
                <p className="text-sm text-gray-900 break-all font-mono">{result.probability_path}</p>
              </div>
            )}

            {/* Info Note */}
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
              <div className="flex items-start">
                <AlertCircle className="text-blue-500 mt-0.5 mr-2 flex-shrink-0" size={20} />
                <div className="text-sm text-blue-800">
                  <p className="font-medium mb-1">Note</p>
                  <p>
                    The segmentation mask has been saved to the outputs directory. 
                    Use the download button below to retrieve the file.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* MRI Visualization Card */}
        <div className="bg-white rounded-xl shadow-lg p-6 sm:p-8 mb-6">
          <h2 className="text-2xl font-semibold text-gray-900 mb-6 flex items-center">
            <FileText className="mr-2 text-indigo-600" size={24} />
            MRI Visualization
          </h2>

          <NiiVueViewer
            mriPath={result.mri_path}
            maskPath={result.mask_path}
          />
        </div>

        {/* Action Buttons */}
        <div className="flex flex-col sm:flex-row gap-4 justify-center">
          <button
            onClick={handleDownload}
            disabled={isDownloading}
            className={`
              px-6 py-3 rounded-lg font-semibold text-white transition-all duration-200
              flex items-center justify-center space-x-2
              ${isDownloading
                ? 'bg-gray-400 cursor-not-allowed'
                : 'bg-indigo-600 hover:bg-indigo-700 shadow-md hover:shadow-lg'
              }
            `}
          >
            {isDownloading ? (
              <>
                <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
                <span>Downloading...</span>
              </>
            ) : (
              <>
                <Download size={20} />
                <span>Download Prediction</span>
              </>
            )}
          </button>

          <button
            onClick={handlePredictAnother}
            className="px-6 py-3 rounded-lg font-semibold text-white bg-green-600 hover:bg-green-700 shadow-md hover:shadow-lg transition-all duration-200 flex items-center justify-center space-x-2"
          >
            <ArrowLeft size={20} />
            <span>Predict Another Patient</span>
          </button>
        </div>

        {/* Footer Info */}
        <div className="mt-8 text-center text-sm text-gray-500">
          <p>Brain Tumor Segmentation AI System</p>
          <p className="mt-1">Powered by 3D U-Net and MONAI</p>
        </div>
      </div>
    </div>
  );
}
