import { useEffect, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import toast from 'react-hot-toast';
import { CheckCircle, Download, ArrowLeft, FileText, AlertCircle, Activity } from 'lucide-react';
import predictionService from '../services/predictionService';
import NiiVueViewer from '../components/NiiVueViewer';

export default function Results() {
  const navigate = useNavigate();
  const location = useLocation();
  const predictionState = location.state;
  const [isDownloading, setIsDownloading] = useState(false);
  const [result, setResult] = useState(null);
  const [visibleClasses, setVisibleClasses] = useState([1, 2, 3]);
  const [classAnalysis, setClassAnalysis] = useState(null);
  const [individualClassAnalysis, setIndividualClassAnalysis] = useState(null);
  const [loadingAnalysis, setLoadingAnalysis] = useState(false);
  const [expandedAccordions, setExpandedAccordions] = useState({ 1: false, 2: false, 3: false });

  // Load prediction result from navigation state or sessionStorage
  useEffect(() => {
    let predictionResult = predictionState?.result;

    if (!predictionResult) {
      try {
        const storedPrediction = sessionStorage.getItem('brainTumorLatestPrediction');
        if (storedPrediction) {
          predictionResult = JSON.parse(storedPrediction);
        }
      } catch (error) {
        console.error('Failed to parse stored prediction from sessionStorage:', error);
      }
    }

    if (!predictionResult) {
      toast.error('No prediction data found', {
        position: 'top-center',
      });
      navigate('/predict', { replace: true });
    } else {
      setResult(predictionResult);
      toast.success('Prediction completed successfully!', {
        position: 'top-center',
        duration: 4000,
      });
    }
  }, [navigate, predictionState]);

  // Load initial total tumor analysis for WT (classes 1,2,3) - MUST be before early return
  useEffect(() => {
    if (result && result.mask_path) {
      setLoadingAnalysis(true);
      // Convert Windows backslashes to forward slashes and extract relative path
      const normalizedMaskPath = result.mask_path.replace(/\\/g, '/');
      const relativeMaskPath = normalizedMaskPath.replace(/.*outputs\/predictions\//, '');
      
      // Fetch total tumor analysis for WT (all classes)
      predictionService.getClassAnalysis(relativeMaskPath, [1, 2, 3])
        .then(response => {
          setClassAnalysis(response);
        })
        .catch(error => {
          console.error('Failed to get total tumor analysis:', error);
          setClassAnalysis(null);
        });
      
      // Fetch individual class analysis for all classes (for dynamic display)
      predictionService.getIndividualClassAnalysis(relativeMaskPath)
        .then(response => {
          // Extract class_analysis from the response
          const classAnalysisData = response.class_analysis || {};
          setIndividualClassAnalysis(classAnalysisData);
        })
        .catch(error => {
          console.error('Failed to get individual class analysis:', error);
          setIndividualClassAnalysis(null);
        })
        .finally(() => setLoadingAnalysis(false));
    }
  }, [result]);

  const handleClassChange = async (newVisibleClasses) => {
    setVisibleClasses(newVisibleClasses);
    
    // Class analysis is no longer updated on class change
    // Total tumor analysis remains constant (WT = classes 1,2,3)
    // Individual class analysis is already loaded and displayed based on visibleClasses
  };

  const toggleAccordion = (classId) => {
    setExpandedAccordions(prev => ({
      ...prev,
      [classId]: !prev[classId]
    }));
  };

  const handlePredictAnother = () => {
    navigate('/predict');
  };

  const handleDownload = async () => {
    setIsDownloading(true);
    try {
      // Convert Windows backslashes to forward slashes and extract relative path
      const normalizedPath = result.mask_path.replace(/\\/g, '/');
      const relativePath = normalizedPath.replace(/.*outputs\/predictions\//, '');
      const blob = await predictionService.downloadPrediction(relativePath);
      
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = result.mask_path.split(/[/\\]/).pop();
      
      document.body.appendChild(link);
      link.click();
      
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

  // Early return MUST be after all hooks
  if (!result) {
    return null;
  }

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

        {/* Tumor Analysis Card */}
        <div className="bg-white rounded-xl shadow-lg p-6 sm:p-8 mb-6">
          <h2 className="text-2xl font-semibold text-gray-900 mb-6 flex items-center">
            <Activity className="mr-2 text-indigo-600" size={24} />
            Tumor Analysis
          </h2>

          {/* Total Tumor Analysis - Always Visible */}
          {classAnalysis && (
            <div className="space-y-4 mb-6">
              <div className="bg-gradient-to-br from-blue-50 to-cyan-50 rounded-lg p-5 border border-blue-100">
                <p className="text-sm font-medium text-gray-600 mb-1">Total Tumor (WT)</p>
                <p className="text-3xl font-bold text-gray-900 mb-3">
                  {parseFloat(classAnalysis.volume_cm3).toFixed(2)} cm<sup>3</sup>
                </p>
                <div className="grid grid-cols-3 gap-4">
                  <div className="text-center">
                    <p className="text-xs text-gray-500 mb-1">Height</p>
                    <p className="text-xl font-bold text-gray-900">
                      {parseFloat(classAnalysis.dimensions_mm.height_mm).toFixed(2)} mm
                    </p>
                  </div>
                  <div className="text-center">
                    <p className="text-xs text-gray-500 mb-1">Width</p>
                    <p className="text-xl font-bold text-gray-900">
                      {parseFloat(classAnalysis.dimensions_mm.width_mm).toFixed(2)} mm
                    </p>
                  </div>
                  <div className="text-center">
                    <p className="text-xs text-gray-500 mb-1">Length</p>
                    <p className="text-xl font-bold text-gray-900">
                      {parseFloat(classAnalysis.dimensions_mm.length_mm).toFixed(2)} mm
                    </p>
                  </div>
                </div>
              </div>

              {/* Individual Class Analysis - Always visible as collapsible accordions */}
              {individualClassAnalysis && (
                <div className="space-y-3">
                  <p className="text-sm font-medium text-gray-500">Individual Class Analysis</p>
                  
                  {/* NCR/NET Accordion */}
                  <div className="bg-red-50 rounded-lg border border-red-100 overflow-hidden">
                    <button
                      type="button"
                      onClick={() => toggleAccordion(1)}
                      className="w-full flex items-center justify-between p-4 text-left hover:bg-red-100 transition-colors"
                    >
                      <div className="flex items-center gap-2">
                        <span className="w-3 h-3 rounded-full bg-red-500" />
                        <p className="text-sm font-semibold text-red-900">NCR/NET</p>
                      </div>
                      <span className="text-gray-500">
                        {expandedAccordions[1] ? '▲' : '▼'}
                      </span>
                    </button>
                    {expandedAccordions[1] && individualClassAnalysis['1'] && (
                      <div className="p-4 pt-0 border-t border-red-200">
                        <div className="grid grid-cols-4 gap-3">
                          <div>
                            <p className="text-xs text-gray-600">Volume</p>
                            <p className="text-sm font-bold text-gray-900">
                              {parseFloat(individualClassAnalysis['1'].volume_cm3).toFixed(2)} cm³
                            </p>
                          </div>
                          <div>
                            <p className="text-xs text-gray-600">Height</p>
                            <p className="text-sm font-bold text-gray-900">
                              {parseFloat(individualClassAnalysis['1'].dimensions_mm.height_mm).toFixed(2)} mm
                            </p>
                          </div>
                          <div>
                            <p className="text-xs text-gray-600">Width</p>
                            <p className="text-sm font-bold text-gray-900">
                              {parseFloat(individualClassAnalysis['1'].dimensions_mm.width_mm).toFixed(2)} mm
                            </p>
                          </div>
                          <div>
                            <p className="text-xs text-gray-600">Length</p>
                            <p className="text-sm font-bold text-gray-900">
                              {parseFloat(individualClassAnalysis['1'].dimensions_mm.length_mm).toFixed(2)} mm
                            </p>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Edema Accordion */}
                  <div className="bg-green-50 rounded-lg border border-green-100 overflow-hidden">
                    <button
                      type="button"
                      onClick={() => toggleAccordion(2)}
                      className="w-full flex items-center justify-between p-4 text-left hover:bg-green-100 transition-colors"
                    >
                      <div className="flex items-center gap-2">
                        <span className="w-3 h-3 rounded-full bg-green-500" />
                        <p className="text-sm font-semibold text-green-900">Edema</p>
                      </div>
                      <span className="text-gray-500">
                        {expandedAccordions[2] ? '▲' : '▼'}
                      </span>
                    </button>
                    {expandedAccordions[2] && individualClassAnalysis['2'] && (
                      <div className="p-4 pt-0 border-t border-green-200">
                        <div className="grid grid-cols-4 gap-3">
                          <div>
                            <p className="text-xs text-gray-600">Volume</p>
                            <p className="text-sm font-bold text-gray-900">
                              {parseFloat(individualClassAnalysis['2'].volume_cm3).toFixed(2)} cm³
                            </p>
                          </div>
                          <div>
                            <p className="text-xs text-gray-600">Height</p>
                            <p className="text-sm font-bold text-gray-900">
                              {parseFloat(individualClassAnalysis['2'].dimensions_mm.height_mm).toFixed(2)} mm
                            </p>
                          </div>
                          <div>
                            <p className="text-xs text-gray-600">Width</p>
                            <p className="text-sm font-bold text-gray-900">
                              {parseFloat(individualClassAnalysis['2'].dimensions_mm.width_mm).toFixed(2)} mm
                            </p>
                          </div>
                          <div>
                            <p className="text-xs text-gray-600">Length</p>
                            <p className="text-sm font-bold text-gray-900">
                              {parseFloat(individualClassAnalysis['2'].dimensions_mm.length_mm).toFixed(2)} mm
                            </p>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>

                  {/* ET Accordion */}
                  <div className="bg-fuchsia-50 rounded-lg border border-fuchsia-100 overflow-hidden">
                    <button
                      type="button"
                      onClick={() => toggleAccordion(3)}
                      className="w-full flex items-center justify-between p-4 text-left hover:bg-fuchsia-100 transition-colors"
                    >
                      <div className="flex items-center gap-2">
                        <span className="w-3 h-3 rounded-full bg-fuchsia-500" />
                        <p className="text-sm font-semibold text-fuchsia-900">Enhancing Tumor (ET)</p>
                      </div>
                      <span className="text-gray-500">
                        {expandedAccordions[3] ? '▲' : '▼'}
                      </span>
                    </button>
                    {expandedAccordions[3] && individualClassAnalysis['3'] && (
                      <div className="p-4 pt-0 border-t border-fuchsia-200">
                        <div className="grid grid-cols-4 gap-3">
                          <div>
                            <p className="text-xs text-gray-600">Volume</p>
                            <p className="text-sm font-bold text-gray-900">
                              {parseFloat(individualClassAnalysis['3'].volume_cm3).toFixed(2)} cm³
                            </p>
                          </div>
                          <div>
                            <p className="text-xs text-gray-600">Height</p>
                            <p className="text-sm font-bold text-gray-900">
                              {parseFloat(individualClassAnalysis['3'].dimensions_mm.height_mm).toFixed(2)} mm
                            </p>
                          </div>
                          <div>
                            <p className="text-xs text-gray-600">Width</p>
                            <p className="text-sm font-bold text-gray-900">
                              {parseFloat(individualClassAnalysis['3'].dimensions_mm.width_mm).toFixed(2)} mm
                            </p>
                          </div>
                          <div>
                            <p className="text-xs text-gray-600">Length</p>
                            <p className="text-sm font-bold text-gray-900">
                              {parseFloat(individualClassAnalysis['3'].dimensions_mm.length_mm).toFixed(2)} mm
                            </p>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          )}

          {!classAnalysis && (
            <div className="bg-gradient-to-br from-gray-50 to-gray-100 rounded-lg p-6 border border-gray-200">
              <p className="text-gray-500 text-center">
                Loading tumor analysis...
              </p>
            </div>
          )}

          {loadingAnalysis && (
            <div className="flex items-center justify-center py-4">
              <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-indigo-600"></div>
              <span className="ml-3 text-gray-600">Calculating analysis...</span>
            </div>
          )}
        </div>

        {/* MRI Visualization Workstation */}
        <div className="mb-6 overflow-hidden rounded-xl shadow-lg">
          <NiiVueViewer
            mriPath={result.mri_path}
            maskPath={result.mask_path}
            classMasks={result.class_masks}
            onClassChange={handleClassChange}
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
