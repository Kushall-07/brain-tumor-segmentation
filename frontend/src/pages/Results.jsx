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

  useEffect(() => {
    if (result && result.mask_path) {
      setLoadingAnalysis(true);
      const normalizedMaskPath = result.mask_path.replace(/\\/g, '/');
      const relativeMaskPath = normalizedMaskPath.replace(/.*outputs\/predictions\//, '');

      predictionService.getClassAnalysis(relativeMaskPath, [1, 2, 3])
        .then(response => {
          setClassAnalysis(response);
        })
        .catch(error => {
          console.error('Failed to get total tumor analysis:', error);
          setClassAnalysis(null);
        });

      predictionService.getIndividualClassAnalysis(relativeMaskPath)
        .then(response => {
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

  if (!result) {
    return null;
  }

  return (
    <div className="min-h-screen bg-parchment py-12 px-4 sm:px-6 lg:px-8 pt-28">
      <div className="max-w-4xl mx-auto">
        {/* 1. RESULTS HEADER */}
        <div className="text-center mb-6 pb-8 border-b border-sepia-border">
          <div className="flex items-center justify-center mb-4">
            <CheckCircle className="text-annotation" size={48} strokeWidth={1.5} />
          </div>
          <p className="atlas-label mb-2">Clinical Analysis Complete</p>
          <h1 className="font-serif text-4xl font-semibold text-ink mb-2">
            Prediction Successful
          </h1>
          <p className="text-sepia-muted text-lg mb-4">
            Case ID: <span className="font-mono text-ink-mono">{result.case_id}</span>
          </p>
        </div>

        {/* Prediction Details (compact) */}
        <div className="bg-parchment border border-sepia-border rounded-sm p-6 sm:p-8 mb-6">
          <div className="atlas-section-header">
            <FileText className="text-arterial" size={22} strokeWidth={1.5} />
            <h2 className="font-serif text-2xl font-semibold text-ink">
              Prediction Details
            </h2>
          </div>

          <div className="space-y-4">
            <div className="atlas-metadata-field">
              <p className="atlas-label mb-1">Segmentation Mask</p>
              <p className="text-sm text-ink break-all font-mono">{result.mask_path}</p>
            </div>

            {result.probability_path && (
              <div className="atlas-metadata-field">
                <p className="atlas-label mb-1">Probability Map</p>
                <p className="text-sm text-ink break-all font-mono">{result.probability_path}</p>
              </div>
            )}

            <div className="bg-parchment-dark border border-sepia-border rounded-sm p-4">
              <div className="flex items-start">
                <AlertCircle className="text-annotation mt-0.5 mr-2 flex-shrink-0" size={20} strokeWidth={1.5} />
                <div className="text-sm text-ink-body">
                  <p className="font-medium text-ink mb-1">Note</p>
                  <p>
                    The segmentation mask has been saved to the outputs directory.
                    Use the download button below to retrieve the file.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* 2. MRI VISUALIZATION */}
        <div className="mb-6 overflow-hidden rounded-sm border border-brass/60">
          <NiiVueViewer
            mriPath={result.mri_path}
            maskPath={result.mask_path}
            classMasks={result.class_masks}
            onClassChange={handleClassChange}
          />
        </div>

        {/* 3. TUMOR ANALYSIS */}
        <div className="bg-parchment border border-sepia-border rounded-sm p-6 sm:p-8 mb-6">
          <div className="atlas-section-header">
            <Activity className="text-arterial" size={22} strokeWidth={1.5} />
            <h2 className="font-serif text-2xl font-semibold text-ink">
              Tumor Analysis
            </h2>
          </div>

          {classAnalysis && (
            <div className="space-y-6 mb-6">
              <div className="border border-sepia-border rounded-sm p-5 bg-parchment-dark">
                <p className="atlas-label mb-2">Total Tumor (WT)</p>
                <p className="text-3xl font-mono font-medium text-ink-mono mb-4">
                  {parseFloat(classAnalysis.volume_cm3).toFixed(2)} cm<sup>3</sup>
                </p>
                <div className="atlas-divider pt-4">
                  <div className="grid grid-cols-3 gap-4 mt-4">
                    <div className="text-center sm:text-left">
                      <p className="atlas-label mb-1">Height</p>
                      <p className="text-xl font-mono font-medium text-ink-mono">
                        {parseFloat(classAnalysis.dimensions_mm.height_mm).toFixed(2)} mm
                      </p>
                    </div>
                    <div className="text-center sm:text-left">
                      <p className="atlas-label mb-1">Width</p>
                      <p className="text-xl font-mono font-medium text-ink-mono">
                        {parseFloat(classAnalysis.dimensions_mm.width_mm).toFixed(2)} mm
                      </p>
                    </div>
                    <div className="text-center sm:text-left">
                      <p className="atlas-label mb-1">Length</p>
                      <p className="text-xl font-mono font-medium text-ink-mono">
                        {parseFloat(classAnalysis.dimensions_mm.length_mm).toFixed(2)} mm
                      </p>
                    </div>
                  </div>
                </div>
              </div>

              {individualClassAnalysis && (
                <div className="space-y-3">
                  <p className="atlas-label">Individual Class Analysis</p>

                  {/* NCR/NET Accordion */}
                  <div className="border border-sepia-border rounded-sm overflow-hidden bg-parchment">
                    <button
                      type="button"
                      onClick={() => toggleAccordion(1)}
                      className="w-full flex items-center justify-between p-4 text-left hover:bg-parchment-dark transition-colors border-b border-sepia-border/50"
                    >
                      <div className="flex items-center gap-2">
                        <span className="w-2.5 h-2.5 rounded-full bg-red-500" />
                        <p className="atlas-label text-ink">NCR/NET</p>
                      </div>
                      <span className="text-sepia-muted text-sm font-mono">
                        {expandedAccordions[1] ? '▲' : '▼'}
                      </span>
                    </button>
                    {expandedAccordions[1] && individualClassAnalysis['1'] && (
                      <div className="p-4 bg-parchment-dark">
                        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                          <div>
                            <p className="atlas-label mb-1">Volume</p>
                            <p className="text-sm font-mono font-medium text-ink-mono">
                              {parseFloat(individualClassAnalysis['1'].volume_cm3).toFixed(2)} cm³
                            </p>
                          </div>
                          <div>
                            <p className="atlas-label mb-1">Height</p>
                            <p className="text-sm font-mono font-medium text-ink-mono">
                              {parseFloat(individualClassAnalysis['1'].dimensions_mm.height_mm).toFixed(2)} mm
                            </p>
                          </div>
                          <div>
                            <p className="atlas-label mb-1">Width</p>
                            <p className="text-sm font-mono font-medium text-ink-mono">
                              {parseFloat(individualClassAnalysis['1'].dimensions_mm.width_mm).toFixed(2)} mm
                            </p>
                          </div>
                          <div>
                            <p className="atlas-label mb-1">Length</p>
                            <p className="text-sm font-mono font-medium text-ink-mono">
                              {parseFloat(individualClassAnalysis['1'].dimensions_mm.length_mm).toFixed(2)} mm
                            </p>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Edema Accordion */}
                  <div className="border border-sepia-border rounded-sm overflow-hidden bg-parchment">
                    <button
                      type="button"
                      onClick={() => toggleAccordion(2)}
                      className="w-full flex items-center justify-between p-4 text-left hover:bg-parchment-dark transition-colors border-b border-sepia-border/50"
                    >
                      <div className="flex items-center gap-2">
                        <span className="w-2.5 h-2.5 rounded-full bg-green-500" />
                        <p className="atlas-label text-ink">Edema</p>
                      </div>
                      <span className="text-sepia-muted text-sm font-mono">
                        {expandedAccordions[2] ? '▲' : '▼'}
                      </span>
                    </button>
                    {expandedAccordions[2] && individualClassAnalysis['2'] && (
                      <div className="p-4 bg-parchment-dark">
                        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                          <div>
                            <p className="atlas-label mb-1">Volume</p>
                            <p className="text-sm font-mono font-medium text-ink-mono">
                              {parseFloat(individualClassAnalysis['2'].volume_cm3).toFixed(2)} cm³
                            </p>
                          </div>
                          <div>
                            <p className="atlas-label mb-1">Height</p>
                            <p className="text-sm font-mono font-medium text-ink-mono">
                              {parseFloat(individualClassAnalysis['2'].dimensions_mm.height_mm).toFixed(2)} mm
                            </p>
                          </div>
                          <div>
                            <p className="atlas-label mb-1">Width</p>
                            <p className="text-sm font-mono font-medium text-ink-mono">
                              {parseFloat(individualClassAnalysis['2'].dimensions_mm.width_mm).toFixed(2)} mm
                            </p>
                          </div>
                          <div>
                            <p className="atlas-label mb-1">Length</p>
                            <p className="text-sm font-mono font-medium text-ink-mono">
                              {parseFloat(individualClassAnalysis['2'].dimensions_mm.length_mm).toFixed(2)} mm
                            </p>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>

                  {/* ET Accordion */}
                  <div className="border border-sepia-border rounded-sm overflow-hidden bg-parchment">
                    <button
                      type="button"
                      onClick={() => toggleAccordion(3)}
                      className="w-full flex items-center justify-between p-4 text-left hover:bg-parchment-dark transition-colors border-b border-sepia-border/50"
                    >
                      <div className="flex items-center gap-2">
                        <span className="w-2.5 h-2.5 rounded-full bg-fuchsia-500" />
                        <p className="atlas-label text-ink">Enhancing Tumor (ET)</p>
                      </div>
                      <span className="text-sepia-muted text-sm font-mono">
                        {expandedAccordions[3] ? '▲' : '▼'}
                      </span>
                    </button>
                    {expandedAccordions[3] && individualClassAnalysis['3'] && (
                      <div className="p-4 bg-parchment-dark">
                        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                          <div>
                            <p className="atlas-label mb-1">Volume</p>
                            <p className="text-sm font-mono font-medium text-ink-mono">
                              {parseFloat(individualClassAnalysis['3'].volume_cm3).toFixed(2)} cm³
                            </p>
                          </div>
                          <div>
                            <p className="atlas-label mb-1">Height</p>
                            <p className="text-sm font-mono font-medium text-ink-mono">
                              {parseFloat(individualClassAnalysis['3'].dimensions_mm.height_mm).toFixed(2)} mm
                            </p>
                          </div>
                          <div>
                            <p className="atlas-label mb-1">Width</p>
                            <p className="text-sm font-mono font-medium text-ink-mono">
                              {parseFloat(individualClassAnalysis['3'].dimensions_mm.width_mm).toFixed(2)} mm
                            </p>
                          </div>
                          <div>
                            <p className="atlas-label mb-1">Length</p>
                            <p className="text-sm font-mono font-medium text-ink-mono">
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
            <div className="border border-sepia-border rounded-sm p-6 bg-parchment-dark">
              <p className="text-sepia-muted text-center">
                Loading tumor analysis...
              </p>
            </div>
          )}

          {loadingAnalysis && (
            <div className="flex items-center justify-center py-4">
              <div className="animate-spin rounded-full h-6 w-6 border-2 border-arterial border-t-transparent"></div>
              <span className="ml-3 text-sepia-muted">Calculating analysis...</span>
            </div>
          )}
        </div>

        {/* Action Buttons */}
        <div className="flex flex-col sm:flex-row gap-4 justify-center">
          <button
            onClick={handleDownload}
            disabled={isDownloading}
            className={`
              px-6 py-3 rounded-sm font-medium transition-colors duration-200
              flex items-center justify-center space-x-2
              ${isDownloading
                ? 'bg-sepia-border text-sepia-muted cursor-not-allowed'
                : 'bg-arterial hover:bg-arterial-light text-parchment'
              }
            `}
          >
            {isDownloading ? (
              <>
                <div className="animate-spin rounded-full h-5 w-5 border-2 border-parchment border-t-transparent"></div>
                <span>Downloading...</span>
              </>
            ) : (
              <>
                <Download size={20} strokeWidth={1.5} />
                <span>Download Prediction</span>
              </>
            )}
          </button>

          <button
            onClick={handlePredictAnother}
            className="px-6 py-3 rounded-sm font-medium text-parchment bg-annotation hover:bg-annotation/90 transition-colors duration-200 flex items-center justify-center space-x-2"
          >
            <ArrowLeft size={20} strokeWidth={1.5} />
            <span>Predict Another Patient</span>
          </button>
        </div>

        {/* Footer Info */}
        <div className="mt-8 pt-6 border-t border-sepia-border text-center text-sm text-sepia-muted">
          <p className="font-serif text-ink">Brain Tumor Segmentation AI System</p>
          <p className="mt-1">Powered by 3D U-Net and MONAI</p>
        </div>
      </div>
    </div>
  );
}
