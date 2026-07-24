import { useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import ResultsViewer from '../components/ResultsViewer';
import api from '../services/api';

export default function Results() {
  const navigate = useNavigate();
  const location = useLocation();
  const [results, setResults] = useState(null);
  const [jobId, setJobId] = useState(null);
  const [isDownloading, setIsDownloading] = useState(false);
  const [isGeneratingReport, setIsGeneratingReport] = useState(false);

  useEffect(() => {
    if (location.state?.results) {
      setResults(location.state.results);
      setJobId(location.state.jobId);
    } else if (location.state?.jobId) {
      // If we only have jobId, fetch results
      setJobId(location.state.jobId);
      fetchResults(location.state.jobId);
    } else {
      // No results or jobId, redirect to home
      navigate('/');
    }
  }, [location, navigate]);

  const fetchResults = async (id) => {
    try {
      // In a real app, this would fetch from the API
      // For now, use mock data
      const mockResults = {
        jobId: id,
        status: 'completed',
        processingTime: 2.3,
        volumes: {
          'Necrotic Core': 12.5,
          'Edema': 45.2,
          'Enhancing Tumor': 8.3,
          'Whole Tumor': 66.0,
          'Tumor Core': 20.8,
          'ET': 8.3
        },
        diceScores: {
          'Whole Tumor': 0.91,
          'Tumor Core': 0.84,
          'Enhancing Tumor': 0.80
        }
      };
      setResults(mockResults);
    } catch (err) {
      console.error('Failed to fetch results:', err);
      navigate('/');
    }
  };

  const handleDownload = async () => {
    setIsDownloading(true);
    try {
      await api.downloadResults(jobId);
    } catch (err) {
      console.error('Download failed:', err);
    } finally {
      setIsDownloading(false);
    }
  };

  const handleGenerateReport = async () => {
    setIsGeneratingReport(true);
    try {
      await api.generateReport(jobId, results);
    } catch (err) {
      console.error('Report generation failed:', err);
    } finally {
      setIsGeneratingReport(false);
    }
  };

  if (!results) {
    return (
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="text-center">
          <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-primary-500/20 mb-6">
            <svg className="w-8 h-8 text-primary-500 animate-spin" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
            </svg>
          </div>
          <h2 className="text-2xl font-bold text-white mb-2">Loading Results...</h2>
          <p className="text-slate-400">Fetching segmentation results</p>
        </div>
      </div>
    );
  }

  return (
    <div className="h-screen flex flex-col">
      <ResultsViewer results={results} jobId={jobId} />
    </div>
  );
}