import axios from 'axios';

/**
 * Axios instance for the FastAPI backend.
 * API calls are implemented in dedicated service modules (e.g. predictionService.js).
 */
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 60000,
  headers: {
    Accept: 'application/json',
  },
});

// Export as 'api' for backward compatibility with Processing.jsx
export const api = {
  createJob: async (files) => {
    // Placeholder for job creation - not currently used in Phase 2/3
    return { jobId: 'mock-job-id' };
  }
};

export default apiClient;