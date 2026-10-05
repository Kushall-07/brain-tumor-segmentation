import axios from 'axios';

/**
 * Single shared axios instance for the FastAPI backend.
 * All API calls go through dedicated service modules (e.g. predictionService.js)
 * built on top of this client, rather than creating their own instances.
 */
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  // Large MRI volume uploads and inference can take a while; matches the
  // timeout predictionService previously set on its own separate instance.
  timeout: 300000,
});

export default apiClient;