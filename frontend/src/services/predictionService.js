import axios from 'axios';

const API_BASE_URL = 'http://127.0.0.1:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 300000, // 5 minutes timeout for large file uploads
});

export const predictionService = {
  /**
   * Start an asynchronous prediction job by uploading MRI modalities
   * @param {FormData} formData - FormData with t1, t1ce, t2, flair files
   * @param {Object} params - Query parameters (checkpoint_path, save_probabilities)
   * @returns {Promise<Object>} API response with job_id and status
   */
  async startPrediction(formData, params = {}) {
    try {
      const response = await api.post('/predict/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        params,
      });
      return response.data;
    } catch (error) {
      if (error.response) {
        throw {
          status: error.response.status,
          message: error.response.data.detail || 'Upload failed',
          data: error.response.data,
        };
      } else if (error.request) {
        throw {
          status: 0,
          message: 'Network error - unable to connect to server',
        };
      } else {
        throw {
          status: 0,
          message: error.message || 'Request failed',
        };
      }
    }
  },

  /**
   * Poll the status of an asynchronous prediction job
   * @param {string} jobId - The job ID returned by startPrediction
   * @returns {Promise<Object>} Job status with stage, message, result, and error
   */
  async getPredictionStatus(jobId) {
    try {
      const response = await api.get(`/predict/status/${jobId}`);
      return response.data;
    } catch (error) {
      if (error.response) {
        throw {
          status: error.response.status,
          message: error.response.data.detail || 'Status check failed',
          data: error.response.data,
        };
      } else if (error.request) {
        throw {
          status: 0,
          message: 'Network error - unable to connect to server',
        };
      } else {
        throw {
          status: 0,
          message: error.message || 'Status request failed',
        };
      }
    }
  },

  /**
   * Upload MRI modalities and run brain tumor segmentation (legacy synchronous)
   * @deprecated Use startPrediction + getPredictionStatus instead
   * @param {Object} formData - FormData with t1, t1ce, t2, flair files
   * @param {Object} params - Query parameters (checkpoint_path, save_probabilities)
   * @returns {Promise<Object>} API response with prediction results
   */
  async uploadPrediction(formData, params = {}) {
    return this.startPrediction(formData, params);
  },

  /**
   * Check API health
   * @returns {Promise<Object>} Health status
   */
  async healthCheck() {
    try {
      const response = await api.get('/health');
      return response.data;
    } catch (error) {
      throw {
        status: error.response?.status || 0,
        message: 'Health check failed',
      };
    }
  },

  /**
   * Download a prediction file
   * @param {string} filePath - Relative path within outputs/predictions/
   * @returns {Promise<Blob>} File blob
   */
  async downloadPrediction(filePath) {
    try {
      const response = await api.get(`/download/${filePath}`, {
        responseType: 'blob',
      });
      return response.data;
    } catch (error) {
      if (error.response) {
        throw {
          status: error.response.status,
          message: error.response.data.detail || 'Download failed',
        };
      } else if (error.request) {
        throw {
          status: 0,
          message: 'Network error - unable to connect to server',
        };
      } else {
        throw {
          status: 0,
          message: error.message || 'Download request failed',
        };
      }
    }
  },
};

export default predictionService;
