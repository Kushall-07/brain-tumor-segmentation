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

  /**
   * Calculate volume and dimensions for specific tumor classes
   * @param {string} maskPath - Relative path to the original mask
   * @param {Array<number>} classes - Array of class IDs to analyze (e.g., [1, 2, 3])
   * @returns {Promise<Object>} API response with volume_cm3 and dimensions_mm
   */
  async getClassAnalysis(maskPath, classes) {
    try {
      const response = await api.post('/predict/class-analysis', {
        mask_path: maskPath,
        classes,
      });
      return response.data;
    } catch (error) {
      if (error.response) {
        throw {
          status: error.response.status,
          message: error.response.data.detail || 'Class analysis failed',
        };
      } else if (error.request) {
        throw {
          status: 0,
          message: 'Network error - unable to connect to server',
        };
      } else {
        throw {
          status: 0,
          message: error.message || 'Class analysis request failed',
        };
      }
    }
  },

  /**
   * Calculate volume and dimensions for each individual tumor class
   * @param {string} maskPath - Relative path to the original mask
   * @returns {Promise<Object>} API response with individual class analysis
   */
  async getIndividualClassAnalysis(maskPath) {
    try {
      const response = await api.post('/predict/individual-class-analysis', {
        mask_path: maskPath,
        classes: [],
      });
      return response.data;
    } catch (error) {
      if (error.response) {
        throw {
          status: error.response.status,
          message: error.response.data.detail || 'Individual class analysis failed',
        };
      } else if (error.request) {
        throw {
          status: 0,
          message: 'Network error - unable to connect to server',
        };
      } else {
        throw {
          status: 0,
          message: error.message || 'Individual class analysis request failed',
        };
      }
    }
  },

  async getMethodsSummary() {
    const response = await api.get('/research/methods');
    return response.data;
  },

  async getModelInfo(checkpointPath) {
    const response = await api.get('/research/model-info', {
      params: checkpointPath ? { checkpoint_path: checkpointPath } : {},
    });
    return response.data;
  },

  async validateCase(predictionMaskPath, groundTruthMaskPath) {
    try {
      const response = await api.post('/predict/validate-case', {
        prediction_mask_path: predictionMaskPath,
        ground_truth_mask_path: groundTruthMaskPath,
      });
      console.log('[VALIDATION] API RESPONSE:', response.data);
      return response.data.validation;
    } catch (error) {
      if (error.response) {
        throw {
          status: error.response.status,
          message: error.response.data.detail || 'Validation failed',
          response: error.response,
          data: error.response.data,
        };
      }
      throw error;
    }
  },
};

export default predictionService;
