import api from './api';

/**
 * Normalize an axios error into the shape every predictionService method
 * throws, so each call site below only needs one line instead of a
 * repeated three-branch try/catch.
 */
function normalizeError(error, fallbackMessage) {
  if (error.response) {
    return {
      status: error.response.status,
      message: error.response.data?.detail || fallbackMessage,
      data: error.response.data,
      response: error.response,
    };
  }
  if (error.request) {
    return {
      status: 0,
      message: 'Network error - unable to connect to server',
    };
  }
  return {
    status: 0,
    message: error.message || fallbackMessage,
  };
}

export const predictionService = {
  /**
   * Start an asynchronous prediction job by uploading MRI modalities.
   * The backend selects the checkpoint server-side — only `save_probabilities`
   * (and similar non-sensitive flags) belong in `params`.
   * @param {FormData} formData - FormData with t1, t1ce, t2, flair files
   * @param {Object} params - Query parameters (e.g. save_probabilities)
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
      throw normalizeError(error, 'Upload failed');
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
      throw normalizeError(error, 'Status check failed');
    }
  },

  /**
   * Upload MRI modalities and run brain tumor segmentation (legacy synchronous)
   * @deprecated Use startPrediction + getPredictionStatus instead
   * @param {Object} formData - FormData with t1, t1ce, t2, flair files
   * @param {Object} params - Query parameters (e.g. save_probabilities)
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
      throw normalizeError(error, 'Health check failed');
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
      throw normalizeError(error, 'Download failed');
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
      throw normalizeError(error, 'Class analysis failed');
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
      throw normalizeError(error, 'Individual class analysis failed');
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
      return response.data.validation;
    } catch (error) {
      throw normalizeError(error, 'Validation failed');
    }
  },
};

export default predictionService;
