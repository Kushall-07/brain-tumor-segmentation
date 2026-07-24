const API_BASE = '/api';

class ApiError extends Error {
  constructor(message, status, data) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.data = data;
  }
}

async function request(endpoint, options = {}) {
  const url = `${API_BASE}${endpoint}`;

  const config = {
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
    ...options,
  };

  if (options.body && !(options.body instanceof FormData)) {
    config.body = JSON.stringify(options.body);
  }

  const response = await fetch(url, config);

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new ApiError(
      errorData.message || `HTTP error! status: ${response.status}`,
      response.status,
      errorData
    );
  }

  if (response.status === 204) {
    return null;
  }

  return response.json();
}

export const api = {
  // Job management
  createJob: (files) => {
    const formData = new FormData();
    Object.entries(files).forEach(([modality, file]) => {
      formData.append(modality, file);
    });
    return request('/jobs', {
      method: 'POST',
      body: formData,
      headers: {}, // Let browser set Content-Type for FormData
    });
  },

  getJob: (jobId) => request(`/jobs/${jobId}`),

  getJobStatus: (jobId) => request(`/jobs/${jobId}/status`),

  getJobResults: (jobId) => request(`/jobs/${jobId}/results`),

  // Streaming progress
  streamJobProgress: (jobId, onMessage, onError, onClose) => {
    const eventSource = new EventSource(`${API_BASE}/jobs/${jobId}/stream`);

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        onMessage(data);
      } catch (e) {
        console.error('Failed to parse SSE message:', e);
      }
    };

    eventSource.onerror = (err) => {
      eventSource.close();
      if (onError) onError(err);
    };

    return () => eventSource.close();
  },

  // Download results
  downloadResults: (jobId, format = 'nifti') => {
    return fetch(`${API_BASE}/jobs/${jobId}/download?format=${format}`)
      .then(res => {
        if (!res.ok) throw new Error('Download failed');
        return res.blob();
      });
  },

  // Generate report
  generateReport: (jobId, options = {}) => {
    return request(`/jobs/${jobId}/report`, {
      method: 'POST',
      body: options,
    });
  },

  // Health check
  healthCheck: () => request('/health'),
};

export { ApiError };
export default api;