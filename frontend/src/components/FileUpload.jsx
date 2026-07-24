import { useState, useRef, useCallback } from 'react';

const MODALITIES = ['t1c', 't1n', 't2f', 't2w'];

function FileUpload({ onFilesUploaded, onProcessingStart }) {
  const [files, setFiles] = useState({});
  const [dragActive, setDragActive] = useState(false);
  const [errors, setErrors] = useState({});
  const [uploadProgress, setUploadProgress] = useState({});
  const fileInputsRef = useRef({});

  const validateFile = (file, modality) => {
    const validExtensions = ['.nii', '.nii.gz', '.npy', '.npz'];
    const maxSize = 500 * 1024 * 1024; // 500MB

    const hasValidExtension = validExtensions.some(ext => file.name.endsWith(ext));
    if (!hasValidExtension) {
      return `Invalid file format for ${modality.toUpperCase()}. Please use .nii, .nii.gz, .npy, or .npz files.`;
    }

    if (file.size > maxSize) {
      return `File size exceeds 500MB limit for ${modality.toUpperCase()}.`;
    }

    return null;
  };

  const handleFileSelect = (modality, event) => {
    const file = event.target.files[0];
    if (!file) return;

    const error = validateFile(file, modality);
    if (error) {
      setErrors(prev => ({ ...prev, [modality]: error }));
      event.target.value = '';
      return;
    }

    setErrors(prev => {
      const next = { ...prev };
      delete next[modality];
      return next;
    });

    setFiles(prev => ({ ...prev, [modality]: file }));
    setUploadProgress(prev => ({ ...prev, [modality]: 100 }));
  };

  const handleDrag = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((modality, e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    const file = e.dataTransfer.files[0];
    if (!file) return;

    const error = validateFile(file, modality);
    if (error) {
      setErrors(prev => ({ ...prev, [modality]: error }));
      return;
    }

    setErrors(prev => {
      const next = { ...prev };
      delete next[modality];
      return next;
    });

    setFiles(prev => ({ ...prev, [modality]: file }));
    setUploadProgress(prev => ({ ...prev, [modality]: 100 }));
  }, []);

  const removeFile = (modality) => {
    setFiles(prev => {
      const next = { ...prev };
      delete next[modality];
      return next;
    });
    setErrors(prev => {
      const next = { ...prev };
      delete next[modality];
      return next;
    });
    setUploadProgress(prev => {
      const next = { ...prev };
      delete next[modality];
      return next;
    });
    if (fileInputsRef.current[modality]) {
      fileInputsRef.current[modality].value = '';
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    const uploadedFiles = Object.keys(files).length;
    if (uploadedFiles < 4) {
      setErrors(prev => ({
        ...prev,
        general: 'Please upload all 4 MRI modalities (T1c, T1n, T2f, T2w) before processing.'
      }));
      return;
    }

    setErrors(prev => {
      const next = { ...prev };
      delete next.general;
      return next;
    });

    onFilesUploaded(files);
    onProcessingStart();
  };

  const getModalityInfo = (modality) => {
    const info = {
      t1c: { name: 'T1c (T1 Contrast)', description: 'T1-weighted with contrast enhancement', color: 'red' },
      t1n: { name: 'T1n (T1 Native)', description: 'T1-weighted native (non-contrast)', color: 'blue' },
      t2f: { name: 'T2f (T2 FLAIR)', description: 'T2-weighted FLAIR', color: 'green' },
      t2w: { name: 'T2w (T2 Weighted)', description: 'T2-weighted', color: 'purple' }
    };
    return info[modality];
  };

  const ModalityColors = {
    red: 'border-red-500/50 bg-red-500/10 text-red-400',
    blue: 'border-blue-500/50 bg-blue-500/10 text-blue-400',
    green: 'border-green-500/50 bg-green-500/10 text-green-400',
    purple: 'border-purple-500/50 bg-purple-500/10 text-purple-400'
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {MODALITIES.map(modality => {
          const info = getModalityInfo(modality);
          const colorClass = ModalityColors[info.color];
          const file = files[modality];
          const error = errors[modality];
          const progress = uploadProgress[modality];

          return (
            <div
              key={modality}
              className={`relative border-2 rounded-xl p-4 transition-all ${dragActive ? 'border-cyan-500/50 bg-cyan-500/5' : `border-slate-700 ${colorClass}`}`}
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={(e) => handleDrop(modality, e)}
            >
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center space-x-2">
                  <div className={`w-3 h-3 rounded-full ${colorClass.replace('border-', 'bg-').replace('/50 bg-', '/50 bg-')}`} />
                  <span className="font-medium text-slate-100">{info.name}</span>
                </div>
                {file && (
                  <button
                    type="button"
                    onClick={() => removeFile(modality)}
                    className="text-slate-400 hover:text-red-400 transition-colors p-1"
                    aria-label={`Remove ${info.name}`}
                  >
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </button>
                )}
              </div>

              <p className="text-xs text-slate-500 mb-3">{info.description}</p>

              {!file ? (
                <>
                  <input
                    ref={el => { fileInputsRef.current[modality] = el; }}
                    type="file"
                    id={`upload-${modality}`}
                    accept=".nii,.nii.gz,.npy,.npz"
                    onChange={(e) => handleFileSelect(modality, e)}
                    className="hidden"
                  />
                  <label
                    htmlFor={`upload-${modality}`}
                    className={`w-full py-3 px-4 text-center rounded-lg border-2 border-dashed cursor-pointer transition-all ${
                      dragActive ? 'border-cyan-500 bg-cyan-500/10' : `border-slate-600 hover:border-${info.color}-500 hover:bg-slate-700`
                    }`}
                  >
                    <svg className="mx-auto mb-2 w-8 h-8 text-slate-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3 3m0 0l-3-3m3 3V12" />
                    </svg>
                    <p className="text-sm text-slate-400">Drag & drop or click to upload</p>
                    <p className="text-xs text-slate-500 mt-1">.nii, .nii.gz, .npy, .npz</p>
                  </label>
                  {error && (
                    <p className="mt-2 text-xs text-red-400 text-center">{error}</p>
                  )}
                </>
              ) : (
                <div className="space-y-2">
                  <div className="flex items-center justify-between p-2 bg-slate-800/50 rounded-lg">
                    <div className="flex items-center space-x-2 flex-1 min-w-0">
                      <svg className="w-5 h-5 text-slate-400 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 15l-2 5L9 9l11 4-5 2zm0 0l5 5M7.188 2.239l.777 2.897M5.136 7.965l-2.898-.777M13.95 4.05l-2.122 2.122m-5.657 5.656l-2.12 2.122" />
                      </svg>
                      <span className="text-sm text-slate-200 truncate">{file.name}</span>
                    </div>
                    <span className="text-xs text-slate-400 ml-2 flex-shrink-0">
                      {(file.size / (1024 * 1024)).toFixed(2)} MB
                    </span>
                  </div>
                  {progress !== undefined && (
                    <div className="w-full h-1.5 bg-slate-700 rounded-full overflow-hidden">
                      <div
                        className={`h-full rounded-full transition-all duration-300 ${colorClass.replace('border-', 'bg-').replace('/50 bg-', '')}`}
                        style={{ width: `${progress}%` }}
                      />
                    </div>
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>

      {errors.general && (
        <div className="p-4 bg-red-500/10 border border-red-500/30 rounded-lg text-red-400 text-sm text-center">
          {errors.general}
        </div>
      )}

      <div className="flex justify-center pt-4">
        <button
          type="submit"
          disabled={Object.keys(files).length < 4}
          className={`w-full md:w-1/2 lg:w-1/3 px-8 py-4 rounded-xl font-semibold text-lg transition-all ${
            Object.keys(files).length >= 4
              ? 'bg-gradient-to-r from-cyan-500 to-blue-600 text-white hover:from-cyan-600 hover:to-blue-700 shadow-lg shadow-cyan-500/25'
              : 'bg-slate-700 text-slate-400 cursor-not-allowed'
          }`}
        >
          {Object.keys(files).length >= 4 ? 'Start Processing' : `Upload All 4 Modalities (${Object.keys(files).length}/4)`}
        </button>
      </div>

      <div className="text-center text-sm text-slate-500">
        <p>Required modalities: <span className="font-medium text-slate-300">T1c, T1n, T2f, T2w</span></p>
        <p className="mt-1">Supported formats: <span className="font-medium text-slate-300">.nii, .nii.gz, .npy, .npz</span> (max 500MB each)</p>
      </div>
    </form>
  );
}

export default FileUpload;