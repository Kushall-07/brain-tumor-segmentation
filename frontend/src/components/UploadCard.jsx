import { useState, useRef } from 'react';
import { Upload, X, FileText, AlertCircle } from 'lucide-react';

const UploadCard = ({ 
  modality, 
  onFileSelect, 
  onRemove, 
  selectedFile, 
  disabled = false,
  error 
}) => {
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef(null);

  const validExtensions = ['.nii', '.nii.gz'];

  const handleDragOver = (e) => {
    e.preventDefault();
    if (!disabled) {
      setIsDragging(true);
    }
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);

    if (disabled) return;

    const files = e.dataTransfer.files;
    if (files.length > 0) {
      validateAndSelectFile(files[0]);
    }
  };

  const handleFileInput = (e) => {
    const files = e.target.files;
    if (files.length > 0) {
      validateAndSelectFile(files[0]);
    }
  };

  const validateAndSelectFile = (file) => {
    const fileName = file.name.toLowerCase();
    const isValidExtension = validExtensions.some(ext => fileName.endsWith(ext));

    if (!isValidExtension) {
      onRemove?.();
      return;
    }

    onFileSelect(file);
  };

  const handleClick = () => {
    if (!disabled && !selectedFile) {
      fileInputRef.current?.click();
    }
  };

  const handleRemove = (e) => {
    e.stopPropagation();
    onRemove?.();
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + ' ' + sizes[i];
  };

  return (
    <div className="relative">
      <input
        ref={fileInputRef}
        type="file"
        accept=".nii,.nii.gz"
        onChange={handleFileInput}
        className="hidden"
        disabled={disabled}
      />

      <div
        onClick={handleClick}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className={`
          relative border-2 border-dashed rounded-lg p-6 transition-all duration-200
          ${isDragging ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-gray-400'}
          ${selectedFile ? 'border-green-500 bg-green-50' : ''}
          ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
          ${error ? 'border-red-500 bg-red-50' : ''}
        `}
      >
        {/* Remove Button */}
        {selectedFile && !disabled && (
          <button
            onClick={handleRemove}
            className="absolute top-2 right-2 p-1 bg-red-500 text-white rounded-full hover:bg-red-600 transition-colors"
            type="button"
          >
            <X size={16} />
          </button>
        )}

        {/* Error Icon */}
        {error && !selectedFile && (
          <div className="absolute top-2 right-2 text-red-500">
            <AlertCircle size={20} />
          </div>
        )}

        {/* Content */}
        <div className="flex flex-col items-center justify-center space-y-3">
          {selectedFile ? (
            <>
              <FileText className="text-green-600" size={32} />
              <div className="text-center">
                <p className="text-sm font-medium text-gray-900 truncate max-w-[150px]">
                  {selectedFile.name}
                </p>
                <p className="text-xs text-gray-500">
                  {formatFileSize(selectedFile.size)}
                </p>
              </div>
            </>
          ) : (
            <>
              <Upload className={isDragging ? 'text-blue-500' : 'text-gray-400'} size={32} />
              <div className="text-center">
                <p className="text-sm font-medium text-gray-700">
                  {modality.toUpperCase()}
                </p>
                <p className="text-xs text-gray-500">
                  Drag & drop or click to browse
                </p>
                <p className="text-xs text-gray-400 mt-1">
                  .nii or .nii.gz
                </p>
              </div>
            </>
          )}
        </div>

        {/* Upload Status Badge */}
        {selectedFile && (
          <div className="absolute bottom-2 left-2">
            <span className="inline-flex items-center px-2 py-1 text-xs font-medium text-green-800 bg-green-100 rounded-full">
              Ready
            </span>
          </div>
        )}
      </div>

      {/* Error Message */}
      {error && !selectedFile && (
        <p className="mt-1 text-xs text-red-600">{error}</p>
      )}
    </div>
  );
};

export default UploadCard;
