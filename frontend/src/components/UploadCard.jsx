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
          relative border border-dashed rounded-sm p-6 transition-colors duration-200 bg-parchment
          ${isDragging ? 'border-annotation bg-parchment-dark' : 'border-sepia-border hover:border-sepia-muted'}
          ${selectedFile ? 'border-annotation bg-parchment-dark' : ''}
          ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
          ${error ? 'border-arterial bg-parchment-dark' : ''}
        `}
      >
        {selectedFile && !disabled && (
          <button
            onClick={handleRemove}
            className="absolute top-2 right-2 p-1 bg-arterial text-parchment rounded-sm hover:bg-arterial-light transition-colors"
            type="button"
          >
            <X size={14} />
          </button>
        )}

        {error && !selectedFile && (
          <div className="absolute top-2 right-2 text-arterial">
            <AlertCircle size={18} />
          </div>
        )}

        <div className="flex flex-col items-center justify-center space-y-3">
          {selectedFile ? (
            <>
              <FileText className="text-annotation" size={28} strokeWidth={1.5} />
              <div className="text-center">
                <p className="text-sm font-medium text-ink truncate max-w-[150px]">
                  {selectedFile.name}
                </p>
                <p className="text-xs font-mono text-ink-label mt-1">
                  {formatFileSize(selectedFile.size)}
                </p>
              </div>
            </>
          ) : (
            <>
              <Upload className={isDragging ? 'text-annotation' : 'text-sepia-muted'} size={28} strokeWidth={1.5} />
              <div className="text-center">
                <p className="atlas-label text-ink">
                  {modality.toUpperCase()}
                </p>
                <p className="text-xs text-ink-body mt-2">
                  Drag & drop or click to browse
                </p>
                <p className="text-xs font-mono text-sepia-muted mt-1">
                  .nii or .nii.gz
                </p>
              </div>
            </>
          )}
        </div>

        {selectedFile && (
          <div className="absolute bottom-2 left-2">
            <span className="inline-flex items-center px-2 py-0.5 text-[10px] font-medium uppercase tracking-wider text-annotation border border-sepia-border bg-parchment rounded-sm">
              Ready
            </span>
          </div>
        )}
      </div>

      {error && !selectedFile && (
        <p className="mt-1 text-xs text-arterial">{error}</p>
      )}
    </div>
  );
};

export default UploadCard;
