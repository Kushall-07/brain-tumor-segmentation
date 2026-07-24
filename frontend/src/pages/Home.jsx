import { Link, useNavigate } from 'react-router-dom';
import FileUpload from '../components/FileUpload';

export default function Home() {
  const navigate = useNavigate();

  const handleFilesUploaded = (files) => {
    navigate('/processing', { state: { files } });
  };

  const handleProcessingStart = () => {
    // Navigation handled by handleFilesUploaded
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
      <div className="text-center mb-16">
        <h1 className="text-5xl md:text-7xl font-bold bg-linear-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent mb-6">
          Brain Tumor Segmentation AI
        </h1>
        <p className="text-xl md:text-2xl text-slate-300 max-w-3xl mx-auto leading-relaxed">
          Upload multi-modal MRI scans and get AI-powered tumor segmentation
          using state-of-the-art SwinUNETR architecture. Supports T1c, T1n, T2f, and T2w modalities.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-16">
        <FeatureCard
          icon={
            <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3 3m0 0l-3-3m3 3V12" />
            </svg>
          }
          title="Multi-Modal Upload"
          description="Drag & drop or click to upload all four MRI modalities (T1c, T1n, T2f, T2w) with automatic validation and format checking."
        />
        <FeatureCard
          icon={
            <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
          }
          title="AI-Powered Segmentation"
          description="State-of-the-art SwinUNETR transformer architecture for accurate 3D brain tumor segmentation with Dice scores >0.85."
        />
        <FeatureCard
          icon={
            <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
          }
          title="Interactive 3D Visualization"
          description="Explore results with multi-planar views (axial, coronal, sagittal), adjustable overlays, and volumetric measurements."
        />
      </div>

      <div className="max-w-4xl mx-auto">
        <FileUpload
          onFilesUploaded={handleFilesUploaded}
          onProcessingStart={handleProcessingStart}
        />
      </div>

      <div className="mt-16 grid grid-cols-1 md:grid-cols-3 gap-8 text-center">
        <div className="p-6 bg-slate-800/50 rounded-xl border border-slate-700">
          <div className="text-4xl font-bold text-cyan-400 mb-2">4</div>
          <div className="text-slate-400">Required Modalities</div>
        </div>
        <div className="p-6 bg-slate-800/50 rounded-xl border border-slate-700">
          <div className="text-4xl font-bold text-cyan-400 mb-2">3</div>
          <div className="text-slate-400">Tumor Classes</div>
        </div>
        <div className="p-6 bg-slate-800/50 rounded-xl border border-slate-700">
          <div className="text-4xl font-bold text-cyan-400 mb-2">{"<3s"}</div>
          <div className="text-slate-400">Processing Time</div>
        </div>
      </div>
    </div>
  );
}

function FeatureCard({ icon, title, description }) {
  return (
    <div className="p-6 bg-slate-800/50 rounded-xl border border-slate-700 hover:border-cyan-500/50 transition-colors">
      <div className="w-16 h-16 rounded-xl bg-cyan-500/10 flex items-center justify-center text-cyan-400 mb-4">
        {icon}
      </div>
      <h3 className="text-xl font-semibold text-white mb-2">{title}</h3>
      <p className="text-slate-400">{description}</p>
    </div>
  );
}