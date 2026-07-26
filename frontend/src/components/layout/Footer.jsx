import { Brain } from 'lucide-react';

export default function Footer() {
  return (
    <footer className="bg-slate-900 border-t border-slate-800 text-slate-400">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
          {/* Logo and Title */}
          <div className="flex items-center space-x-2">
            <Brain className="text-cyan-500" size={20} />
            <span className="text-sm font-medium text-slate-300">
              Brain Tumor Segmentation AI
            </span>
          </div>

          {/* Copyright */}
          <p className="text-xs text-slate-500">
            © {new Date().getFullYear()} Powered by 3D U-Net and MONAI
          </p>

          {/* Version/Status */}
          <div className="flex items-center space-x-2">
            <span className="inline-flex items-center px-2 py-1 text-xs font-medium text-green-400 bg-green-900/30 rounded-full">
              <span className="w-2 h-2 bg-green-500 rounded-full mr-1.5 animate-pulse"></span>
              System Online
            </span>
          </div>
        </div>
      </div>
    </footer>
  );
}
