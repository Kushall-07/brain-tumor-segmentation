import { Brain } from 'lucide-react';

export default function Footer() {
  return (
    <footer className="bg-parchment-dark border-t border-sepia-border text-ink-body">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center space-x-2">
            <Brain className="text-brass" size={18} strokeWidth={1.5} />
            <span className="text-sm font-serif text-ink">
              Brain Tumor Segmentation AI
            </span>
          </div>

          <p className="text-xs tracking-wide">
            © {new Date().getFullYear()} Powered by 3D U-Net and MONAI
          </p>

          <div className="flex items-center space-x-2">
            <span className="inline-flex items-center px-2.5 py-1 text-xs font-medium text-annotation border border-sepia-border rounded-sm bg-parchment">
              <span className="w-1.5 h-1.5 bg-annotation rounded-full mr-2" />
              System Online
            </span>
          </div>
        </div>
      </div>
    </footer>
  );
}
