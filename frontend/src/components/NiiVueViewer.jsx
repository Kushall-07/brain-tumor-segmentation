import { useEffect, useRef, useState } from 'react';
import { Niivue } from '@niivue/niivue';
import { Eye, EyeOff } from 'lucide-react';
import predictionService from '../services/predictionService';

export default function NiiVueViewer({ mriPath, maskPath }) {
  const canvasRef = useRef(null);
  const nvRef = useRef(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [showOverlay, setShowOverlay] = useState(true);
  const [overlayOpacity, setOverlayOpacity] = useState(50);

  useEffect(() => {
    let mriUrl = null;
    let maskUrl = null;

    const loadVolumes = async () => {
      try {
        setLoading(true);
        setError(null);

        // Convert full paths to relative paths for download endpoint
        const relativeMriPath = mriPath.replace('outputs/predictions/', '');
        const relativeMaskPath = maskPath.replace('outputs/predictions/', '');

        // Extract filenames
        const fileName = relativeMriPath.split('/').pop();
        const maskFileName = relativeMaskPath.split('/').pop();

        // Download both files as blobs
        const [mriBlob, maskBlob] = await Promise.all([
          predictionService.downloadPrediction(relativeMriPath),
          predictionService.downloadPrediction(relativeMaskPath),
        ]);

        // Create object URLs
        mriUrl = URL.createObjectURL(mriBlob);
        maskUrl = URL.createObjectURL(maskBlob);

        // Initialize NiiVue
        const nv = new Niivue({
          show3Dcrosshair: true,
          isColorbar: true,
          logging: false,
        });
        nvRef.current = nv;

        // Attach to canvas
        nv.attachToCanvas(canvasRef.current);

        // Create volume objects with proper configuration
        const mriVolume = {
          url: mriUrl,
          name: fileName,
          colormap: 'gray',
          opacity: 1,
          visible: true,
        };

        const maskVolume = {
          url: maskUrl,
          name: maskFileName,
          colormap: 'red',
          opacity: overlayOpacity / 100,
          visible: showOverlay,
          isLabelMap: true,
        };

        // Load both volumes
        await nv.loadVolumes([mriVolume, maskVolume]);

        console.log("=== Visibility-related methods ===");
        console.log(
          Object.keys(nv).filter(k =>
            k.toLowerCase().includes("visible") ||
            k.toLowerCase().includes("show") ||
            k.toLowerCase().includes("hide") ||
            k.toLowerCase().includes("volume")
          )
        );

        console.log("=== Segmentation volume properties ===");
        console.dir(nv.volumes[1]);

        setLoading(false);
      } catch (err) {
        console.error('Failed to load volumes:', err);
        setError('Failed to load MRI visualization');
        setLoading(false);
      }
    };

    if (mriPath && maskPath) {
      loadVolumes();
    }

    return () => {
      // Clean up object URLs
      if (mriUrl) URL.revokeObjectURL(mriUrl);
      if (maskUrl) URL.revokeObjectURL(maskUrl);
      if (nvRef.current) {
        nvRef.current = null;
      }
    };
  }, [mriPath, maskPath]);

  // Update overlay visibility
  useEffect(() => {
    if (nvRef.current && nvRef.current.volumes.length > 1) {
      const opacity = showOverlay ? overlayOpacity / 100 : 0;
      nvRef.current.setOpacity(1, opacity);
    }
  }, [showOverlay, overlayOpacity]);

  const handleToggleOverlay = () => {
    setShowOverlay(!showOverlay);
  };

  const handleOpacityChange = (e) => {
    setOverlayOpacity(parseInt(e.target.value));
  };

  return (
    <div className="relative w-full h-[650px] bg-slate-900 rounded-xl border border-slate-800 overflow-hidden">
      <canvas ref={canvasRef} className="w-full h-full" />
      
      {/* Controls Overlay */}
      {!loading && !error && (
        <div className="absolute top-4 right-4 bg-slate-800/90 backdrop-blur-sm rounded-lg p-4 border border-slate-700">
          <div className="flex flex-col gap-3">
            {/* Toggle Button */}
            <button
              onClick={handleToggleOverlay}
              className="flex items-center gap-2 px-3 py-2 bg-slate-700 hover:bg-slate-600 rounded-md transition-colors text-white text-sm"
            >
              {showOverlay ? (
                <>
                  <Eye size={16} />
                  <span>Hide Segmentation</span>
                </>
              ) : (
                <>
                  <EyeOff size={16} />
                  <span>Show Segmentation</span>
                </>
              )}
            </button>
            
            {/* Opacity Slider */}
            <div className="flex flex-col gap-1">
              <label className="text-xs text-slate-300">Overlay Opacity: {overlayOpacity}%</label>
              <input
                type="range"
                min="0"
                max="100"
                value={overlayOpacity}
                onChange={handleOpacityChange}
                className="w-full h-2 bg-slate-600 rounded-lg appearance-none cursor-pointer accent-cyan-500"
              />
            </div>
          </div>
        </div>
      )}
      
      {loading && (
        <div className="absolute inset-0 bg-slate-900/90 flex items-center justify-center">
          <div className="text-center">
            <div className="inline-block w-8 h-8 border-2 border-cyan-500 border-t-transparent rounded-full animate-spin mb-3"></div>
            <p className="text-slate-400">Loading MRI visualization...</p>
          </div>
        </div>
      )}
      
      {error && (
        <div className="absolute inset-0 bg-slate-900/90 flex items-center justify-center">
          <div className="text-center">
            <p className="text-red-400 mb-2">Failed to load visualization</p>
            <p className="text-slate-500 text-sm">{error}</p>
          </div>
        </div>
      )}
    </div>
  );
}
