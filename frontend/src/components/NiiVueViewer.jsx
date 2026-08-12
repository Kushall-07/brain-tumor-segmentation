import { useEffect, useRef, useState } from 'react';
import { Niivue, DRAG_MODE } from '@niivue/niivue';
import predictionService from '../services/predictionService';

export default function NiiVueViewer({ mriPath, maskPath, classMasks, onClassChange }) {
  const canvasRef = useRef(null);
  const nvRef = useRef(null);
  const measurementHandlerRef = useRef(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [showOverlay, setShowOverlay] = useState(true);
  const [overlayOpacity, setOverlayOpacity] = useState(50);
  const [activeTool, setActiveTool] = useState('crosshair');
  const [measurementResult, setMeasurementResult] = useState(null);
  const [instructionText, setInstructionText] = useState('');
  const [visibleClasses, setVisibleClasses] = useState([1, 2, 3]);

  useEffect(() => {
    let mriUrl = null;
    let ncrNetUrl = null;
    let edemaUrl = null;
    let etUrl = null;
    let ncrNetFileName = 'NCR/NET.nii.gz';
    let edemaFileName = 'Edema.nii.gz';
    let etFileName = 'ET.nii.gz';

    const loadVolumes = async () => {
      try {
        setLoading(true);
        setError(null);

        console.log('[NiiVueViewer] Starting volume load...');
        console.log('[NiiVueViewer] mriPath:', mriPath);
        console.log('[NiiVueViewer] maskPath:', maskPath);
        console.log('[NiiVueViewer] classMasks:', classMasks);

        // Convert Windows backslashes to forward slashes and extract relative path
        const normalizedMriPath = mriPath.replace(/\\/g, '/');
        const relativeMriPath = normalizedMriPath.replace(/.*outputs\/predictions\//, '');
        const fileName = relativeMriPath.split('/').pop() || 'MRI.nii.gz';

        console.log('[NiiVueViewer] relativeMriPath:', relativeMriPath);
        console.log('[NiiVueViewer] fileName:', fileName);

        // Download MRI
        console.log('[NiiVueViewer] Downloading MRI...');
        const mriBlob = await predictionService.downloadPrediction(relativeMriPath);
        console.log('[NiiVueViewer] MRI blob size:', mriBlob.size);
        mriUrl = URL.createObjectURL(mriBlob);
        console.log('[NiiVueViewer] MRI URL created:', mriUrl);
        console.log('[NiiVueViewer] MRI URL type:', typeof mriUrl);
        console.log('[NiiVueViewer] MRI URL length:', mriUrl.length);

        // Download class-specific masks if available
        let ncrNetBlob = null;
        let edemaBlob = null;
        let etBlob = null;

        if (classMasks && classMasks.ncr_net && classMasks.edema && classMasks.et) {
          try {
            const normalizedNcrNetPath = classMasks.ncr_net.replace(/\\/g, '/');
            const normalizedEdemaPath = classMasks.edema.replace(/\\/g, '/');
            const normalizedEtPath = classMasks.et.replace(/\\/g, '/');

            const relativeNcrNetPath = normalizedNcrNetPath.replace(/.*outputs\/predictions\//, '');
            const relativeEdemaPath = normalizedEdemaPath.replace(/.*outputs\/predictions\//, '');
            const relativeEtPath = normalizedEtPath.replace(/.*outputs\/predictions\//, '');

            ncrNetFileName = relativeNcrNetPath.split('/').pop() || 'NCR/NET.nii.gz';
            edemaFileName = relativeEdemaPath.split('/').pop() || 'Edema.nii.gz';
            etFileName = relativeEtPath.split('/').pop() || 'ET.nii.gz';

            console.log('[NiiVueViewer] Class mask paths:');
            console.log('[NiiVueViewer]   NCR/NET:', relativeNcrNetPath, 'filename:', ncrNetFileName);
            console.log('[NiiVueViewer]   Edema:', relativeEdemaPath, 'filename:', edemaFileName);
            console.log('[NiiVueViewer]   ET:', relativeEtPath, 'filename:', etFileName);

            console.log('[NiiVueViewer] Downloading class masks...');
            [ncrNetBlob, edemaBlob, etBlob] = await Promise.all([
              predictionService.downloadPrediction(relativeNcrNetPath),
              predictionService.downloadPrediction(relativeEdemaPath),
              predictionService.downloadPrediction(relativeEtPath),
            ]);

            console.log('[NiiVueViewer] Class mask blob sizes:');
            console.log('[NiiVueViewer]   NCR/NET:', ncrNetBlob?.size);
            console.log('[NiiVueViewer]   Edema:', edemaBlob?.size);
            console.log('[NiiVueViewer]   ET:', etBlob?.size);

            // Only create URLs if all blobs were successfully downloaded
            if (ncrNetBlob && edemaBlob && etBlob) {
              ncrNetUrl = URL.createObjectURL(ncrNetBlob);
              edemaUrl = URL.createObjectURL(edemaBlob);
              etUrl = URL.createObjectURL(etBlob);

              console.log('[NiiVueViewer] Class mask URLs created');
              console.log('[NiiVueViewer] NCR/NET URL:', ncrNetUrl, 'type:', typeof ncrNetUrl);
              console.log('[NiiVueViewer] Edema URL:', edemaUrl, 'type:', typeof edemaUrl);
              console.log('[NiiVueViewer] ET URL:', etUrl, 'type:', typeof etUrl);
            } else {
              console.warn('[NiiVueViewer] Some class mask blobs are null, falling back to original mask');
            }
          } catch (err) {
            console.warn('[NiiVueViewer] Failed to load class-specific masks, using original mask:', err);
          }
        } else {
          console.log('[NiiVueViewer] No classMasks provided or classMasks incomplete, will use original mask');
        }

        console.log('[NiiVueViewer] Creating Niivue instance...');

        const nv = new Niivue({
          show3Dcrosshair: true,
          isColorbar: true,
          logging: true,  // Enable NiiVue logging for debugging
        });
        nvRef.current = nv;

        console.log('[NiiVueViewer] Attaching to canvas...');
        nv.attachToCanvas(canvasRef.current);

        nv.setMouseEventConfig({
          leftButton: { primary: DRAG_MODE.crosshair },
          rightButton: DRAG_MODE.pan,
          centerButton: DRAG_MODE.pan,
        });

        const handleMeasurementCompleted = (event) => {
          const measurement = event.detail;
          if (measurement && measurement.distance !== undefined) {
            setMeasurementResult({
              type: 'distance',
              value: measurement.distance,
            });
            setInstructionText('');
          }
        };

        measurementHandlerRef.current = handleMeasurementCompleted;
        nv.addEventListener('measurementCompleted', handleMeasurementCompleted);

        // Load MRI volume
        const mriVolume = {
          url: mriUrl,
          name: fileName,
          colormap: 'gray',
          opacity: 1,
          visible: true,
        };

        const volumesToLoad = [mriVolume];
        console.log('[NiiVueViewer] volumesToLoad array initialized with MRI');

        // Load class-specific masks with different colors
        if (ncrNetUrl && edemaUrl && etUrl) {
          console.log('[NiiVueViewer] Adding 3 class-specific masks to volumes array');
          // NCR/NET - red
          volumesToLoad.push({
            url: ncrNetUrl,
            name: ncrNetFileName,
            colormap: 'red',
            opacity: overlayOpacity / 100,
            visible: true,
            isLabelMap: true,
          });

          // Edema - green
          volumesToLoad.push({
            url: edemaUrl,
            name: edemaFileName,
            colormap: 'green',
            opacity: overlayOpacity / 100,
            visible: true,
            isLabelMap: true,
          });

          // ET - magenta/pink
          volumesToLoad.push({
            url: etUrl,
            name: etFileName,
            colormap: 'violet',
            opacity: 0.8,
            visible: true,
            isLabelMap: true,
          });
          
          // Set initial visibility based on visibleClasses
          // Note: We'll set this after loading
        } else {
          // Fallback: load original single mask
          console.log('[NiiVueViewer] Loading original single mask as fallback');
          const normalizedMaskPath = maskPath.replace(/\\/g, '/');
          const relativeMaskPath = normalizedMaskPath.replace(/.*outputs\/predictions\//, '');
          console.log('[NiiVueViewer] relativeMaskPath:', relativeMaskPath);
          const maskBlob = await predictionService.downloadPrediction(relativeMaskPath);
          console.log('[NiiVueViewer] Mask blob size:', maskBlob.size);
          const maskUrl = URL.createObjectURL(maskBlob);
          const maskFileName = relativeMaskPath.split('/').pop() || 'Segmentation.nii.gz';

          volumesToLoad.push({
            url: maskUrl,
            name: maskFileName,
            colormap: 'red',
            opacity: overlayOpacity / 100,
            visible: showOverlay,
            isLabelMap: true,
          });
        }

        console.log('[NiiVueViewer] Total volumes to load:', volumesToLoad.length);
        
        // Diagnostic logging for volume names
        volumesToLoad.forEach((v, i) => {
          console.log(
            `[NiiVueViewer] Volume ${i} name:`,
            v.name,
            '| url:',
            v.url
          );
        });
        
        // Print COMPLETE volumesToLoad for debugging
        console.log(
          '[NiiVueViewer] COMPLETE volumesToLoad:',
          JSON.stringify(
            volumesToLoad,
            (key, value) => {
              if (value instanceof Blob) {
                return {
                  type: 'Blob',
                  size: value.size,
                  mime: value.type
                };
              }
              return value;
            },
            2
          )
        );
        
        // Print each volume individually
        volumesToLoad.forEach((volume, index) => {
          console.log(`[NiiVueViewer] VOLUME ${index}`, volume);
          console.log(`[NiiVueViewer] VOLUME ${index} url:`, volume?.url);
          console.log(`[NiiVueViewer] VOLUME ${index} url type:`, typeof volume?.url);
          console.log(`[NiiVueViewer] VOLUME ${index} name:`, volume?.name);
        });
        
        // Strict validation before calling nv.loadVolumes
        volumesToLoad.forEach((volume, index) => {
          if (!volume) {
            throw new Error(`Volume ${index} is undefined`);
          }

          if (typeof volume.url !== 'string') {
            throw new Error(
              `Volume ${index} has invalid URL. Type=${typeof volume.url}, value=${volume.url}` 
            );
          }

          if (!volume.url.trim()) {
            throw new Error(`Volume ${index} has an empty URL`);
          }

          if (!volume.name || typeof volume.name !== 'string' || !volume.name.trim()) {
            throw new Error(
              `Volume ${index} has invalid name. Type=${typeof volume.name}, value=${volume.name}` 
            );
          }

          console.log(
            `[NiiVueViewer] VALID VOLUME ${index}:`,
            volume.url,
            volume.name
          );
        });
        
        console.log('[NiiVueViewer] All volume URLs and names validated successfully');
        
        // Diagnostic logging before nv.loadVolumes
        volumesToLoad.forEach((volume, index) => {
          console.log(
            `[NiiVueViewer] FINAL VOLUME ${index}:`,
            {
              url: volume.url,
              name: volume.name,
              colormap: volume.colormap,
              opacity: volume.opacity,
            }
          );
        });
        
        console.log('[NiiVueViewer] Calling nv.loadVolumes...');
        await nv.loadVolumes(volumesToLoad);
        console.log('[NiiVueViewer] nv.loadVolumes completed');
        console.log('[NiiVueViewer] nv.volumes.length after load:', nv.volumes.length);

        // Set initial visibility for class-specific masks
        if (ncrNetUrl && edemaUrl && etUrl) {
          console.log('[NiiVueViewer] Setting initial visibility for class-specific masks');
          // Volume indices: 0 = MRI, 1 = NCR/NET, 2 = Edema, 3 = ET
          if (nv.volumes.length > 1) {
            const ncrNetOpacity = visibleClasses.includes(1) ? overlayOpacity / 100 : 0;
            nv.setOpacity(1, ncrNetOpacity);
            console.log('[NiiVueViewer] NCR/NET opacity set to:', ncrNetOpacity);
          }
          if (nv.volumes.length > 2) {
            const edemaOpacity = visibleClasses.includes(2) ? overlayOpacity / 100 : 0;
            nv.setOpacity(2, edemaOpacity);
            console.log('[NiiVueViewer] Edema opacity set to:', edemaOpacity);
          }
          if (nv.volumes.length > 3) {
            // ET uses fixed 0.8 opacity for better visibility
            const etOpacity = visibleClasses.includes(3) ? 0.8 : 0;
            nv.setOpacity(3, etOpacity);
            console.log('[NiiVueViewer] ET opacity set to:', etOpacity);
          }
        }

        console.log('[NiiVueViewer] Volume loading completed successfully');
        setLoading(false);
      } catch (err) {
        console.error('[NiiVueViewer] Failed to load volumes:', err);
        console.error('[NiiVueViewer] Error details:', err.message, err.stack);
        setError('Failed to load MRI visualization');
        setLoading(false);
      }
    };

    if (mriPath && maskPath) {
      loadVolumes();
    }

    return () => {
      if (mriUrl) URL.revokeObjectURL(mriUrl);
      if (ncrNetUrl) URL.revokeObjectURL(ncrNetUrl);
      if (edemaUrl) URL.revokeObjectURL(edemaUrl);
      if (etUrl) URL.revokeObjectURL(etUrl);
      if (nvRef.current) {
        if (measurementHandlerRef.current) {
          nvRef.current.removeEventListener('measurementCompleted', measurementHandlerRef.current);
        }
        nvRef.current = null;
      }
    };
  }, [mriPath, maskPath, classMasks]);

  // Update volume visibility (handles both class toggles and opacity slider)
  useEffect(() => {
    if (nvRef.current && nvRef.current.volumes.length > 1) {
      const opacity = showOverlay ? overlayOpacity / 100 : 0;
      // Update all segmentation volumes (indices 1, 2, 3)
      for (let i = 1; i < nvRef.current.volumes.length; i++) {
        const classId = i; // 1 -> NCR/NET, 2 -> Edema, 3 -> ET
        if (visibleClasses.includes(classId)) {
          nvRef.current.setOpacity(i, opacity);
        } else {
          nvRef.current.setOpacity(i, 0);
        }
      }
    }
  }, [visibleClasses, showOverlay, overlayOpacity]);

  // Handle class visibility toggle by changing volume visibility
  const handleClassToggle = (classId) => {
    const newVisibleClasses = visibleClasses.includes(classId)
      ? visibleClasses.filter(id => id !== classId)
      : [...visibleClasses, classId];
    
    setVisibleClasses(newVisibleClasses);
    
    // Notify parent component for analysis updates
    if (onClassChange) {
      onClassChange(newVisibleClasses);
    }
    
    // Update NiiVue volume visibility using opacity
    if (nvRef.current && nvRef.current.volumes.length > 1) {
      // Volume indices: 0 = MRI, 1 = NCR/NET, 2 = Edema, 3 = ET
      const volumeIndex = classId; // 1 -> NCR/NET, 2 -> Edema, 3 -> ET
      if (volumeIndex < nvRef.current.volumes.length) {
        // ET (class 3) uses fixed 0.8 opacity for better visibility
        const opacity = classId === 3 
          ? (newVisibleClasses.includes(classId) ? 0.8 : 0)
          : (newVisibleClasses.includes(classId) ? overlayOpacity / 100 : 0);
        nvRef.current.setOpacity(volumeIndex, opacity);
      }
    }
  };

  const handleToggleOverlay = () => {
    setShowOverlay((current) => !current);
  };

  const handleOpacityChange = (e) => {
    const newOpacity = parseInt(e.target.value, 10);
    setOverlayOpacity(newOpacity);
    
    // Update overlay opacities in NiiVue
    if (nvRef.current && nvRef.current.volumes.length > 1 && showOverlay) {
      // Volume indices: 0 = MRI, 1 = NCR/NET, 2 = Edema, 3 = ET
      if (nvRef.current.volumes.length > 1) {
        const ncrNetOpacity = visibleClasses.includes(1) ? newOpacity / 100 : 0;
        nvRef.current.setOpacity(1, ncrNetOpacity);
      }
      if (nvRef.current.volumes.length > 2) {
        const edemaOpacity = visibleClasses.includes(2) ? newOpacity / 100 : 0;
        nvRef.current.setOpacity(2, edemaOpacity);
      }
      // ET uses fixed 0.8 opacity for better visibility
      if (nvRef.current.volumes.length > 3) {
        const etOpacity = visibleClasses.includes(3) ? 0.8 : 0;
        nvRef.current.setOpacity(3, etOpacity);
      }
    }
  };

  const handleToolChange = (tool) => {
    if (!nvRef.current) return;

    setActiveTool(tool);
    setMeasurementResult(null);

    switch (tool) {
      case 'crosshair':
        nvRef.current.setMouseEventConfig({
          leftButton: { primary: DRAG_MODE.crosshair },
          rightButton: DRAG_MODE.pan,
          centerButton: DRAG_MODE.pan,
        });
        setInstructionText('');
        break;
      case 'ruler':
        nvRef.current.setMouseEventConfig({
          leftButton: { primary: DRAG_MODE.measurement },
          rightButton: DRAG_MODE.pan,
          centerButton: DRAG_MODE.pan,
        });
        setInstructionText('Draw between two points to measure distance');
        break;
    }
  };

  const handleClearMeasurements = () => {
    if (!nvRef.current) return;
    nvRef.current.clearAllMeasurements();
    setMeasurementResult(null);
    setInstructionText(activeTool === 'ruler' ? 'Draw between two points to measure distance' : '');
  };

  return (
    <div className="w-full rounded-xl border border-stone-200 overflow-hidden bg-white">
      <div className="flex flex-col gap-3 border-b border-stone-200 bg-stone-50 px-4 py-3 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h3 className="text-base font-semibold text-stone-900">MRI Viewer</h3>
          <p className="text-sm text-stone-500">
            Multi-planar MRI with segmentation overlay
          </p>
        </div>
        <span className="inline-flex w-fit items-center rounded-full border border-stone-200 bg-white px-3 py-1 text-xs font-medium text-stone-600">
          4-View
        </span>
      </div>

      {!loading && !error && (
        <div className="flex flex-wrap items-center gap-x-6 gap-y-3 border-b border-stone-200 bg-white px-4 py-3">
          <div className="flex items-center gap-2">
            <span className="text-sm font-medium text-stone-700">Tools</span>
            <div className="flex items-center gap-1">
              <button
                type="button"
                onClick={() => handleToolChange('crosshair')}
                aria-pressed={activeTool === 'crosshair'}
                className={`inline-flex items-center rounded-md border px-3 py-1.5 text-sm font-medium transition-colors ${activeTool === 'crosshair' ? 'border-teal-300 bg-teal-50 text-teal-800' : 'border-stone-200 bg-white text-stone-600 hover:bg-stone-50'}`}
              >
                Crosshair
              </button>
              <button
                type="button"
                onClick={() => handleToolChange('ruler')}
                aria-pressed={activeTool === 'ruler'}
                className={`inline-flex items-center rounded-md border px-3 py-1.5 text-sm font-medium transition-colors ${activeTool === 'ruler' ? 'border-teal-300 bg-teal-50 text-teal-800' : 'border-stone-200 bg-white text-stone-600 hover:bg-stone-50'}`}
              >
                Ruler
              </button>
            </div>
          </div>

          {measurementResult && (
            <div className="flex items-center gap-2 rounded-md border border-stone-200 bg-stone-50 px-3 py-1.5">
              <span className="text-sm font-medium text-stone-700">Distance</span>
              <span className="text-sm font-semibold text-stone-900 tabular-nums">
                {`${measurementResult.value.toFixed(2)} mm`}
              </span>
            </div>
          )}

          {instructionText && !measurementResult && (
            <div className="text-sm text-stone-500">{instructionText}</div>
          )}

          {activeTool === 'ruler' && (
            <button
              type="button"
              onClick={handleClearMeasurements}
              className="inline-flex items-center rounded-md border border-stone-200 px-3 py-1.5 text-sm font-medium text-stone-600 transition-colors hover:bg-stone-50"
            >
              Clear Measurements
            </button>
          )}

          <div className="hidden flex-1 lg:block" aria-hidden="true" />

          <div className="flex items-center gap-3">
            <span className="text-sm font-medium text-stone-700">Segmentation</span>
            <button
              type="button"
              onClick={handleToggleOverlay}
              aria-pressed={showOverlay}
              className={`inline-flex items-center gap-2 rounded-md border px-3 py-1.5 text-sm font-medium transition-colors ${showOverlay ? 'border-teal-200 bg-teal-50 text-teal-800' : 'border-stone-200 bg-stone-100 text-stone-600'}`}
            >
              <span className={`relative inline-flex h-5 w-9 shrink-0 items-center rounded-full transition-colors ${showOverlay ? 'bg-teal-600' : 'bg-stone-300'}`}>
                <span className={`inline-block h-4 w-4 transform rounded-full bg-white shadow transition-transform ${showOverlay ? 'translate-x-4' : 'translate-x-0.5'}`} />
              </span>
              <span>{showOverlay ? 'ON' : 'OFF'}</span>
            </button>
          </div>

          <div className="flex min-w-[220px] flex-1 flex-wrap items-center gap-3 sm:flex-none">
            <label htmlFor="overlay-opacity" className={`text-sm font-medium ${showOverlay ? 'text-stone-700' : 'text-stone-400'}`}>
              Overlay Opacity
            </label>
            <input
              id="overlay-opacity"
              type="range"
              min="0"
              max="100"
              value={overlayOpacity}
              onChange={handleOpacityChange}
              disabled={!showOverlay}
              className={`h-2 w-36 appearance-none rounded-lg sm:w-44 ${showOverlay ? 'cursor-pointer bg-stone-200 accent-teal-600' : 'cursor-not-allowed bg-stone-100 accent-stone-300 opacity-60'}`}
            />
            <span className={`min-w-[3rem] text-sm font-medium tabular-nums ${showOverlay ? 'text-stone-700' : 'text-stone-400'}`}>
              {overlayOpacity}%
            </span>
          </div>

          <div className="flex items-center gap-3 border-l border-stone-200 pl-4">
            <span className="text-sm font-medium text-stone-700">Classes</span>
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={() => handleClassToggle(1)}
                aria-pressed={visibleClasses.includes(1)}
                className={`inline-flex items-center gap-1.5 rounded-md border px-2.5 py-1 text-xs font-medium transition-colors ${visibleClasses.includes(1) ? 'border-red-300 bg-red-50 text-red-800' : 'border-stone-200 bg-stone-50 text-stone-500'}`}
              >
                <span className="w-2.5 h-2.5 rounded-full bg-red-500" />
                NCR/NET
              </button>
              <button
                type="button"
                onClick={() => handleClassToggle(2)}
                aria-pressed={visibleClasses.includes(2)}
                className={`inline-flex items-center gap-1.5 rounded-md border px-2.5 py-1 text-xs font-medium transition-colors ${visibleClasses.includes(2) ? 'border-green-300 bg-green-50 text-green-800' : 'border-stone-200 bg-stone-50 text-stone-500'}`}
              >
                <span className="w-2.5 h-2.5 rounded-full bg-green-500" />
                Edema
              </button>
              <button
                type="button"
                onClick={() => handleClassToggle(3)}
                aria-pressed={visibleClasses.includes(3)}
                className={`inline-flex items-center gap-1.5 rounded-md border px-2.5 py-1 text-xs font-medium transition-colors ${visibleClasses.includes(3) ? 'border-fuchsia-300 bg-fuchsia-50 text-fuchsia-800' : 'border-stone-200 bg-stone-50 text-stone-500'}`}
              >
                <span className="w-2.5 h-2.5 rounded-full bg-fuchsia-500" />
                ET
              </button>
            </div>
          </div>
        </div>
      )}

      <div className="relative h-[620px] w-full bg-slate-900">
        <canvas ref={canvasRef} className="h-full w-full" />

        {loading && (
          <div className="absolute inset-0 flex items-center justify-center bg-slate-900/90">
            <div className="text-center">
              <div className="mb-3 inline-block h-8 w-8 animate-spin rounded-full border-2 border-teal-600 border-t-transparent" />
              <p className="text-stone-400">Loading MRI visualization...</p>
            </div>
          </div>
        )}

        {error && (
          <div className="absolute inset-0 flex items-center justify-center bg-slate-900/90">
            <div className="text-center">
              <p className="mb-2 text-red-400">Failed to load visualization</p>
              <p className="text-sm text-stone-500">{error}</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
