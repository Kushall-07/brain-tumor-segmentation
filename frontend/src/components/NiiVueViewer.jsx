import { useEffect, useRef, useState } from 'react';
import { Niivue, DRAG_MODE } from '@niivue/niivue';
import predictionService from '../services/predictionService';

export default function NiiVueViewer({ mriPath, maskPath, tumorDimensions, tumorMeasurementGeometry }) {
  const canvasRef = useRef(null);
  const nvRef = useRef(null);
  const measurementHandlerRef = useRef(null);
  const measurementMeshRef = useRef(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [showOverlay, setShowOverlay] = useState(true);
  const [overlayOpacity, setOverlayOpacity] = useState(50);
  const [show3DLength, setShow3DLength] = useState(true);
  const [activeTool, setActiveTool] = useState('crosshair'); // 'crosshair' | 'ruler'
  const [measurementResult, setMeasurementResult] = useState(null); // { type: 'distance', value: number }
  const [instructionText, setInstructionText] = useState('');

  useEffect(() => {
    let mriUrl = null;
    let maskUrl = null;
    let cancelled = false; // StrictMode safety: prevents stale async runs after cleanup

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
          meshXRay: 0.6,
        });
        nvRef.current = nv;

        // Attach to canvas
        nv.attachToCanvas(canvasRef.current);

        // Set default mouse event config for crosshair tool
        nv.setMouseEventConfig({
          leftButton: { primary: DRAG_MODE.crosshair },
          rightButton: DRAG_MODE.pan,
          centerButton: DRAG_MODE.pan,
        });

        // Register measurement event listener
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

        // StrictMode safety: if cleanup ran during await, abort this run
        if (cancelled) {
          return;
        }

        // Create 3D length measurement mesh if geometry is available
        if (tumorMeasurementGeometry && tumorMeasurementGeometry.length) {
          const geometry = tumorMeasurementGeometry.length;

          // Validate geometry before creating mesh
          const isValidGeometry = (
            Array.isArray(geometry.start_mm) &&
            geometry.start_mm.length === 3 &&
            geometry.start_mm.every(v => isFinite(v)) &&
            Array.isArray(geometry.end_mm) &&
            geometry.end_mm.length === 3 &&
            geometry.end_mm.every(v => isFinite(v)) &&
            isFinite(geometry.value_mm) &&
            geometry.value_mm > 0
          );
          
          if (!isValidGeometry) {
            console.warn('Invalid tumor measurement geometry, skipping visualization');
            return;
          }
          
          const connectome = {
            name: 'tumor_length_measurement',
            nodeColormap: 'blue',
            nodeColormapNegative: 'winter',
            nodeMinColor: 0,
            nodeMaxColor: 1,
            nodeScale: 1.2,
            edgeColormap: 'blue',
            edgeColormapNegative: 'winter',
            edgeMin: 0,
            edgeMax: 1,
            edgeScale: 2,
            legendLineThickness: 0,
            showLegend: false,
            nodes: [
              {
                name: 'start',
                x: geometry.start_mm[0],
                y: geometry.start_mm[1],
                z: geometry.start_mm[2],
                colorValue: 0.5,
                sizeValue: 0.8
              },
              {
                name: 'end',
                x: geometry.end_mm[0],
                y: geometry.end_mm[1],
                z: geometry.end_mm[2],
                colorValue: 0.5,
                sizeValue: 0.8
              }
            ],
            edges: [
              {
                first: 0,
                second: 1,
                colorValue: 0.5
              }
            ]
          };

          try {
            const measurementMesh = await nv.loadConnectomeAsMesh(connectome);

            // StrictMode safety: if cleanup ran during await, abort this run
            if (cancelled) {
              return;
            }

            measurementMesh.colorbarVisible = false;
            measurementMeshRef.current = measurementMesh;
            nv.addMesh(measurementMesh);

            // Force visibility
            measurementMesh.visible = true;
            measurementMesh.colorbarVisible = false;
            nv.drawScene();
          } catch (err) {
            console.error('Failed to create measurement mesh:', err);
          }
        }

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
      // Mark this run as cancelled (StrictMode safety)
      cancelled = true;
      // Clean up object URLs
      if (mriUrl) URL.revokeObjectURL(mriUrl);
      if (maskUrl) URL.revokeObjectURL(maskUrl);
      if (nvRef.current) {
        // Remove event listener
        if (measurementHandlerRef.current) {
          nvRef.current.removeEventListener('measurementCompleted', measurementHandlerRef.current);
        }
        // Remove measurement mesh if exists
        if (measurementMeshRef.current) {
          nvRef.current.removeMesh(measurementMeshRef.current);
        }
        nvRef.current = null;
        measurementMeshRef.current = null;
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

  // Update measurement mesh visibility when overlay or 3D length toggle changes
  useEffect(() => {
    if (nvRef.current && measurementMeshRef.current) {
      const effectiveVisibility = showOverlay && show3DLength;
      measurementMeshRef.current.visible = effectiveVisibility;
      nvRef.current.drawScene();
    }
  }, [showOverlay, show3DLength]);

  const handleToggleOverlay = () => {
    setShowOverlay((current) => !current);
  };

  const handleOpacityChange = (e) => {
    setOverlayOpacity(parseInt(e.target.value, 10));
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

  const handleToggle3DLength = () => {
    setShow3DLength(!show3DLength);
  };

  return (
    <div className="w-full rounded-xl border border-stone-200 overflow-hidden bg-white">
      {/* Viewer Header */}
      <div className="flex flex-col gap-3 border-b border-stone-200 bg-stone-50 px-4 py-3 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h3 className="text-base font-semibold text-stone-900">MRI Viewer</h3>
          <p className="text-sm text-stone-500">
            Multi-planar MRI with segmentation overlay
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          {tumorDimensions && (
            <span className="inline-flex items-center gap-1.5 rounded-full border border-teal-200 bg-teal-50 px-3 py-1 text-xs font-semibold text-teal-800">
              <span className="text-teal-600 font-normal">Auto 3D Dimensions:</span>
              {tumorDimensions.length} &times; {tumorDimensions.width} &times; {tumorDimensions.height} mm
            </span>
          )}
          <span className="inline-flex w-fit items-center rounded-full border border-stone-200 bg-white px-3 py-1 text-xs font-medium text-stone-600">
            4-View
          </span>
        </div>
      </div>

      {/* Toolbar */}
      {!loading && !error && (
        <div className="flex flex-wrap items-center gap-x-6 gap-y-3 border-b border-stone-200 bg-white px-4 py-3">
          {/* Measurement Tools */}
          <div className="flex items-center gap-2">
            <span className="text-sm font-medium text-stone-700">Tools</span>
            <div className="flex items-center gap-1">
              <button
                type="button"
                onClick={() => handleToolChange('crosshair')}
                aria-pressed={activeTool === 'crosshair'}
                className={`inline-flex items-center rounded-md border px-3 py-1.5 text-sm font-medium transition-colors ${
                  activeTool === 'crosshair'
                    ? 'border-teal-300 bg-teal-50 text-teal-800'
                    : 'border-stone-200 bg-white text-stone-600 hover:bg-stone-50'
                }`}
              >
                Crosshair
              </button>
              <button
                type="button"
                onClick={() => handleToolChange('ruler')}
                aria-pressed={activeTool === 'ruler'}
                className={`inline-flex items-center rounded-md border px-3 py-1.5 text-sm font-medium transition-colors ${
                  activeTool === 'ruler'
                    ? 'border-teal-300 bg-teal-50 text-teal-800'
                    : 'border-stone-200 bg-white text-stone-600 hover:bg-stone-50'
                }`}
              >
                Ruler
              </button>
            </div>
          </div>

          {/* Measurement Result */}
          {measurementResult && (
            <div className="flex items-center gap-2 rounded-md border border-stone-200 bg-stone-50 px-3 py-1.5">
              <span className="text-sm font-medium text-stone-700">
                Distance
              </span>
              <span className="text-sm font-semibold text-stone-900 tabular-nums">
                {`${measurementResult.value.toFixed(2)} mm`}
              </span>
            </div>
          )}

          {/* Instruction Text */}
          {instructionText && !measurementResult && (
            <div className="text-sm text-stone-500">
              {instructionText}
            </div>
          )}

          {/* Clear Measurements */}
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

          {/* 3D Length Toggle */}
          {tumorMeasurementGeometry && (
            <div className="flex items-center gap-2">
              <span className="text-sm font-medium text-stone-700">3D Length</span>
              <button
                type="button"
                onClick={handleToggle3DLength}
                aria-pressed={show3DLength}
                className={`inline-flex items-center gap-2 rounded-md border px-3 py-1.5 text-sm font-medium transition-colors ${
                  show3DLength
                    ? 'border-teal-200 bg-teal-50 text-teal-800'
                    : 'border-stone-200 bg-stone-100 text-stone-600'
                }`}
              >
                <span
                  className={`relative inline-flex h-5 w-9 shrink-0 items-center rounded-full transition-colors ${
                    show3DLength ? 'bg-teal-600' : 'bg-stone-300'
                  }`}
                >
                  <span
                    className={`inline-block h-4 w-4 transform rounded-full bg-white shadow transition-transform ${
                      show3DLength ? 'translate-x-4' : 'translate-x-0.5'
                    }`}
                  />
                </span>
                <span>{show3DLength ? 'ON' : 'OFF'}</span>
              </button>
            </div>
          )}

          <div className="hidden flex-1 lg:block" aria-hidden="true" />

          {/* Segmentation Controls */}
          <div className="flex items-center gap-3">
            <span className="text-sm font-medium text-stone-700">Segmentation</span>
            <button
              type="button"
              onClick={handleToggleOverlay}
              aria-pressed={showOverlay}
              className={`inline-flex items-center gap-2 rounded-md border px-3 py-1.5 text-sm font-medium transition-colors ${
                showOverlay
                  ? 'border-teal-200 bg-teal-50 text-teal-800'
                  : 'border-stone-200 bg-stone-100 text-stone-600'
              }`}
            >
              <span
                className={`relative inline-flex h-5 w-9 shrink-0 items-center rounded-full transition-colors ${
                  showOverlay ? 'bg-teal-600' : 'bg-stone-300'
                }`}
              >
                <span
                  className={`inline-block h-4 w-4 transform rounded-full bg-white shadow transition-transform ${
                    showOverlay ? 'translate-x-4' : 'translate-x-0.5'
                  }`}
                />
              </span>
              <span>{showOverlay ? 'ON' : 'OFF'}</span>
            </button>
          </div>

          <div className="flex min-w-[220px] flex-1 flex-wrap items-center gap-3 sm:flex-none">
            <label
              htmlFor="overlay-opacity"
              className={`text-sm font-medium ${showOverlay ? 'text-stone-700' : 'text-stone-400'}`}
            >
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
              className={`h-2 w-36 appearance-none rounded-lg sm:w-44 ${
                showOverlay
                  ? 'cursor-pointer bg-stone-200 accent-teal-600'
                  : 'cursor-not-allowed bg-stone-100 accent-stone-300 opacity-60'
              }`}
            />
            <span className={`min-w-[3rem] text-sm font-medium tabular-nums ${showOverlay ? 'text-stone-700' : 'text-stone-400'}`}>
              {overlayOpacity}%
            </span>
          </div>
        </div>
      )}

      {/* NiiVue Canvas */}
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
