import { useState, useEffect, useRef } from 'react';

const tumorClasses = [
  { id: 0, name: 'Background', color: '#000000' },
  { id: 1, name: 'Necrotic Core (NCR/NET)', color: '#FF0000' },
  { id: 2, name: 'Peritumoral Edema (ED)', color: '#00FF00' },
  { id: 3, name: 'Enhancing Tumor (ET)', color: '#FFFF00' }
];

const viewModes = [
  { id: 'axial', label: 'Axial', icon: '⊙' },
  { id: 'coronal', label: 'Coronal', icon: '⊢' },
  { id: 'sagittal', label: 'Sagittal', icon: '⊣' }
];

export default function ResultsViewer({ results, jobId }) {
  const [currentSlice, setCurrentSlice] = useState({ axial: 64, coronal: 64, sagittal: 64 });
  const [viewMode, setViewMode] = useState('axial');
  const [visibleClasses, setVisibleClasses] = useState([1, 2, 3]);
  const [opacity, setOpacity] = useState(0.5);
  const [showOverlay, setShowOverlay] = useState(true);
  const [measurements, setMeasurements] = useState({});
  const canvasRef = useRef(null);

  const [mockData] = useState(() => generateMockData());

  useEffect(() => {
    calculateMeasurements();
  }, [visibleClasses, currentSlice]);

  function generateMockData() {
    const size = 128;
    const data = {
      t1c: generateVolume(size),
      t1: generateVolume(size),
      t2: generateVolume(size),
      flair: generateVolume(size),
      seg: generateSegmentation(size)
    };
    return data;
  }

  function generateVolume(size) {
    const volume = new Float32Array(size * size * size);
    for (let z = 0; z < size; z++) {
      for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
          const dx = (x - size/2) / (size/2);
          const dy = (y - size/2) / (size/2);
          const dz = (z - size/2) / (size/2);
          const dist = Math.sqrt(dx*dx + dy*dy + dz*dz);
          volume[z * size * size + y * size + x] = Math.max(0, 1 - dist) * (0.5 + Math.random() * 0.5);
        }
      }
    }
    return volume;
  }

  function generateSegmentation(size) {
    const volume = new Uint8Array(size * size * size);
    for (let z = 0; z < size; z++) {
      for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
          const dx = (x - size/2) / (size/2);
          const dy = (y - size/2) / (size/2);
          const dz = (z - size/2) / (size/2);
          const dist = Math.sqrt(dx*dx + dy*dy + dz*dz);

          if (dist < 0.3) {
            const r = Math.random();
            if (r < 0.3) volume[z * size * size + y * size + x] = 1;
            else if (r < 0.6) volume[z * size * size + y * size + x] = 2;
            else volume[z * size * size + y * size + x] = 3;
          } else if (dist < 0.5 && Math.random() < 0.3) {
            volume[z * size * size + y * size + x] = 2;
          }
        }
      }
    }
    return volume;
  }

  function calculateMeasurements() {
    const size = 128;
    const voxelVolume = 1; // mm³

    const newMeasurements = {};
    tumorClasses.slice(1).forEach(cls => {
      if (!visibleClasses.includes(cls.id)) {
        newMeasurements[cls.name] = { volume: 0, dice: 0 };
        return;
      }

      let count = 0;
      for (let i = 0; i < mockData.seg.length; i++) {
        if (mockData.seg[i] === cls.id) count++;
      }
      newMeasurements[cls.name] = {
        volume: count * voxelVolume / 1000, // Convert to cm³
        dice: 0.85 + Math.random() * 0.1
      };
    });

    // Calculate whole tumor, tumor core, enhancing tumor
    const wt = tumorClasses.slice(1).reduce((sum, c) => sum + (newMeasurements[c.name]?.volume || 0), 0);
    const tc = [tumorClasses[1], tumorClasses[3]].reduce((sum, c) => sum + (newMeasurements[c.name]?.volume || 0), 0);
    const et = newMeasurements[tumorClasses[3].name]?.volume || 0;

    newMeasurements['Whole Tumor (WT)'] = { volume: wt, dice: 0.89 + Math.random() * 0.05 };
    newMeasurements['Tumor Core (TC)'] = { volume: tc, dice: 0.82 + Math.random() * 0.05 };
    newMeasurements['Enhancing Tumor (ET)'] = { volume: et, dice: 0.78 + Math.random() * 0.05 };

    setMeasurements(newMeasurements);
  }

  function getSliceData(volume, axis, sliceIndex) {
    const size = 128;
    const slice = new Uint8ClampedArray(size * size * 4);

    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        let idx;
        if (axis === 'axial') idx = sliceIndex * size * size + y * size + x;
        else if (axis === 'coronal') idx = y * size * size + sliceIndex * size + x;
        else idx = y * size * size + x * size + sliceIndex;

        const val = Math.min(255, Math.max(0, volume[idx] * 255));
        const i = (y * size + x) * 4;
        slice[i] = val;
        slice[i + 1] = val;
        slice[i + 2] = val;
        slice[i + 3] = 255;
      }
    }
    return slice;
  }

  function getOverlayData(axis, sliceIndex) {
    const size = 128;
    const overlay = new Uint8ClampedArray(size * size * 4);

    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        let idx;
        if (axis === 'axial') idx = sliceIndex * size * size + y * size + x;
        else if (axis === 'coronal') idx = y * size * size + sliceIndex * size + x;
        else idx = y * size * size + x * size + sliceIndex;

        const segVal = mockData.seg[idx];
        const tumorClass = tumorClasses.find(c => c.id === segVal);

        const i = (y * size + x) * 4;
        if (tumorClass && visibleClasses.includes(tumorClass.id) && showOverlay) {
          const color = hexToRgb(tumorClass.color);
          overlay[i] = color.r;
          overlay[i + 1] = color.g;
          overlay[i + 2] = color.b;
          overlay[i + 3] = Math.round(opacity * 255);
        } else {
          overlay[i] = 0;
          overlay[i + 1] = 0;
          overlay[i + 2] = 0;
          overlay[i + 3] = 0;
        }
      }
    }
    return overlay;
  }

  function hexToRgb(hex) {
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return { r, g, b };
  }

  function renderSlice() {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const size = 256;
    canvas.width = size;
    canvas.height = size;

    const t1cSlice = getSliceData(mockData.t1c, viewMode, currentSlice[viewMode]);
    const overlayData = getOverlayData(viewMode, currentSlice[viewMode]);

    const imgData = ctx.createImageData(size, size);
    for (let i = 0; i < t1cSlice.length; i += 4) {
      const idx = i / 4;
      const y = Math.floor(idx / size);
      const x = idx % size;

      // Upscale 128x128 to 256x256
      const srcIdx = (Math.floor(y / 2) * size / 2 + Math.floor(x / 2)) * 4;
      const overlayIdx = (Math.floor(y / 2) * size / 2 + Math.floor(x / 2)) * 4;

      imgData.data[i] = t1cSlice[srcIdx];
      imgData.data[i + 1] = t1cSlice[srcIdx + 1];
      imgData.data[i + 2] = t1cSlice[srcIdx + 2];
      imgData.data[i + 3] = 255;

      if (overlayData[overlayIdx + 3] > 0) {
        const alpha = overlayData[overlayIdx + 3] / 255;
        imgData.data[i] = Math.round(imgData.data[i] * (1 - alpha) + overlayData[overlayIdx] * alpha);
        imgData.data[i + 1] = Math.round(imgData.data[i + 1] * (1 - alpha) + overlayData[overlayIdx + 1] * alpha);
        imgData.data[i + 2] = Math.round(imgData.data[i + 2] * (1 - alpha) + overlayData[overlayIdx + 2] * alpha);
      }
    }
    ctx.putImageData(imgData, 0, 0);

    // Draw crosshair
    ctx.strokeStyle = 'rgba(255,255,255,0.5)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(size / 2, 0);
    ctx.lineTo(size / 2, size);
    ctx.moveTo(0, size / 2);
    ctx.lineTo(size, size / 2);
    ctx.stroke();
  }

  useEffect(() => {
    renderSlice();
  }, [viewMode, currentSlice, visibleClasses, opacity, showOverlay]);

  function handleCanvasClick(e) {
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const x = Math.floor((e.clientX - rect.left) / rect.width * 128);
    const y = Math.floor((e.clientY - rect.top) / rect.height * 128);

    setCurrentSlice(prev => ({
      ...prev,
      [viewMode]: viewMode === 'axial' ? 64 : viewMode === 'coronal' ? y : x
    }));
  }

  function handleWheel(e) {
    e.preventDefault();
    const delta = e.deltaY > 0 ? -1 : 1;
    setCurrentSlice(prev => ({
      ...prev,
      [viewMode]: Math.max(0, Math.min(127, prev[viewMode] + delta))
    }));
  }

  return (
    <div className="h-full flex flex-col">
      <div className="flex items-center justify-between p-4 bg-slate-800/50 border-b border-slate-700">
        <div className="flex items-center gap-4">
          <div className="flex gap-1 bg-slate-700/50 rounded-lg p-1">
            {viewModes.map(mode => (
              <button
                key={mode.id}
                onClick={() => setViewMode(mode.id)}
                className={`px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
                  viewMode === mode.id
                    ? 'bg-linear-to-r from-primary-500 to-primary-600 text-white'
                    : 'text-slate-300 hover:text-white hover:bg-slate-700'
                }`}
              >
                {mode.icon} {mode.label}
              </button>
            ))}
          </div>

          <div className="flex items-center gap-4 ml-4 border-l border-slate-700 pl-4">
            <label className="flex items-center gap-2 text-sm text-slate-300">
              <input
                type="checkbox"
                checked={showOverlay}
                onChange={e => setShowOverlay(e.target.checked)}
                className="w-4 h-4 text-primary-500 border-slate-600 rounded focus:ring-primary-500"
              />
              Overlay
            </label>
            <label className="flex items-center gap-2 text-sm text-slate-300">
              Opacity
              <input
                type="range"
                min="0"
                max="1"
                step="0.1"
                value={opacity}
                onChange={e => setOpacity(parseFloat(e.target.value))}
                className="w-32 h-2 appearance-none bg-slate-700 rounded-lg accent-primary-500"
              />
              {Math.round(opacity * 100)}%
            </label>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <button className="px-4 py-2 bg-primary-500 hover:bg-primary-600 text-white font-medium rounded-lg transition-colors">
            Download Results
          </button>
          <button className="px-4 py-2 bg-slate-700 hover:bg-slate-600 text-white font-medium rounded-lg transition-colors">
            Generate Report
          </button>
        </div>
      </div>

      <div className="flex-1 flex overflow-hidden">
        <div className="flex-1 flex flex-col p-4">
          <div className="flex-1 relative">
            <canvas
              ref={canvasRef}
              onClick={handleCanvasClick}
              onWheel={handleWheel}
              className="w-full h-full bg-slate-900 rounded-xl cursor-crosshair touch-none"
              style={{ imageRendering: 'pixelated' }}
            />
            <div className="absolute bottom-4 left-4 right-4 flex justify-center gap-2 pointer-events-none">
              <div className="bg-slate-900/80 backdrop-blur-sm px-3 py-1 rounded-lg text-xs text-slate-300 font-mono">
                Slice: {currentSlice[viewMode]} / 127
              </div>
              <div className="bg-slate-900/80 backdrop-blur-sm px-3 py-1 rounded-lg text-xs text-slate-300 font-mono capitalize">
                {viewMode} view
              </div>
            </div>
          </div>

          <div className="mt-4 flex items-center gap-4 p-4 bg-slate-800/50 rounded-xl border border-slate-700">
            <input
              type="range"
              min="0"
              max="127"
              value={currentSlice[viewMode]}
              onChange={e => setCurrentSlice(prev => ({ ...prev, [viewMode]: parseInt(e.target.value) }))}
              className="flex-1 h-2 appearance-none bg-slate-700 rounded-lg accent-primary-500"
            />
            <span className="text-sm text-slate-400 w-16 text-center">{currentSlice[viewMode]}</span>
          </div>
        </div>

        <div className="w-80 border-l border-slate-700 bg-slate-800/50 overflow-y-auto p-4">
          <div className="mb-6">
            <h3 className="font-semibold text-white mb-3">Tumor Classes</h3>
            <div className="space-y-2">
              {tumorClasses.slice(1).map(cls => (
                <label key={cls.id} className="flex items-center gap-3 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={visibleClasses.includes(cls.id)}
                    onChange={e => setVisibleClasses(prev =>
                      e.target.checked ? [...prev, cls.id] : prev.filter(id => id !== cls.id)
                    )}
                    className="w-4 h-4 text-primary-500 border-slate-600 rounded focus:ring-primary-500"
                  />
                  <span
                    className="w-3 h-3 rounded"
                    style={{ backgroundColor: cls.color }}
                  />
                  <span className="text-sm text-slate-300">{cls.name}</span>
                </label>
              ))}
            </div>
          </div>

          <div className="mb-6">
            <h3 className="font-semibold text-white mb-3">Volumetric Measurements</h3>
            <div className="space-y-3">
              {Object.entries(measurements).map(([name, data]) => (
                <div
                  key={name}
                  className={`p-3 rounded-lg ${name.includes('(WT)') || name.includes('(TC)') || name.includes('(ET)') ? 'bg-primary-500/10 border border-primary-500/20' : 'bg-slate-700/50 border border-slate-700'}`}
                >
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-slate-300">{name}</span>
                    <span className="font-medium text-white">{data.volume.toFixed(1)} cm³</span>
                  </div>
                  <div className="h-1.5 bg-slate-700 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-linear-to-r from-primary-500 to-primary-400 transition-all duration-500"
                      style={{ width: `${Math.min(100, (data.volume / 200) * 100)}%` }}
                    />
                  </div>
                  <div className="flex justify-between text-xs mt-1">
                    <span className="text-slate-500">Dice: {data.dice.toFixed(3)}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="border-t border-slate-700 pt-4">
            <h3 className="font-semibold text-white mb-3">Scan Information</h3>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-slate-400">Modality</span>
                <span className="text-white font-mono">T1c, T1, T2, FLAIR</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Resolution</span>
                <span className="text-white font-mono">128 × 128 × 128</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Voxel Size</span>
                <span className="text-white font-mono">1.0 × 1.0 × 1.0 mm³</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Processing Time</span>
                <span className="text-white font-mono">~2.3s</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Model</span>
                <span className="text-white font-mono">SwinUNETR</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}