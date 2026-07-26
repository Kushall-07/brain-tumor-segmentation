import { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Layers,
  BrainCircuit,
  Zap,
  Stethoscope,
  ArrowRight,
  Sparkles,
  ShieldCheck,
  CheckCircle2,
  Cpu,
  FileSpreadsheet,
  Activity,
  Microscope
} from 'lucide-react';
import Card from '../components/ui/Card';
import Button from '../components/ui/Button';
import Badge from '../components/ui/Badge';
import BrainVisualizerGraphic from '../components/common/BrainVisualizerGraphic';
import {
  PROJECT_TITLE,
  FEATURE_CARDS,
  SYSTEM_STEPS,
  TUMOR_CLASSES,
  MRI_MODALITIES
} from '../utils/constants';

export function Home() {
  const navigate = useNavigate();

  useEffect(() => {
    if (window.location.hash === '#about') {
      document.getElementById('about')?.scrollIntoView({ behavior: 'smooth' });
    }
  }, []);

  // Helper map for icons in Feature Cards
  const iconMap = {
    Layers: Layers,
    BrainCircuit: BrainCircuit,
    Zap: Zap,
    Stethoscope: Stethoscope,
  };

  return (
    <div className="relative overflow-hidden pt-28 pb-20">
      {/* Background Decorative Ambient Glows */}
      <div className="absolute top-1/4 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-cyan-500/10 rounded-full blur-[140px] pointer-events-none" />
      <div className="absolute top-3/4 right-0 w-[400px] h-[400px] bg-blue-600/10 rounded-full blur-[120px] pointer-events-none" />

      {/* ========================================================================= */}
      {/* 1. HERO SECTION */}
      {/* ========================================================================= */}
      <section className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-6 pb-20">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-12 items-center">
          
          {/* Left Hero Content */}
          <div className="lg:col-span-7 space-y-8 text-center lg:text-left">
            
            {/* Top Academic Tag */}
            <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-cyan-500/10 border border-cyan-500/30 text-cyan-300 text-xs font-semibold uppercase tracking-wider backdrop-blur-md shadow-xs">
              <Sparkles className="w-3.5 h-3.5 text-cyan-400 animate-pulse" />
              <span>Visualize The Tumor</span>
            </div>

            {/* Project Title */}
            <h1 className="text-4xl sm:text-5xl lg:text-6xl font-extrabold text-white tracking-tight leading-[1.15]">
              AI-Assisted{' '}
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 via-sky-300 to-blue-500">
                Brain Tumor Segmentation
              </span>{' '}
              & Clinical Decision Support
            </h1>

            {/* Short Project Description */}
            <p className="text-lg sm:text-xl text-slate-300 max-w-2xl mx-auto lg:mx-0 leading-relaxed font-normal">
              An advanced deep learning framework utilizing 3D SwinUNETR Vision Transformers to perform automated voxel-level segmentation of complex glioma structures across multi-modal MRI scans (T1c, T1n, T2f, T2w).
            </p>

            {/* CTAs & Badges */}
            <div className="pt-2 flex flex-col sm:flex-row items-center justify-center lg:justify-start gap-4">
              <Button
                variant="primary"
                size="lg"
                onClick={() => navigate('/predict')}
                icon={ArrowRight}
                iconPosition="right"
                className="w-full sm:w-auto"
              >
                Start Prediction
              </Button>

              <a
                href="#about"
                className="w-full sm:w-auto inline-flex items-center justify-center gap-2 px-6 py-3.5 rounded-xl bg-slate-900/80 hover:bg-slate-800 text-slate-300 hover:text-white border border-slate-700/80 transition-all duration-300 text-base font-medium backdrop-blur-md"
              >
                <span>Learn How It Works</span>
              </a>
            </div>

            {/* System Highlights Pills */}
            <div className="pt-6 grid grid-cols-3 gap-4 border-t border-slate-800/80 max-w-xl mx-auto lg:mx-0">
              <div className="text-center lg:text-left">
                <div className="text-2xl font-bold text-cyan-400">4 Modalities</div>
                <div className="text-xs text-slate-400 mt-0.5">T1c, T1n, T2f, T2w</div>
              </div>
              <div className="text-center lg:text-left">
                <div className="text-2xl font-bold text-cyan-400">3 Sub-Regions</div>
                <div className="text-xs text-slate-400 mt-0.5">ET, TC, WT Classes</div>
              </div>
              <div className="text-center lg:text-left">
                <div className="text-2xl font-bold text-emerald-400">&lt; 3 Seconds</div>
                <div className="text-xs text-slate-400 mt-0.5">GPU 3D Inference</div>
              </div>
            </div>

          </div>

          {/* Right Hero Interactive Brain Graphic */}
          <div className="lg:col-span-5 flex justify-center">
            <BrainVisualizerGraphic />
          </div>

        </div>
      </section>


      {/* ========================================================================= */}
      {/* 2. FEATURES SECTION */}
      {/* ========================================================================= */}
      <section className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16 border-t border-slate-800/80">
        <div className="text-center space-y-4 mb-16">
          <Badge variant="cyan" icon={Cpu}>Core System Capabilities</Badge>
          <h2 className="text-3xl sm:text-4xl font-bold text-white tracking-tight">
            Designed for Precision Medical AI
          </h2>
          <p className="text-slate-400 max-w-2xl mx-auto text-base">
            Combining multi-sequence MRI image fusion, 3D transformer neural architectures, and automated volumetric clinical metrics.
          </p>
        </div>

        {/* 4 Feature Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {FEATURE_CARDS.map((card) => {
            const IconComponent = iconMap[card.iconName] || BrainCircuit;
            return (
              <Card key={card.id} hoverEffect={true} className="flex flex-col justify-between group">
                <div className="space-y-4">
                  <div className="w-12 h-12 rounded-xl bg-gradient-to-tr from-cyan-500/20 to-blue-500/20 border border-cyan-500/30 flex items-center justify-center text-cyan-400 group-hover:scale-110 group-hover:border-cyan-400 transition-all duration-300">
                    <IconComponent className="w-6 h-6" />
                  </div>
                  <Badge variant="blue">{card.badge}</Badge>
                  <h3 className="text-xl font-bold text-white group-hover:text-cyan-300 transition-colors">
                    {card.title}
                  </h3>
                  <p className="text-sm text-slate-400 leading-relaxed">
                    {card.description}
                  </p>
                </div>
              </Card>
            );
          })}
        </div>
      </section>


      {/* ========================================================================= */}
      {/* 3. ABOUT SECTION */}
      {/* ========================================================================= */}
      <section id="about" className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20 border-t border-slate-800/80 scroll-mt-24">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-12 items-center">
          
          {/* Left Column: System Architecture Explanation */}
          <div className="lg:col-span-6 space-y-6">
            <Badge variant="cyan" icon={Microscope}>System Workflow</Badge>
            <h2 className="text-3xl sm:text-4xl font-bold text-white tracking-tight">
              How the Segmentation Pipeline Works
            </h2>
            <p className="text-slate-300 leading-relaxed text-base">
              The platform integrates a end-to-end 3D computer vision pipeline engineered specifically for high-resolution brain MRI analysis. By processing four distinct magnetic resonance sequences simultaneously, the model captures complementary tissue contrast characteristics.
            </p>

            {/* Pipeline Step Cards */}
            <div className="space-y-4 pt-2">
              {SYSTEM_STEPS.map((stepItem) => (
                <div
                  key={stepItem.step}
                  className="flex gap-4 p-4 rounded-xl bg-slate-900/60 border border-slate-800/80 hover:border-slate-700 transition-all"
                >
                  <div className="shrink-0 w-10 h-10 rounded-lg bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center font-mono font-bold text-cyan-400 text-sm">
                    {stepItem.step}
                  </div>
                  <div className="space-y-1">
                    <h4 className="text-base font-semibold text-white">
                      {stepItem.title}
                    </h4>
                    <p className="text-xs text-slate-400 leading-normal">
                      {stepItem.description}
                    </p>
                  </div>
                </div>
              ))}
            </div>

          </div>

          {/* Right Column: Tumor Sub-region & Modality Technical Cards */}
          <div className="lg:col-span-6 space-y-6">
            
            {/* Tumor Sub-regions Breakdown Card */}
            <Card glow={true} className="space-y-5">
              <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                <h3 className="text-lg font-bold text-white flex items-center gap-2">
                  <Activity className="w-5 h-5 text-cyan-400" />
                  <span>Tumor Sub-Region Classification</span>
                </h3>
                <span className="text-xs text-slate-400 font-mono">BraTS Standard</span>
              </div>

              <div className="space-y-3">
                {TUMOR_CLASSES.map((cls) => (
                  <div key={cls.id} className="p-3 rounded-xl bg-slate-950/80 border border-slate-800/80 flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <span className={`w-3.5 h-3.5 rounded-full bg-gradient-to-r ${cls.color}`} />
                      <span className="text-sm font-semibold text-slate-200">{cls.name}</span>
                    </div>
                    <span className="text-xs font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-400 border border-slate-700">
                      Voxel Class
                    </span>
                  </div>
                ))}
              </div>
            </Card>

            {/* MRI Modality Grid */}
            <Card className="space-y-4">
              <h3 className="text-base font-bold text-white flex items-center gap-2">
                <FileSpreadsheet className="w-4 h-4 text-blue-400" />
                <span>Multi-Modal MRI Input Channels</span>
              </h3>
              <div className="grid grid-cols-2 gap-3">
                {MRI_MODALITIES.map((mod) => (
                  <div key={mod.id} className="p-3 rounded-xl bg-slate-950/60 border border-slate-800/80 space-y-1">
                    <span className="text-xs font-bold text-cyan-400 block">{mod.name}</span>
                    <p className="text-[11px] text-slate-400 line-clamp-2">{mod.description}</p>
                  </div>
                ))}
              </div>
            </Card>

          </div>

        </div>
      </section>


      {/* ========================================================================= */}
      {/* 4. PRE-FOOTER CTA BANNER */}
      {/* ========================================================================= */}
      <section className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-12">
        <div className="relative rounded-3xl bg-gradient-to-r from-slate-900 via-cyan-950/40 to-slate-900 border border-cyan-500/30 p-8 sm:p-12 text-center overflow-hidden shadow-2xl">
          <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,var(--color-cyan-500),transparent_70%)] opacity-10 pointer-events-none" />
          
          <div className="relative z-10 max-w-3xl mx-auto space-y-6">
            <h2 className="text-3xl sm:text-4xl font-extrabold text-white tracking-tight">
              Ready to Run AI Brain Tumor Segmentation?
            </h2>
            <p className="text-slate-300 text-base leading-relaxed">
              Navigate to the prediction suite to prepare multi-modal MRI scans for automated 3D neural network inference.
            </p>
            <Button
              variant="primary"
              size="lg"
              onClick={() => navigate('/predict')}
              icon={ArrowRight}
              iconPosition="right"
              className="mx-auto"
            >
              Launch Prediction Interface
            </Button>
          </div>
        </div>
      </section>

    </div>
  );
}

export default Home;