import { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Layers,
  BrainCircuit,
  Zap,
  Stethoscope,
  ArrowRight,
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

  const iconMap = {
    Layers: Layers,
    BrainCircuit: BrainCircuit,
    Zap: Zap,
    Stethoscope: Stethoscope,
  };

  return (
    <div className="relative overflow-hidden pt-28 pb-20">
      {/* ========================================================================= */}
      {/* 1. HERO SECTION */}
      {/* ========================================================================= */}
      <section className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-6 pb-20">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-12 items-center">

          <div className="lg:col-span-7 space-y-8 text-center lg:text-left">

            <div className="inline-flex items-center gap-2 px-3 py-1.5 border border-sepia-border bg-parchment-dark text-ink-caption text-xs font-medium uppercase tracking-[0.12em]">
              <span className="w-1.5 h-1.5 bg-brass rounded-full" />
              <span>Clinical Decision Support</span>
            </div>

            <h1 className="font-serif text-4xl sm:text-5xl lg:text-6xl font-semibold text-ink tracking-tight leading-[1.15]">
              AI-Assisted{' '}
              <span className="text-arterial">Brain Tumor Segmentation</span>
              {' '}& Clinical Analysis
            </h1>

            <p className="text-lg sm:text-xl text-ink-body max-w-2xl mx-auto lg:mx-0 leading-relaxed">
              An advanced deep learning framework utilizing 3D SwinUNETR Vision Transformers to perform automated voxel-level segmentation of complex glioma structures across multi-modal MRI scans (T1c, T1n, T2f, T2w).
            </p>

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
                className="w-full sm:w-auto inline-flex items-center justify-center gap-2 px-6 py-3.5 rounded-sm bg-parchment hover:bg-parchment-dark text-ink border border-sepia-border hover:border-sepia-muted transition-colors text-base font-medium"
              >
                <span>Learn How It Works</span>
              </a>
            </div>

            <div className="pt-6 grid grid-cols-3 gap-4 border-t border-sepia-border max-w-xl mx-auto lg:mx-0">
              <div className="text-center lg:text-left">
                <div className="text-2xl font-mono font-medium text-ink">4</div>
                <div className="atlas-label mt-0.5">Modalities</div>
              </div>
              <div className="text-center lg:text-left">
                <div className="text-2xl font-mono font-medium text-ink">3</div>
                <div className="atlas-label mt-0.5">Sub-Regions</div>
              </div>
              <div className="text-center lg:text-left">
                <div className="text-2xl font-mono font-medium text-arterial">&lt;3s</div>
                <div className="atlas-label mt-0.5">GPU Inference</div>
              </div>
            </div>

          </div>

          <div className="lg:col-span-5 flex justify-center">
            <BrainVisualizerGraphic />
          </div>

        </div>
      </section>


      {/* ========================================================================= */}
      {/* 2. FEATURES SECTION */}
      {/* ========================================================================= */}
      <section className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16 border-t border-sepia-border">
        <div className="text-center space-y-4 mb-16">
          <Badge variant="brass" icon={Cpu}>Core System Capabilities</Badge>
          <h2 className="font-serif text-3xl sm:text-4xl font-semibold text-ink tracking-tight">
            Designed for Precision Medical AI
          </h2>
          <p className="text-ink-body max-w-2xl mx-auto text-base">
            Combining multi-sequence MRI image fusion, 3D transformer neural architectures, and automated volumetric clinical metrics.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {FEATURE_CARDS.map((card) => {
            const IconComponent = iconMap[card.iconName] || BrainCircuit;
            return (
              <Card key={card.id} hoverEffect={true} className="flex flex-col justify-between group">
                <div className="space-y-4">
                  <div className="w-11 h-11 rounded-sm border border-sepia-border bg-parchment-dark flex items-center justify-center text-annotation group-hover:border-brass transition-colors">
                    <IconComponent className="w-5 h-5" strokeWidth={1.5} />
                  </div>
                  <Badge variant="default">{card.badge}</Badge>
                  <h3 className="font-serif text-xl font-semibold text-ink group-hover:text-arterial transition-colors">
                    {card.title}
                  </h3>
                  <p className="text-sm text-ink-body leading-relaxed">
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
      <section id="about" className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20 border-t border-sepia-border scroll-mt-24">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-12 items-center">

          <div className="lg:col-span-6 space-y-6">
            <Badge variant="brass" icon={Microscope}>System Workflow</Badge>
            <h2 className="font-serif text-3xl sm:text-4xl font-semibold text-ink tracking-tight">
              How the Segmentation Pipeline Works
            </h2>
            <p className="text-ink-body leading-relaxed text-base">
              The platform integrates a end-to-end 3D computer vision pipeline engineered specifically for high-resolution brain MRI analysis. By processing four distinct magnetic resonance sequences simultaneously, the model captures complementary tissue contrast characteristics.
            </p>

            <div className="space-y-3 pt-2">
              {SYSTEM_STEPS.map((stepItem) => (
                <div
                  key={stepItem.step}
                  className="flex gap-4 p-4 rounded-sm bg-parchment-dark border border-sepia-border hover:border-sepia-muted transition-colors"
                >
                  <div className="shrink-0 w-10 h-10 rounded-sm border border-brass/50 bg-parchment flex items-center justify-center font-mono font-medium text-brass text-sm">
                    {stepItem.step}
                  </div>
                  <div className="space-y-1">
                    <h4 className="text-base font-medium text-ink">
                      {stepItem.title}
                    </h4>
                    <p className="text-xs text-ink-body leading-normal">
                      {stepItem.description}
                    </p>
                  </div>
                </div>
              ))}
            </div>

          </div>

          <div className="lg:col-span-6 space-y-6">

            <Card className="space-y-5">
              <div className="flex items-center justify-between border-b border-sepia-border pb-3">
                <h3 className="font-serif text-lg font-semibold text-ink flex items-center gap-2">
                  <Activity className="w-5 h-5 text-arterial" strokeWidth={1.5} />
                  <span>Tumor Sub-Region Classification</span>
                </h3>
                <span className="text-xs text-ink-label font-mono">BraTS</span>
              </div>

              <div className="space-y-3">
                {TUMOR_CLASSES.map((cls) => (
                  <div key={cls.id} className="p-3 rounded-sm bg-parchment-dark border border-sepia-border flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <span className={`w-3 h-3 rounded-full ${cls.color}`} />
                      <span className="text-sm font-medium text-ink">{cls.name}</span>
                    </div>
                    <span className="text-xs font-mono px-2 py-0.5 rounded-sm bg-parchment text-ink-label border border-sepia-border">
                      Voxel Class
                    </span>
                  </div>
                ))}
              </div>
            </Card>

            <Card className="space-y-4">
              <h3 className="text-base font-serif font-semibold text-ink flex items-center gap-2">
                <FileSpreadsheet className="w-4 h-4 text-annotation" strokeWidth={1.5} />
                <span>Multi-Modal MRI Input Channels</span>
              </h3>
              <div className="grid grid-cols-2 gap-3">
                {MRI_MODALITIES.map((mod) => (
                  <div key={mod.id} className="p-3 rounded-sm bg-parchment-dark border border-sepia-border space-y-1">
                    <span className="text-xs font-mono font-medium text-annotation block">{mod.name}</span>
                    <p className="text-[11px] text-ink-body line-clamp-2">{mod.description}</p>
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
        <div className="relative rounded-sm bg-ink-deep border border-sepia-border p-8 sm:p-12 text-center overflow-hidden">
          <div className="relative z-10 max-w-3xl mx-auto space-y-6">
            <h2 className="font-serif text-3xl sm:text-4xl font-semibold text-parchment tracking-tight">
              Ready to Run AI Brain Tumor Segmentation?
            </h2>
            <p className="text-parchment/70 text-base leading-relaxed">
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
