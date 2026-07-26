export const PROJECT_TITLE = 'Brain Tumor Segmentation AI';

export const FEATURE_CARDS = [
  {
    id: 1,
    iconName: 'Layers',
    badge: 'Multi-Modal Fusion',
    title: '4-Channel MRI Input',
    description: 'Processes T1c, T1n, T2f, and T2w sequences simultaneously to capture complementary tissue contrast.',
  },
  {
    id: 2,
    iconName: 'BrainCircuit',
    badge: '3D Architecture',
    title: 'SwinUNETR Transformer',
    description: 'State-of-the-art 3D vision transformer architecture for volumetric medical image segmentation.',
  },
  {
    id: 3,
    iconName: 'Zap',
    badge: 'GPU Accelerated',
    title: 'Fast Inference',
    description: 'Optimized CUDA implementation delivers sub-3-second prediction times on modern GPUs.',
  },
  {
    id: 4,
    iconName: 'Stethoscope',
    badge: 'Clinical Metrics',
    title: 'Automated Analysis',
    description: 'Computes volumetric tumor sub-region statistics for enhanced clinical decision support.',
  },
];

export const SYSTEM_STEPS = [
  {
    step: '01',
    title: 'MRI Upload',
    description: 'Upload four co-registered NIfTI files (T1c, T1n, T2f, T2w) through the secure web interface.',
  },
  {
    step: '02',
    title: 'Preprocessing',
    description: 'Automatic z-score normalization, skull stripping, and 3D patch extraction for model input.',
  },
  {
    step: '03',
    title: 'Inference',
    description: '3D SwinUNETR model processes volumetric data to generate voxel-wise tumor segmentation.',
  },
  {
    step: '04',
    title: 'Post-processing',
    description: 'Connected component analysis and morphological operations refine the final segmentation mask.',
  },
];

export const TUMOR_CLASSES = [
  {
    id: 1,
    name: 'Enhancing Tumor (ET)',
    color: 'from-red-500 to-pink-500',
  },
  {
    id: 2,
    name: 'Tumor Core (TC)',
    color: 'from-yellow-500 to-orange-500',
  },
  {
    id: 3,
    name: 'Whole Tumor (WT)',
    color: 'from-green-500 to-emerald-500',
  },
];

export const MRI_MODALITIES = [
  {
    id: 1,
    name: 'T1c',
    description: 'Contrast-enhanced T1-weighted MRI highlights tumor vasculature and blood-brain barrier breakdown.',
  },
  {
    id: 2,
    name: 'T1n',
    description: 'Native T1-weighted MRI provides anatomical reference for brain structure localization.',
  },
  {
    id: 3,
    name: 'T2f',
    description: 'T2-FLAIR suppresses CSF signal to better visualize peritumoral edema and infiltrative tumor.',
  },
  {
    id: 4,
    name: 'T2w',
    description: 'T2-weighted MRI reveals tumor heterogeneity and necrotic regions with high contrast.',
  },
];
