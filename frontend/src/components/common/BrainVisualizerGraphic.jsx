import { useEffect, useRef, useState } from 'react';
import { motion, useReducedMotion, useScroll, useTransform } from 'framer-motion';

const CORNER_POSITIONS = [
  'top-2 left-2 border-t border-l',
  'top-2 right-2 border-t border-r',
  'bottom-2 left-2 border-b border-l',
  'bottom-2 right-2 border-b border-r',
];

export default function BrainVisualizerGraphic() {
  const containerRef = useRef(null);
  const prefersReducedMotion = useReducedMotion();
  const [isNarrowViewport, setIsNarrowViewport] = useState(false);

  useEffect(() => {
    const query = window.matchMedia('(max-width: 1023px)');
    setIsNarrowViewport(query.matches);
    const handleChange = (event) => setIsNarrowViewport(event.matches);
    query.addEventListener('change', handleChange);
    return () => query.removeEventListener('change', handleChange);
  }, []);

  const { scrollYProgress } = useScroll({
    target: containerRef,
    offset: ['start start', 'end start'],
  });

  const sweepEnabled = !prefersReducedMotion && !isNarrowViewport;

  const sweepY = useTransform(scrollYProgress, [0, 1], ['0%', '100%']);
  const sweepOpacity = useTransform(
    scrollYProgress,
    [0, 0.08, 0.85, 1],
    [0, 1, 1, 0]
  );

  return (
    <div ref={containerRef} className="relative w-full max-w-md mx-auto">
      <div className="relative border border-sepia-border rounded-sm p-4 bg-parchment-dark overflow-hidden">
        {CORNER_POSITIONS.map((position) => (
          <motion.span
            key={position}
            className={`absolute w-3 h-3 border-brass ${position}`}
            initial={prefersReducedMotion ? false : { scale: 0, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ duration: 0.4, delay: 0.2, ease: [0.22, 1, 0.36, 1] }}
          />
        ))}

        <div className="relative flex items-center justify-center">
          <img
            src="/brain-tumor-ai-fig1.png"
            alt="Brain tumor segmentation anatomical reference"
            className="w-full h-auto object-contain"
          />

          {sweepEnabled && (
            <motion.div
              aria-hidden="true"
              className="absolute left-0 right-0 h-px pointer-events-none"
              style={{
                top: sweepY,
                opacity: sweepOpacity,
                background:
                  'linear-gradient(90deg, transparent, var(--color-arterial) 20%, var(--color-annotation) 80%, transparent)',
                boxShadow: 'none',
              }}
            />
          )}
        </div>
      </div>
    </div>
  );
}
