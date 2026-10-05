import { motion, useReducedMotion } from 'framer-motion';

/**
 * Viewport-triggered reveal wrapper. Children of a `stagger` container should
 * be wrapped individually in <RevealOnScroll> as well so each gets its own
 * `variants` entry — framer staggers by walking motion-component children.
 *
 * Respects prefers-reduced-motion: renders children at final state with no
 * transform/delay when the user has motion reduced at the OS level.
 */
export function RevealOnScroll({
  children,
  as = 'div',
  className = '',
  y = 24,
  x = 0,
  scale = 1,
  delay = 0,
  duration = 0.5,
  once = true,
  amount = 0.3,
  stagger = false,
  staggerDelay = 0.08,
}) {
  const prefersReducedMotion = useReducedMotion();
  const Component = motion[as] || motion.div;

  if (prefersReducedMotion) {
    const Plain = as;
    return <Plain className={className}>{children}</Plain>;
  }

  if (stagger) {
    return (
      <Component
        className={className}
        initial="hidden"
        whileInView="visible"
        viewport={{ once, amount }}
        variants={{
          hidden: {},
          visible: {
            transition: { staggerChildren: staggerDelay, delayChildren: delay },
          },
        }}
      >
        {children}
      </Component>
    );
  }

  return (
    <Component
      className={className}
      initial={{ opacity: 0, y, x, scale: scale !== 1 ? scale : undefined }}
      whileInView={{ opacity: 1, y: 0, x: 0, scale: 1 }}
      viewport={{ once, amount }}
      transition={{ duration, delay, ease: [0.22, 1, 0.36, 1] }}
    >
      {children}
    </Component>
  );
}

/** Stagger-item variant for children of a `stagger` RevealOnScroll container. */
export function RevealItem({ children, as = 'div', className = '', y = 24, x = 0 }) {
  const prefersReducedMotion = useReducedMotion();
  const Component = motion[as] || motion.div;

  if (prefersReducedMotion) {
    const Plain = as;
    return <Plain className={className}>{children}</Plain>;
  }

  return (
    <Component
      className={className}
      variants={{
        hidden: { opacity: 0, y, x },
        visible: { opacity: 1, y: 0, x: 0, transition: { duration: 0.5, ease: [0.22, 1, 0.36, 1] } },
      }}
    >
      {children}
    </Component>
  );
}

export default RevealOnScroll;
