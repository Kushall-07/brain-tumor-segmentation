import { cn } from '../../utils/cn';

export default function Card({
  children,
  className = '',
  hoverEffect = false,
}) {
  return (
    <div
      className={cn(
        'rounded-sm bg-parchment border border-sepia-border p-6',
        hoverEffect && 'hover:border-sepia-muted hover:bg-parchment-dark/50 transition-colors duration-200',
        className
      )}
    >
      {children}
    </div>
  );
}
