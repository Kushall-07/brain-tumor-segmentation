import { cn } from '../../utils/cn';

export default function Badge({
  children,
  variant = 'default',
  icon: Icon,
  className = '',
}) {
  const variants = {
    default: 'bg-parchment-dark text-ink-label border border-sepia-border',
    cyan: 'bg-parchment-dark text-annotation border border-sepia-border',
    blue: 'bg-parchment-dark text-annotation border border-sepia-border',
    green: 'bg-parchment-dark text-annotation border border-sepia-border',
    red: 'bg-parchment-dark text-arterial border border-sepia-border',
    brass: 'bg-parchment-dark text-brass border border-brass/40',
  };

  return (
    <span
      className={cn(
        'inline-flex items-center gap-1.5 px-3 py-1 rounded-sm text-xs font-medium uppercase tracking-[0.1em]',
        variants[variant],
        className
      )}
    >
      {Icon && <Icon size={14} strokeWidth={1.5} />}
      {children}
    </span>
  );
}
