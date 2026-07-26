import { cn } from '../../utils/cn';

export default function Badge({
  children,
  variant = 'default',
  icon: Icon,
  className = '',
}) {
  const variants = {
    default: 'bg-slate-800 text-slate-300 border border-slate-700',
    cyan: 'bg-cyan-500/10 text-cyan-300 border border-cyan-500/30',
    blue: 'bg-blue-500/10 text-blue-300 border border-blue-500/30',
    green: 'bg-emerald-500/10 text-emerald-300 border border-emerald-500/30',
    red: 'bg-red-500/10 text-red-300 border border-red-500/30',
  };

  return (
    <span
      className={cn(
        'inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold uppercase tracking-wider',
        variants[variant],
        className
      )}
    >
      {Icon && <Icon size={14} />}
      {children}
    </span>
  );
}
