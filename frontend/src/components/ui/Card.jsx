import { cn } from '../../utils/cn';

export default function Card({ 
  children, 
  className = '', 
  hoverEffect = false, 
  glow = false 
}) {
  return (
    <div
      className={cn(
        'rounded-xl bg-slate-900/60 border border-slate-800/80 p-6',
        hoverEffect && 'hover:border-cyan-500/50 hover:bg-slate-900/80 transition-all duration-300',
        glow && 'shadow-[0_0_30px_rgba(6,182,212,0.15)]',
        className
      )}
    >
      {children}
    </div>
  );
}
