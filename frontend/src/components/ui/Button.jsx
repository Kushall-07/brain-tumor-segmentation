import { cn } from '../../utils/cn';

export default function Button({
  children,
  variant = 'primary',
  size = 'md',
  icon: Icon,
  iconPosition = 'left',
  className = '',
  onClick,
  disabled = false,
  type = 'button',
}) {
  const variants = {
    primary: 'bg-arterial hover:bg-arterial-light text-parchment border border-arterial',
    secondary: 'bg-annotation hover:bg-annotation/90 text-parchment border border-annotation',
    outline: 'bg-parchment hover:bg-parchment-dark text-ink border border-sepia-border hover:border-sepia-muted',
  };

  const sizes = {
    sm: 'px-4 py-2 text-sm',
    md: 'px-6 py-3 text-base',
    lg: 'px-8 py-4 text-lg',
  };

  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled}
      className={cn(
        'inline-flex items-center justify-center gap-2 rounded-sm font-medium transition-colors duration-200',
        'disabled:opacity-50 disabled:cursor-not-allowed',
        variants[variant],
        sizes[size],
        className
      )}
    >
      {Icon && iconPosition === 'left' && <Icon size={20} strokeWidth={1.5} />}
      {children}
      {Icon && iconPosition === 'right' && <Icon size={20} strokeWidth={1.5} />}
    </button>
  );
}
