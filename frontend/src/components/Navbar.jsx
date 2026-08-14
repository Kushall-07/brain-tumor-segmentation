import { NavLink } from 'react-router-dom';

function Navbar() {
  return (
    <nav className="fixed top-0 left-0 right-0 z-50 bg-parchment border-b border-sepia-border">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <NavLink to="/" className="flex items-center gap-3 group">
            <img
              src="/brain-tumor-ai-fig1.png"
              alt="Brain tumor logo"
              className="w-16 h-16 object-contain opacity-80"
            />
            <div className="leading-tight">
              <span className="block font-serif text-sm font-semibold text-ink tracking-wide uppercase">
                Brain Tumor
              </span>
              <span className="block font-serif text-xs text-ink-caption tracking-[0.08em] uppercase">
                Segmentation AI
              </span>
            </div>
          </NavLink>

          <div className="hidden md:flex items-center space-x-1">
            <NavLink
              to="/"
              className={({ isActive }) =>
                `px-4 py-2 rounded-sm text-sm font-medium tracking-wide transition-colors ${
                  isActive
                    ? 'bg-arterial/10 text-arterial border border-arterial/20'
                    : 'text-ink-nav hover:text-ink hover:bg-parchment-dark'
                }`
              }
            >
              Home
            </NavLink>
            <NavLink
              to="/predict"
              className={({ isActive }) =>
                `px-4 py-2 rounded-sm text-sm font-medium tracking-wide transition-colors ${
                  isActive
                    ? 'bg-arterial/10 text-arterial border border-arterial/20'
                    : 'text-ink-nav hover:text-ink hover:bg-parchment-dark'
                }`
              }
            >
              Predict
            </NavLink>
            <NavLink
              to="/results"
              className={({ isActive }) =>
                `px-4 py-2 rounded-sm text-sm font-medium tracking-wide transition-colors ${
                  isActive
                    ? 'bg-arterial/10 text-arterial border border-arterial/20'
                    : 'text-ink-nav hover:text-ink hover:bg-parchment-dark'
                }`
              }
            >
              Results
            </NavLink>
            <NavLink
              to="/model-evaluation"
              className={({ isActive }) =>
                `px-4 py-2 rounded-sm text-sm font-medium tracking-wide transition-colors ${
                  isActive
                    ? 'bg-arterial/10 text-arterial border border-arterial/20'
                    : 'text-ink-nav hover:text-ink hover:bg-parchment-dark'
                }`
              }
            >
              Model &amp; Evaluation
            </NavLink>
          </div>

          <div className="md:hidden">
            <button type="button" className="text-ink-nav hover:text-ink p-2 border border-sepia-border rounded-sm">
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 6h16M4 12h16M4 18h16" />
              </svg>
            </button>
          </div>
        </div>
      </div>
    </nav>
  );
}

export default Navbar;
