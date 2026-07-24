import { NavLink } from 'react-router-dom';

function Navbar() {
  return (
    <nav className="fixed top-0 left-0 right-0 z-50 bg-slate-900/95 backdrop-blur-sm border-b border-slate-800">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <NavLink to="/" className="flex items-center space-x-2 text-xl font-bold text-cyan-400">
            <svg className="w-8 h-8" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M12 2a10 10 0 1 0 10 10A10 10 0 0 0 12 2zm0 18a8 8 0 1 1 8-8 8 8 0 0 1-8 8z" />
              <path d="M12 6v6l4 2" />
            </svg>
            <span>Brain Tumor Segmentation AI</span>
          </NavLink>

          <div className="hidden md:flex items-center space-x-8">
            <NavLink
              to="/"
              className={({ isActive }) =>
                `px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                  isActive ? 'bg-cyan-500/20 text-cyan-400' : 'text-slate-300 hover:text-cyan-400 hover:bg-slate-800'
                }`
              }
            >
              Home
            </NavLink>
            <NavLink
              to="/processing"
              className={({ isActive }) =>
                `px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                  isActive ? 'bg-cyan-500/20 text-cyan-400' : 'text-slate-300 hover:text-cyan-400 hover:bg-slate-800'
                }`
              }
            >
              Processing
            </NavLink>
            <NavLink
              to="/results"
              className={({ isActive }) =>
                `px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                  isActive ? 'bg-cyan-500/20 text-cyan-400' : 'text-slate-300 hover:text-cyan-400 hover:bg-slate-800'
                }`
              }
            >
              Results
            </NavLink>
          </div>

          <div className="md:hidden">
            <button className="text-slate-300 hover:text-cyan-400 p-2">
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
              </svg>
            </button>
          </div>
        </div>
      </div>
    </nav>
  );
}

export default Navbar;