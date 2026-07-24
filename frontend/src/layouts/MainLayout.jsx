import { Outlet } from 'react-router-dom';
import Navbar from '../components/Navbar';

export default function MainLayout() {
  return (
    <div className="min-h-screen bg-slate-950 text-white">
      <Navbar />
      <main className="pt-16 min-h-[calc(100vh-4rem)]">
        <Outlet />
      </main>
      <footer className="border-t border-slate-800 py-6 px-4">
        <div className="max-w-7xl mx-auto text-center text-sm text-slate-500">
          <p>Brain Tumor Segmentation AI &copy; 2025</p>
          <p className="mt-1">Powered by SwinUNETR &bull; For Research Use Only</p>
        </div>
      </footer>
    </div>
  );
}