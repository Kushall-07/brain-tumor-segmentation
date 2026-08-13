import { Outlet } from 'react-router-dom';
import Navbar from "../components/Navbar";
import Footer from "../components/layout/Footer";

export function MainLayout() {
  return (
    <div className="min-h-screen bg-parchment text-ink-body flex flex-col font-sans selection:bg-arterial/20 selection:text-ink">
      <Navbar />
      <main className="flex-1 relative z-0">
        <Outlet />
      </main>
      <Footer />
    </div>
  );
}

export default MainLayout;
