import { createBrowserRouter, Navigate } from 'react-router-dom';
import MainLayout from '../layouts/MainLayout';
import Home from '../pages/Home';
import PredictPage from '../pages/PredictPage';
import Processing from '../pages/Processing';
import Results from '../pages/Results';

/**
 * Scalable React Router configuration for Phase 1 and future Phase 2 expansion.
 */
export const router = createBrowserRouter([
  {
    path: '/',
    element: <MainLayout />,
    children: [
      {
        index: true,
        element: <Home />,
      },
      {
        path: 'predict',
        element: <PredictPage />,
      },
      {
        path: 'processing',
        element: <Processing />,
      },
      {
        path: 'results',
        element: <Results />,
      },
      {
        path: '*',
        element: <Navigate to="/" replace />,
      },
    ],
  },
]);

export default router;