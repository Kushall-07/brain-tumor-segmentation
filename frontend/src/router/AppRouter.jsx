import { createBrowserRouter } from 'react-router-dom';
import MainLayout from '../layouts/MainLayout';
import Home from '../pages/Home';
import Processing from '../pages/Processing';
import Results from '../pages/Results';

export const router = createBrowserRouter([
  {
    path: '/',
    element: <MainLayout />,
    children: [
      { index: true, element: <Home /> },
      { path: 'processing', element: <Processing /> },
      { path: 'results', element: <Results /> },
    ],
  },
]);