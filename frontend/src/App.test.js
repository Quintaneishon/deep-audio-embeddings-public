import { render, screen } from '@testing-library/react';
import App from './App';

jest.mock('react-router-dom', () => ({
  BrowserRouter: ({ children }) => children,
  Routes: ({ children }) => children,
  Route: ({ element }) => element,
}), { virtual: true });

jest.mock('./pages/HomePage', () => ({ HomePage: () => <main>Embedding explorer</main> }));
jest.mock('./pages/VisualizePage', () => ({ VisualizePage: () => null }));
jest.mock('./pages/GraphPage', () => ({ GraphPage: () => null }));
jest.mock('./pages/ParticlesPage', () => ({ ParticlesPage: () => null }));
jest.mock('./pages/EvalPage', () => ({ EvalPage: () => null }));

test('renders the application shell', () => {
  render(<App />);
  expect(screen.getByText(/embedding explorer/i)).toBeInTheDocument();
  expect(screen.getByText(/Ajitzi Quintana · UNAM · 2026/i)).toBeInTheDocument();
});
