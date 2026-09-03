import 'bootstrap/dist/css/bootstrap.min.css';
import './App.css';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { HomePage } from './pages/HomePage';
import { VisualizePage } from './pages/VisualizePage';
import { GraphPage } from './pages/GraphPage';
import { ParticlesPage } from './pages/ParticlesPage';
import { EvalPage } from './pages/EvalPage';

function App() {
  return (
    <Router>
      <div className='main'>
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/visualize" element={<VisualizePage />} />
          <Route path="/graph" element={<GraphPage />} />
          <Route path="/particles" element={<ParticlesPage />} />
          <Route path="/eval" element={<EvalPage />} />
        </Routes>
      </div>
      <footer style={{
        position: 'fixed',
        bottom: 0,
        width: '100%',
        textAlign: 'center',
        padding: '6px 0',
        fontSize: '11px',
        fontFamily: 'monospace',
        color: '#4b5563',
        background: 'transparent',
        pointerEvents: 'none',
        userSelect: 'none',
      }}>
        Ajitzi Quintana · UNAM · 2026
      </footer>
    </Router>
  );
}

export default App;
