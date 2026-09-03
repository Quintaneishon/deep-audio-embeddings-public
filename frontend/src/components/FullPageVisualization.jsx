import React, { useState, useEffect } from 'react';
import { DeckGLVisualization } from './DeckGLVisualization';
import '../styles/fullpage.css';

export const FullPageVisualization = () => {
  const [embeddings, setEmbeddings] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchEmbeddings = async () => {
    try {
      setLoading(true);
      const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://127.0.0.1:5000';
      const url = `${API_BASE_URL}/embeddings?red=whisper_contrastive&dataset=base&metodo=umap&dimensions=3`;
      console.log('Fetching embeddings from:', url);
      
      const response = await fetch(url);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const result = await response.json();
      console.log('Embeddings loaded:', result.data?.length || 0, 'points');
      
      setEmbeddings(result.data || []);
      setError(null);
      return result.data || [];
    } catch (err) {
      console.error('Error fetching embeddings:', err);
      setError(err.message);
      return [];
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchEmbeddings();
  }, []);

  if (loading) {
    return (
      <div className="fullpage-container">
        <div className="loading-overlay">
          <div className="loading-spinner"></div>
          <p>Loading 3D audio embeddings...</p>
          <p className="loading-subtitle">Fetching Whisper Contrastive embeddings with UMAP</p>
        </div>
      </div>
    );
  }

  if (error) {
    const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://127.0.0.1:5000';
    return (
      <div className="fullpage-container">
        <div className="error-overlay">
          <div className="error-icon">⚠️</div>
          <h2>Error Loading Data</h2>
          <p>{error}</p>
          <p className="error-subtitle">Make sure the backend server is running at {API_BASE_URL}</p>
          <button 
            className="retry-button"
            onClick={() => window.location.reload()}
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  if (embeddings.length === 0) {
    return (
      <div className="fullpage-container">
        <div className="error-overlay">
          <p>No embeddings data available</p>
        </div>
      </div>
    );
  }

  return (
    <div className="fullpage-container">
      <div className="visualization-header">
        <div className="header-info">
          <span className="info-badge">Whisper Contrastive</span>
          <span className="info-badge">UMAP</span>
          <span className="info-badge">3D</span>
          <span className="info-badge">{embeddings.length} points</span>
        </div>
      </div>
      
      <div className="visualization-content">
        <DeckGLVisualization
          embeddings={embeddings}
          dimensiones={3}
          agruparPor="tag"
          onRefetchEmbeddings={fetchEmbeddings}
        />
      </div>
    </div>
  );
};

