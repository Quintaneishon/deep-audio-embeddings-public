import React, { useState, useEffect, useMemo, useRef } from 'react';
import DeckGL from '@deck.gl/react';
import { PointCloudLayer } from '@deck.gl/layers';
import { OrbitView } from '@deck.gl/core';
import { ThreeAudioVisualizer } from './ThreeAudioVisualizer';
import '../styles/deckgl.css';

const INITIAL_VIEW_STATE = {
  target: [0, 0, 0],
  rotationX: 0,
  rotationOrbit: 0,
  zoom: 0,
  minZoom: -5,
  maxZoom: 8
};

// Static list of 10 predefined colors
const STATIC_COLORS = [
  [255, 0, 0],       // Red
  [255, 105, 180],   // Pink
  [138, 43, 226],    // Purple
  [255, 215, 0],     // Gold
  [0, 255, 255],     // Cyan
  [255, 140, 0],     // Orange
  [34, 139, 34],     // Green
  [0, 0, 255],       // Blue
  [128, 128, 128],   // Gray
  [139, 69, 19]      // Brown
];

const DEFAULT_COLOR = [200, 200, 200];

// Generate a random color
const generateRandomColor = () => {
  return [
    Math.floor(Math.random() * 256),
    Math.floor(Math.random() * 256),
    Math.floor(Math.random() * 256)
  ];
};

// Create genre-to-color mapping dynamically from backend tags
const createGenreColorMap = async () => {
  try {
    const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://127.0.0.1:5000';
    const response = await fetch(`${API_BASE_URL}/tags`);
    const tags = await response.json();
    
    const colorMap = {};
    tags.forEach((tag, index) => {
      if (index < STATIC_COLORS.length) {
        // Use static color for first 10 tags
        colorMap[tag.toLowerCase()] = STATIC_COLORS[index];
      } else {
        // Generate random color for additional tags
        colorMap[tag.toLowerCase()] = generateRandomColor();
      }
    });
    
    return colorMap;
  } catch (error) {
    console.error('Error fetching tags from backend:', error);
    return {};
  }
};

// Function to get color for a genre
const getColorForGenre = (genre, colorMap) => {
  if (!genre || !colorMap) return DEFAULT_COLOR;
  const lowerGenre = genre.toLowerCase();
  return colorMap[lowerGenre] || DEFAULT_COLOR;
};

export const DeckGLVisualization = ({
  embeddings,
  dimensiones,
  onRefetchEmbeddings,
}) => {
  const [viewState, setViewState] = useState(INITIAL_VIEW_STATE);
  const [hoveredPoint, setHoveredPoint] = useState(null);
  const [showVisualizer, setShowVisualizer] = useState(false);
  const [currentAudioUrl, setCurrentAudioUrl] = useState(null);
  const [selectedPointInfo, setSelectedPointInfo] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [blinkingPoint, setBlinkingPoint] = useState(null);
  const [blinkVisible, setBlinkVisible] = useState(true); // Controls blink on/off state
  const blinkingRef = useRef(null); // Track the blinking point for the interval
  const dataZoomRef = useRef(4); // Store the appropriate zoom level for the data scale
  const [isUploading, setIsUploading] = useState(false);
  const [toast, setToast] = useState(null);
  const fileInputRef = useRef(null);
  const skipAutoCenterRef = useRef(false); // Flag to skip auto-centering when navigating to specific point
  const [genreColorMap, setGenreColorMap] = useState(null); // Dynamic genre-to-color mapping
  const [showUploadModal, setShowUploadModal] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploadGenreTag, setUploadGenreTag] = useState('');

  // Load genre color mapping on component mount
  useEffect(() => {
    const loadGenreColors = async () => {
      const colorMap = await createGenreColorMap();
      setGenreColorMap(colorMap);
    };
    loadGenreColors();
  }, []);

  // Log when data loads
  useEffect(() => {
    if (embeddings.length > 0) {
      console.log(`Loaded ${embeddings.length} audio embedding points`);
    }
  }, [embeddings]);

  // Transform embeddings data for deck.gl
  const points = useMemo(() => {
    if (!genreColorMap) return []; // Wait for color map to load
    
    return embeddings.map((embedding, index) => ({
      position: dimensiones === 3 
        ? [embedding.coords[0], embedding.coords[1], embedding.coords[2] || 0]
        : [embedding.coords[0], embedding.coords[1], 0],
      color: getColorForGenre(embedding.tag, genreColorMap),
      name: embedding.name,
      tag: embedding.tag,
      audio: embedding.audio,
      id: `${embedding.name}_${index}`
    }));
  }, [embeddings, dimensiones, genreColorMap]);

  // Calculate the center of the point cloud to properly position the camera
  useEffect(() => {
    console.log('🔍 Auto-center useEffect triggered. Skip flag:', skipAutoCenterRef.current, 'Points:', points.length);
    
    if (points.length > 0) {
      const sum = points.reduce((acc, point) => ({
        x: acc.x + point.position[0],
        y: acc.y + point.position[1],
        z: acc.z + point.position[2]
      }), { x: 0, y: 0, z: 0 });

      const center = [
        sum.x / points.length,
        sum.y / points.length,
        sum.z / points.length
      ];

      // Calculate the spread/scale of the data to set appropriate zoom
      const distances = points.map(p => {
        const dx = p.position[0] - center[0];
        const dy = p.position[1] - center[1];
        const dz = p.position[2] - center[2];
        return Math.sqrt(dx * dx + dy * dy + dz * dz);
      });
      const maxDistance = Math.max(...distances);
      
      // Set zoom based on data scale (larger spread = zoom out more)
      const baseZoom = Math.max(-2, 4 - Math.log2(maxDistance));
      const targetZoom = baseZoom + 5;

      console.log('Point cloud center:', center);
      console.log('Point cloud max distance:', maxDistance);
      console.log('Base zoom level:', baseZoom);
      console.log('Target zoom level:', targetZoom);
      console.log('Total points:', points.length);

      // Store the target zoom for navigation
      dataZoomRef.current = targetZoom;

      // Skip auto-centering if we're navigating to a specific point
      if (skipAutoCenterRef.current) {
        console.log('✓ Skipping auto-center (navigating to specific point)');
        return;
      }
      
      console.log('Performing auto-center for all points');

      // Start at base zoom, then animate to target zoom
      setViewState(prev => ({
        ...prev,
        target: center,
        zoom: baseZoom,
        transitionDuration: 0
      }));

      // Animate zoom in after a short delay
      setTimeout(() => {
        setViewState(prev => ({
          ...prev,
          zoom: targetZoom,
          transitionDuration: 2000 // 2 second smooth zoom animation
        }));
      }, 100);
    }
  }, [points]);

  // Handle keyboard events (ESC to close visualizer)
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === "Escape") {
        if (showVisualizer) {
          setShowVisualizer(false);
          setCurrentAudioUrl(null);
          setSelectedPointInfo(null);
          console.log("Visualizer closed with ESC");
        }
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [showVisualizer]);

  // Filter songs based on search query
  const filteredSongs = useMemo(() => {
    if (!searchQuery.trim()) return [];
    
    const query = searchQuery.toLowerCase();
    return points.filter(point => 
      point.name.toLowerCase().includes(query) ||
      point.tag.toLowerCase().includes(query)
    ).slice(0, 50); // Limit to 50 results for performance
  }, [searchQuery, points]);

  // Clear blinking marker when search query changes
  useEffect(() => {
    if (searchQuery) {
      setBlinkingPoint(null);
    }
  }, [searchQuery]);

  // Animate the blinking effect by toggling hover on/off
  useEffect(() => {
    if (!blinkingPoint) {
      blinkingRef.current = null;
      setBlinkVisible(false);
      return;
    }
    
    blinkingRef.current = blinkingPoint;
    let isOn = true;
    setBlinkVisible(true);
    
    const interval = setInterval(() => {
      if (blinkingRef.current) {
        isOn = !isOn;
        setBlinkVisible(isOn);
        setHoveredPoint(isOn ? blinkingRef.current : null);
      }
    }, 600); // Toggle every 600ms for slower, easier to click blinking
    
    return () => clearInterval(interval);
  }, [blinkingPoint]);

  // Show toast notification
  const showToast = (message, type = 'error') => {
    setToast({ message, type });
    setTimeout(() => setToast(null), 5000); // Auto-dismiss after 5 seconds
  };

  // Handle file selection (opens modal)
  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (!file) return;

    // Validate file type
    if (!file.name.toLowerCase().endsWith('.mp3')) {
      showToast('Only MP3 files are supported', 'error');
      return;
    }

    setSelectedFile(file);
    setUploadGenreTag('');
    setShowUploadModal(true);
  };

  // Handle file upload with genre tag
  const handleFileUpload = async () => {
    if (!selectedFile) return;

    setIsUploading(true);
    setShowUploadModal(false);
    
    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('genre', uploadGenreTag.trim());

    try {
      console.log('Uploading file:', selectedFile.name, 'with genre:', uploadGenreTag);
      const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://127.0.0.1:5000';
      const response = await fetch(`${API_BASE_URL}/upload`, {
        method: 'POST',
        body: formData,
      });

      const result = await response.json();
      console.log('Upload response:', result);

      // Check if the response was successful (status 200-299)
      if (!response.ok) {
        console.error('Upload failed with status:', response.status);
        showToast(result.error || `Upload failed with status ${response.status}`, 'error');
        return;
      }

      if (result.success) {
        showToast('Song uploaded successfully!', 'success');
        
        // Wait a bit for backend to update before refetching
        await new Promise(resolve => setTimeout(resolve, 500));
        
        // Refetch embeddings - this will trigger re-projection and update points
        if (onRefetchEmbeddings) {
          console.log('Refetching embeddings...');
          const newEmbeddings = await onRefetchEmbeddings();
          console.log('📊 Received embeddings count:', newEmbeddings?.length);
          
          if (!newEmbeddings || newEmbeddings.length === 0) {
            console.error('❌ No embeddings returned after refetch');
            showToast('Failed to load updated embeddings', 'error');
          }
        }
      } else {
        console.error('Upload result indicated failure:', result);
        showToast(result.error || 'Failed to upload file', 'error');
      }
    } catch (error) {
      console.error('Upload error:', error);
      showToast(`Error: ${error.message}`, 'error');
    } finally {
      setIsUploading(false);
      setSelectedFile(null);
      setUploadGenreTag('');
      // Reset file input
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  // Cancel upload modal
  const handleCancelUpload = () => {
    setShowUploadModal(false);
    setSelectedFile(null);
    setUploadGenreTag('');
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  // Navigate to a specific point without playing audio
  const navigateToPoint = (point) => {
    // Set flag to prevent auto-centering from interfering
    skipAutoCenterRef.current = true;
    
    // Clear previous blinking point before setting new one
    setBlinkingPoint(null);
    
    setViewState(prev => ({
      ...prev,
      target: point.position,
      // ZOOM OF SEARCHED POINT
      zoom: dataZoomRef.current + 1, // Zoom in closer to the point
      transitionDuration: 1000 // Smooth 1 second animation
    }));
    
    // Set blinking marker (stays until user clicks a point or searches again)
    setBlinkingPoint(point);
    setHoveredPoint(point);
    
    console.log('Navigated to:', point.name);
    
    // Reset the skip flag after navigation animation completes
    setTimeout(() => {
      skipAutoCenterRef.current = false;
    }, 1500);
  };

  const layers = [
    new PointCloudLayer({
      id: 'point-cloud-layer',
      data: points,
      getPosition: d => d.position,
      getColor: d => d.color,
      pointSize: 5,
      sizeUnits: 'pixels',
      pickable: true,
      autoHighlight: true,
      highlightColor: [255, 255, 0, 200],
      onHover: (info) => {
        if (info.object) {
          setHoveredPoint(info.object);
        } else {
          setHoveredPoint(null);
        }
      },
      onClick: (info) => {
        if (info.object) {
          const point = info.object;
          console.log('Clicked:', point.name, point.tag);
          
          // Clear blinking point when clicking on any point
          setBlinkingPoint(null);
          
          // Open Three.js audio visualizer
          if (point.audio) {
            const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://127.0.0.1:5000';
            const audioUrl = `${API_BASE_URL}/audio/${point.audio}`;
            console.log("Opening visualizer for:", audioUrl);
            
            setCurrentAudioUrl(audioUrl);
            setSelectedPointInfo({
              name: point.name,
              tag: point.tag,
              position: point.position
            });
            setShowVisualizer(true);
          }
        }
      }
    }),
    // Blinking marker layer - shows a large bright marker at the searched point
    blinkingPoint && blinkVisible && new PointCloudLayer({
      id: 'blinking-marker-layer',
      data: [blinkingPoint],
      getPosition: d => d.position,
      getColor: [255, 255, 0], // Bright yellow
      // SIZE OF BLINKING MARKER
      pointSize: 8, // Much larger than regular points
      sizeUnits: 'pixels',
      pickable: true,
      onClick: (info) => {
        if (info.object) {
          const point = info.object;
          console.log('Clicked blinking marker:', point.name, point.tag);
          if (point.audio) {
            const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://127.0.0.1:5000';
            const audioUrl = `${API_BASE_URL}/audio/${point.audio}`;
            setCurrentAudioUrl(audioUrl);
            setSelectedPointInfo({ name: point.name, tag: point.tag, position: point.position });
            setShowVisualizer(true);
          }
        }
      }
    }),
    // Outer ring effect for blinking marker
    blinkingPoint && !blinkVisible && new PointCloudLayer({
      id: 'blinking-marker-ring-layer',
      data: [blinkingPoint],
      getPosition: d => d.position,
      getColor: DEFAULT_COLOR, // Orange when "off"
      // SIZE OF SEARCHED POINT
      pointSize: 5,
      sizeUnits: 'pixels',
      pickable: true,
      onClick: (info) => {
        if (info.object) {
          const point = info.object;
          console.log('Clicked blinking marker:', point.name, point.tag);
          if (point.audio) {
            const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://127.0.0.1:5000';
            const audioUrl = `${API_BASE_URL}/audio/${point.audio}`;
            setCurrentAudioUrl(audioUrl);
            setSelectedPointInfo({ name: point.name, tag: point.tag, position: point.position });
            setShowVisualizer(true);
          }
        }
      }
    })
  ].filter(Boolean);

  return (
    <div className="deckgl-container">
      {/* Hidden file input */}
      <input
        ref={fileInputRef}
        type="file"
        accept=".mp3"
        onChange={handleFileSelect}
        style={{ display: 'none' }}
      />

      {/* Upload Modal */}
      {showUploadModal && (
        <div style={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          backgroundColor: 'rgba(0, 0, 0, 0.8)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 200
        }}>
          <div style={{
            backgroundColor: 'rgba(20, 20, 20, 0.95)',
            borderRadius: '12px',
            padding: '32px',
            width: '400px',
            maxWidth: '90%',
            boxShadow: '0 8px 32px rgba(0, 0, 0, 0.7)',
            border: '1px solid rgba(255, 255, 255, 0.1)'
          }}>
            <h2 style={{
              color: 'white',
              margin: '0 0 24px 0',
              fontSize: '20px',
              fontWeight: '600'
            }}>
              Upload Audio File
            </h2>
            
            <div style={{ marginBottom: '20px' }}>
              <div style={{
                color: 'rgba(255, 255, 255, 0.7)',
                fontSize: '14px',
                marginBottom: '8px'
              }}>
                Selected file:
              </div>
              <div style={{
                color: 'white',
                fontSize: '14px',
                padding: '10px 12px',
                backgroundColor: 'rgba(255, 255, 255, 0.05)',
                borderRadius: '6px',
                wordBreak: 'break-all'
              }}>
                {selectedFile?.name}
              </div>
            </div>

            <div style={{ marginBottom: '24px' }}>
              <label style={{
                display: 'block',
                color: 'rgba(255, 255, 255, 0.7)',
                fontSize: '14px',
                marginBottom: '8px'
              }}>
                Genre Tag (optional)
              </label>
              <input
                type="text"
                placeholder="e.g., rock, jazz, classical..."
                value={uploadGenreTag}
                onChange={(e) => setUploadGenreTag(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    handleFileUpload();
                  } else if (e.key === 'Escape') {
                    handleCancelUpload();
                  }
                }}
                autoFocus
                style={{
                  width: '100%',
                  padding: '10px 12px',
                  backgroundColor: 'rgba(255, 255, 255, 0.1)',
                  border: '1px solid rgba(255, 255, 255, 0.2)',
                  borderRadius: '6px',
                  color: 'white',
                  fontSize: '14px',
                  outline: 'none',
                  transition: 'border-color 0.2s',
                  boxSizing: 'border-box'
                }}
                onFocus={(e) => {
                  e.target.style.borderColor = 'rgba(255, 255, 255, 0.4)';
                }}
                onBlur={(e) => {
                  e.target.style.borderColor = 'rgba(255, 255, 255, 0.2)';
                }}
              />
              {genreColorMap && Object.keys(genreColorMap).length > 0 && (
                <div style={{
                  marginTop: '8px',
                  fontSize: '12px',
                  color: 'rgba(255, 255, 255, 0.5)'
                }}>
                  Existing genres: {Object.keys(genreColorMap).slice(0, 5).join(', ')}
                  {Object.keys(genreColorMap).length > 5 && '...'}
                </div>
              )}
            </div>

            <div style={{
              display: 'flex',
              gap: '12px',
              justifyContent: 'flex-end'
            }}>
              <button
                onClick={handleCancelUpload}
                style={{
                  padding: '10px 20px',
                  backgroundColor: 'rgba(255, 255, 255, 0.1)',
                  border: '1px solid rgba(255, 255, 255, 0.2)',
                  borderRadius: '6px',
                  color: 'white',
                  fontSize: '14px',
                  cursor: 'pointer',
                  transition: 'all 0.2s'
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.backgroundColor = 'rgba(255, 255, 255, 0.15)';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.backgroundColor = 'rgba(255, 255, 255, 0.1)';
                }}
              >
                Cancel
              </button>
              <button
                onClick={handleFileUpload}
                style={{
                  padding: '10px 20px',
                  backgroundColor: 'rgba(59, 130, 246, 0.9)',
                  border: 'none',
                  borderRadius: '6px',
                  color: 'white',
                  fontSize: '14px',
                  cursor: 'pointer',
                  transition: 'all 0.2s',
                  fontWeight: '500'
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.backgroundColor = 'rgba(59, 130, 246, 1)';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.backgroundColor = 'rgba(59, 130, 246, 0.9)';
                }}
              >
                Upload
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Loading Overlay */}
      {isUploading && (
        <div style={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          backgroundColor: 'rgba(0, 0, 0, 0.7)',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 100,
          color: 'white'
        }}>
          <div style={{
            width: '50px',
            height: '50px',
            border: '5px solid rgba(255, 255, 255, 0.3)',
            borderTop: '5px solid white',
            borderRadius: '50%',
            animation: 'spin 1s linear infinite'
          }} />
          <p style={{ marginTop: '20px', fontSize: '16px' }}>
            Processing audio file...
          </p>
          <p style={{ marginTop: '8px', fontSize: '12px', opacity: 0.7 }}>
            This may take a few minutes
          </p>
        </div>
      )}

      {/* Toast Notification */}
      {toast && (
        <div style={{
          position: 'absolute',
          top: '70px',
          left: '50%',
          transform: 'translateX(-50%)',
          backgroundColor: toast.type === 'error' ? 'rgba(220, 38, 38, 0.95)' : 'rgba(34, 197, 94, 0.95)',
          color: 'white',
          padding: '12px 24px',
          borderRadius: '8px',
          fontSize: '14px',
          zIndex: 1000,
          boxShadow: '0 4px 12px rgba(0, 0, 0, 0.5)',
          maxWidth: '400px',
          textAlign: 'center'
        }}>
          {toast.message}
        </div>
      )}

      <DeckGL
        views={new OrbitView()}
        viewState={viewState}
        onViewStateChange={({ viewState }) => setViewState(viewState)}
        controller={true}
        layers={layers}
        parameters={{
          clearColor: [0.07, 0.07, 0.07, 1]
        }}
      >
        {hoveredPoint && (
          <div className="tooltip" style={{
            position: 'absolute',
            zIndex: 1,
            pointerEvents: 'none',
            left: '50%',
            top: 10,
            transform: 'translateX(-50%)',
            backgroundColor: 'rgba(0, 0, 0, 0.8)',
            color: 'white',
            padding: '8px 12px',
            borderRadius: '4px',
            fontSize: '14px'
          }}>
            <div><strong>{hoveredPoint.name}</strong></div>
            <div>Genre: {hoveredPoint.tag}</div>
            <div style={{ fontSize: '12px', marginTop: '4px', opacity: 0.8 }}>
              Click to visualize audio
            </div>
          </div>
        )}
      </DeckGL>

      {/* Embedded Audio Visualizer Panel - Bottom Right */}
      {showVisualizer && currentAudioUrl && selectedPointInfo && (
        <div className="visualizer-panel" style={{
          position: 'absolute',
          bottom: 10,
          right: 10,
          width: '340px',
          height: '180px',
          backgroundColor: 'rgba(0, 0, 0, 0.9)',
          borderRadius: '8px',
          zIndex: 1,
          overflow: 'hidden',
          boxShadow: '0 4px 20px rgba(0, 0, 0, 0.5)'
        }}>
          <ThreeAudioVisualizer
            audioUrl={currentAudioUrl}
            songName={selectedPointInfo.name}
            genre={selectedPointInfo.tag}
            coordinates={selectedPointInfo.position}
            genreColorMap={genreColorMap}
            onClose={() => {
              setShowVisualizer(false);
              setCurrentAudioUrl(null);
              setSelectedPointInfo(null);
            }}
          />
        </div>
      )}

       {/* Legend */}
       <div className="legend" style={{
        position: 'absolute',
        top: 60,
        right: 10,
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        color: 'white',
        padding: '12px',
        borderRadius: '4px',
        fontSize: '12px',
        maxHeight: 'calc(100vh - 220px)',
        overflowY: 'auto',
        zIndex: 1
      }}>
        <div style={{ fontWeight: 'bold', marginBottom: '8px' }}>Genres</div>
        {genreColorMap && Object.entries(genreColorMap).map(([genre, color]) => (
          <div key={genre} style={{ display: 'flex', alignItems: 'center', marginBottom: '4px' }}>
            <div style={{
              width: '12px',
              height: '12px',
              backgroundColor: `rgb(${color[0]}, ${color[1]}, ${color[2]})`,
              marginRight: '8px',
              borderRadius: '2px'
            }} />
            <span>{genre}</span>
          </div>
        ))}
      </div>

      {/* Search Panel */}
      <div className="search-panel" style={{
        position: 'absolute',
        bottom: 10,
        left: 10,
        backgroundColor: 'rgba(0, 0, 0, 0.9)',
        color: 'white',
        padding: '12px',
        borderRadius: '8px',
        fontSize: '13px',
        zIndex: 1,
        width: '320px',
        maxHeight: '400px',
        display: 'flex',
        flexDirection: 'column',
        boxShadow: '0 4px 12px rgba(0, 0, 0, 0.5)'
      }}>
        {/* Upload Button - Top Right of Search Panel */}
        <button
          onClick={() => fileInputRef.current?.click()}
          disabled={isUploading}
          style={{
            position: 'absolute',
            top: '-18px',
            right: '-18px',
            width: '40px',
            height: '40px',
            borderRadius: '50%',
            backgroundColor: isUploading ? 'rgba(100, 100, 100, 0.9)' : 'rgba(0, 0, 0, 0.9)',
            border: '2px solid rgba(255, 255, 255, 0.3)',
            color: 'white',
            fontSize: '22px',
            fontWeight: 'bold',
            cursor: isUploading ? 'not-allowed' : 'pointer',
            zIndex: 10,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            transition: 'all 0.2s',
            boxShadow: '0 4px 12px rgba(0, 0, 0, 0.5)'
          }}
          onMouseEnter={(e) => {
            if (!isUploading) {
              e.currentTarget.style.backgroundColor = 'rgba(50, 50, 50, 0.9)';
              e.currentTarget.style.transform = 'scale(1.05)';
            }
          }}
          onMouseLeave={(e) => {
            if (!isUploading) {
              e.currentTarget.style.backgroundColor = 'rgba(0, 0, 0, 0.9)';
              e.currentTarget.style.transform = 'scale(1)';
            }
          }}
          title="Upload MP3 file"
        >
          {isUploading ? '...' : '+'}
        </button>

        <div style={{ marginBottom: '8px', fontWeight: 'bold', fontSize: '14px' }}>
        Search Songs
        </div>
        
        <input
          type="text"
          placeholder="Type song name or genre..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          style={{
            padding: '8px 12px',
            backgroundColor: 'rgba(255, 255, 255, 0.1)',
            border: '1px solid rgba(255, 255, 255, 0.3)',
            borderRadius: '4px',
            color: 'white',
            fontSize: '13px',
            outline: 'none',
            marginBottom: '8px'
          }}
        />
        
        {searchQuery && (
          <div style={{
            fontSize: '11px',
            opacity: 0.7,
            marginBottom: '8px'
          }}>
            {filteredSongs.length} {filteredSongs.length === 1 ? 'result' : 'results'} found
          </div>
        )}
        
        {searchQuery && filteredSongs.length > 0 && (
          <div style={{
            overflowY: 'auto',
            maxHeight: '280px',
            marginTop: '4px'
          }}>
            {filteredSongs.map((point, index) => (
              <div
                key={point.id}
                onClick={() => {
                  navigateToPoint(point);
                  setSearchQuery('');
                }}
                style={{
                  padding: '8px',
                  marginBottom: '4px',
                  backgroundColor: 'rgba(255, 255, 255, 0.05)',
                  borderRadius: '4px',
                  cursor: 'pointer',
                  transition: 'all 0.2s',
                  border: '1px solid transparent'
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.backgroundColor = 'rgba(255, 255, 255, 0.15)';
                  e.currentTarget.style.borderColor = 'rgba(255, 255, 255, 0.3)';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.backgroundColor = 'rgba(255, 255, 255, 0.05)';
                  e.currentTarget.style.borderColor = 'transparent';
                }}
              >
                <div style={{ 
                  fontWeight: '500', 
                  marginBottom: '2px',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  whiteSpace: 'nowrap'
                }}>
                  {point.name}
                </div>
                <div style={{
                  fontSize: '11px',
                  opacity: 0.7,
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px',
                  overflow: 'hidden'
                }}>
                  <span style={{
                    display: 'inline-block',
                    width: '8px',
                    height: '8px',
                    minWidth: '8px',
                    backgroundColor: `rgb(${point.color[0]}, ${point.color[1]}, ${point.color[2]})`,
                    borderRadius: '50%'
                  }} />
                  <span style={{
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap'
                  }}>
                    {point.tag}
                  </span>
                </div>
              </div>
            ))}
          </div>
        )}
        
        {searchQuery && filteredSongs.length === 0 && (
          <div style={{
            padding: '16px',
            textAlign: 'center',
            opacity: 0.6,
            fontSize: '12px'
          }}>
            No songs found matching "{searchQuery}"
          </div>
        )}
      </div>
    </div>
  );
};
