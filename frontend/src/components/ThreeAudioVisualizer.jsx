import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';

// Default color for songs without genre (cyan theme)
const DEFAULT_COLOR = [0, 255, 255]; // Cyan

// Function to convert RGB array [0-255] to HSL
const rgbToHsl = (r, g, b) => {
  r /= 255;
  g /= 255;
  b /= 255;
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  let h, s, l = (max + min) / 2;

  if (max === min) {
    h = s = 0;
  } else {
    const d = max - min;
    s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
    switch (max) {
      case r: h = ((g - b) / d + (g < b ? 6 : 0)) / 6; break;
      case g: h = ((b - r) / d + 2) / 6; break;
      case b: h = ((r - g) / d + 4) / 6; break;
      default: h = 0;
    }
  }
  return [h * 360, s * 100, l * 100];
};

// Get color for a genre
const getColorForGenre = (genre, colorMap) => {
  if (!genre || !colorMap || Object.keys(colorMap).length === 0) {
    return DEFAULT_COLOR;
  }
  const lowerGenre = genre.toLowerCase();
  return colorMap[lowerGenre] || DEFAULT_COLOR;
};

export const ThreeAudioVisualizer = ({ audioUrl, songName, genre, coordinates, genreColorMap, onClose }) => {
  const containerRef = useRef(null);
  const rendererRef = useRef(null);
  const sceneRef = useRef(null);
  const cameraRef = useRef(null);
  const analyserRef = useRef(null);
  const audioContextRef = useRef(null);
  const sourceRef = useRef(null);
  const animationFrameRef = useRef(null);
  const audioElementRef = useRef(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [genreColor, setGenreColor] = useState(DEFAULT_COLOR);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [isSeeking, setIsSeeking] = useState(false);

  // CSS for custom slider styling with genre color
  const sliderStyles = `
    input[type="range"]::-webkit-slider-thumb {
      -webkit-appearance: none;
      appearance: none;
      width: 14px;
      height: 14px;
      border-radius: 50%;
      background: rgb(${genreColor[0]}, ${genreColor[1]}, ${genreColor[2]});
      cursor: pointer;
      box-shadow: 0 0 6px rgba(${genreColor[0]}, ${genreColor[1]}, ${genreColor[2]}, 0.7);
      transition: all 0.15s ease-in-out;
    }
    
    input[type="range"]::-webkit-slider-thumb:hover {
      transform: scale(1.3);
      box-shadow: 0 0 12px rgba(${genreColor[0]}, ${genreColor[1]}, ${genreColor[2]}, 1);
    }
    
    input[type="range"]::-moz-range-thumb {
      width: 14px;
      height: 14px;
      border-radius: 50%;
      background: rgb(${genreColor[0]}, ${genreColor[1]}, ${genreColor[2]});
      cursor: pointer;
      border: none;
      box-shadow: 0 0 6px rgba(${genreColor[0]}, ${genreColor[1]}, ${genreColor[2]}, 0.7);
      transition: all 0.15s ease-in-out;
    }
    
    input[type="range"]::-moz-range-thumb:hover {
      transform: scale(1.3);
      box-shadow: 0 0 12px rgba(${genreColor[0]}, ${genreColor[1]}, ${genreColor[2]}, 1);
    }
  `;

  // Fetch genre color on mount or when genre changes
  useEffect(() => {
    if (genreColorMap && genre) {
      const color = getColorForGenre(genre, genreColorMap);
      setGenreColor(color);
      console.log(`Genre: ${genre}, Color:`, color);
    }
  }, [genre, genreColorMap]);

  useEffect(() => {
    if (!containerRef.current) return;
    
    const container = containerRef.current;

    // Initialize Three.js scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0a1a);
    sceneRef.current = scene;

    // Setup orthographic camera for full-screen effect
    const aspect = container.clientWidth / container.clientHeight;
    const frustumSize = 2;
    const camera = new THREE.OrthographicCamera(
      frustumSize * aspect / -2,
      frustumSize * aspect / 2,
      frustumSize / 2,
      frustumSize / -2,
      0.1,
      1000
    );
    camera.position.z = 5;
    cameraRef.current = camera;

    // Setup renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    container.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // Create canvas for spectrum visualization
    const canvas = document.createElement('canvas');
    canvas.width = 512;
    canvas.height = 256;
    const ctx = canvas.getContext('2d');
    
    // Create texture from canvas
    const texture = new THREE.CanvasTexture(canvas);
    
    // Create plane to display the spectrum (fill the viewport)
    const planeWidth = frustumSize * aspect;
    const planeHeight = frustumSize;
    const planeGeometry = new THREE.PlaneGeometry(planeWidth, planeHeight);
    const planeMaterial = new THREE.MeshBasicMaterial({ 
      map: texture,
      side: THREE.DoubleSide
    });
    const plane = new THREE.Mesh(planeGeometry, planeMaterial);
    scene.add(plane);

    // Animation loop
    const animate = () => {
      animationFrameRef.current = requestAnimationFrame(animate);

      if (analyserRef.current && ctx) {
        const dataArray = new Uint8Array(analyserRef.current.frequencyBinCount);
        analyserRef.current.getByteFrequencyData(dataArray);

        // Convert genre color to HSL for dynamic color generation
        const [baseHue, baseSat] = rgbToHsl(genreColor[0], genreColor[1], genreColor[2]);

        // Clear canvas with gradient background (with subtle genre color tint)
        const gradient = ctx.createLinearGradient(0, 0, 0, canvas.height);
        gradient.addColorStop(0, `rgba(${genreColor[0] * 0.05}, ${genreColor[1] * 0.05}, ${genreColor[2] * 0.05}, 1)`);
        gradient.addColorStop(0.5, `rgba(${genreColor[0] * 0.08}, ${genreColor[1] * 0.08}, ${genreColor[2] * 0.08}, 1)`);
        gradient.addColorStop(1, `rgba(${genreColor[0] * 0.05}, ${genreColor[1] * 0.05}, ${genreColor[2] * 0.05}, 1)`);
        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // Draw spectrum with genre-based color theme
        const barWidth = canvas.width / dataArray.length;
        const centerY = canvas.height / 2;
        
        for (let i = 0; i < dataArray.length; i++) {
          const barHeight = (dataArray[i] / 255) * (canvas.height * 0.45);
          
          // Create color based on frequency intensity using genre color
          const hue = baseHue + (i / dataArray.length) * 40 - 20; // Vary hue slightly around base color
          const intensity = dataArray[i] / 255;
          
          // Draw mirrored bars from center with genre color
          const gradient2 = ctx.createLinearGradient(0, centerY - barHeight, 0, centerY + barHeight);
          gradient2.addColorStop(0, `hsla(${hue}, ${baseSat}%, ${intensity * 50 + 30}%, 0.85)`);
          gradient2.addColorStop(0.5, `hsla(${hue}, ${Math.min(baseSat + 10, 100)}%, ${intensity * 60 + 40}%, 1)`);
          gradient2.addColorStop(1, `hsla(${hue}, ${baseSat}%, ${intensity * 50 + 30}%, 0.85)`);
          
          ctx.fillStyle = gradient2;
          ctx.fillRect(i * barWidth, centerY - barHeight, barWidth - 1, barHeight * 2);
        }

        // Add stronger glow effect with genre color
        ctx.shadowBlur = 20;
        ctx.shadowColor = `rgba(${genreColor[0]}, ${genreColor[1]}, ${genreColor[2]}, 0.5)`;

        // Update texture
        texture.needsUpdate = true;
      }

      renderer.render(scene, camera);
    };
    animate();

    // Handle window resize
    const handleResize = () => {
      if (!container) return;
      const newAspect = container.clientWidth / container.clientHeight;
      camera.left = frustumSize * newAspect / -2;
      camera.right = frustumSize * newAspect / 2;
      camera.top = frustumSize / 2;
      camera.bottom = frustumSize / -2;
      camera.updateProjectionMatrix();
      
      // Update plane size to match new aspect ratio
      const newPlaneWidth = frustumSize * newAspect;
      plane.scale.x = newPlaneWidth / planeWidth;
      
      renderer.setSize(container.clientWidth, container.clientHeight);
    };
    window.addEventListener('resize', handleResize);

    // Cleanup
    return () => {
      window.removeEventListener('resize', handleResize);
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      if (container && renderer.domElement && container.contains(renderer.domElement)) {
        container.removeChild(renderer.domElement);
      }
      renderer.dispose();
      planeGeometry.dispose();
      planeMaterial.dispose();
      texture.dispose();
    };
  }, [genreColor]);

  const playAudio = async () => {
    try {
      // Create audio context
      const audioContext = new (window.AudioContext || window.webkitAudioContext)();
      audioContextRef.current = audioContext;

      // Create analyser
      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 256;
      analyserRef.current = analyser;

      // Create audio element
      const audio = new Audio(audioUrl);
      audio.crossOrigin = "anonymous"; // Enable CORS for Web Audio API
      audioElementRef.current = audio;

      // Create media source
      const source = audioContext.createMediaElementSource(audio);
      sourceRef.current = source;

      // Connect audio graph
      source.connect(analyser);
      analyser.connect(audioContext.destination);

      // Add event listeners
      audio.addEventListener('ended', () => {
        setIsPlaying(false);
      });

      audio.addEventListener('error', (e) => {
        console.error('Audio error:', e);
        setIsPlaying(false);
      });

      audio.addEventListener('loadedmetadata', () => {
        setDuration(audio.duration);
      });

      audio.addEventListener('timeupdate', () => {
        if (!isSeeking) {
          setCurrentTime(audio.currentTime);
        }
      });

      // Play audio
      await audio.play();
      setIsPlaying(true);
    } catch (err) {
      console.error('Error playing audio:', err);
      setIsPlaying(false);
    }
  };

  const stopAudio = () => {
    if (audioElementRef.current) {
      audioElementRef.current.pause();
      audioElementRef.current.currentTime = 0;
    }
    setIsPlaying(false);
  };

  const handleSeek = (e) => {
    const newTime = parseFloat(e.target.value);
    setCurrentTime(newTime);
    if (audioElementRef.current) {
      audioElementRef.current.currentTime = newTime;
    }
  };

  const handleSeekStart = () => {
    setIsSeeking(true);
  };

  const handleSeekEnd = () => {
    setIsSeeking(false);
  };

  const formatTime = (time) => {
    if (isNaN(time)) return '0:00';
    const minutes = Math.floor(time / 60);
    const seconds = Math.floor(time % 60);
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
  };

  const handleClose = () => {
    stopAudio();
    if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
      audioContextRef.current.close().catch(err => {
        console.log('AudioContext already closed:', err);
      });
    }
    onClose();
  };

  useEffect(() => {
    // Auto-play when component mounts
    let mounted = true;
    
    const initAudio = async () => {
      if (mounted) {
        await playAudio();
      }
    };
    
    initAudio();

    return () => {
      mounted = false;
      stopAudio();
      if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
        audioContextRef.current.close().catch(err => {
          console.log('AudioContext already closed:', err);
        });
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [audioUrl]);

  // Always render as embedded panel
  return (
    <>
      <style>{sliderStyles}</style>
      <div style={{
      width: '100%',
      height: '100%',
      backgroundColor: 'rgba(0, 0, 0, 0.95)',
      borderRadius: '8px',
      display: 'flex',
      flexDirection: 'column',
      overflow: 'hidden',
      border: `1px solid rgba(${genreColor[0]}, ${genreColor[1]}, ${genreColor[2]}, 0.25)`,
      boxShadow: `0 4px 20px rgba(${genreColor[0]}, ${genreColor[1]}, ${genreColor[2]}, 0.1)`
    }}>
      {/* Compact Header */}
      <div style={{
        padding: '10px 12px',
        color: 'white',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        backgroundColor: 'rgba(0, 20, 40, 0.8)',
        borderBottom: `1px solid rgba(${genreColor[0]}, ${genreColor[1]}, ${genreColor[2]}, 0.2)`
      }}>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ 
            fontSize: '13px', 
            fontWeight: '600', 
            marginBottom: '2px',
            whiteSpace: 'nowrap',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            color: 'white'
          }}>
            {songName}
          </div>
          <div style={{ fontSize: '11px', color: '#888' }}>
            {genre}
          </div>
        </div>
        <div style={{ display: 'flex', gap: '6px', flexShrink: 0 }}>
          <button
            onClick={isPlaying ? stopAudio : playAudio}
            style={{
              width: '32px',
              height: '32px',
              backgroundColor: isPlaying ? 'rgba(255, 80, 80, 0.8)' : `rgba(${genreColor[0]}, ${genreColor[1]}, ${genreColor[2]}, 0.8)`,
              color: isPlaying ? 'white' : '#000',
              border: 'none',
              borderRadius: '50%',
              cursor: 'pointer',
              fontSize: '14px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              transition: 'all 0.2s'
            }}
          >
            {isPlaying ? '⏸' : '▶'}
          </button>
          <button
            onClick={handleClose}
            style={{
              width: '32px',
              height: '32px',
              backgroundColor: 'rgba(100, 100, 100, 0.5)',
              color: 'white',
              border: 'none',
              borderRadius: '50%',
              cursor: 'pointer',
              fontSize: '12px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              transition: 'all 0.2s'
            }}
          >
            ✕
          </button>
        </div>
      </div>

      {/* Time Slider Controls */}
      <div style={{
        padding: '8px 12px',
        backgroundColor: 'rgba(0, 20, 40, 0.8)',
        borderBottom: `1px solid rgba(${genreColor[0]}, ${genreColor[1]}, ${genreColor[2]}, 0.2)`,
        display: 'flex',
        alignItems: 'center',
        gap: '10px'
      }}>
        <span style={{ 
          fontSize: '11px', 
          color: '#888',
          minWidth: '35px',
          fontFamily: 'monospace'
        }}>
          {formatTime(currentTime)}
        </span>
        <input
          type="range"
          min="0"
          max={duration || 0}
          value={currentTime}
          onChange={handleSeek}
          onMouseDown={handleSeekStart}
          onMouseUp={handleSeekEnd}
          onTouchStart={handleSeekStart}
          onTouchEnd={handleSeekEnd}
          style={{
            flex: 1,
            height: '5px',
            borderRadius: '3px',
            outline: 'none',
            cursor: 'pointer',
            WebkitAppearance: 'none',
            appearance: 'none',
            background: `linear-gradient(to right, rgba(${genreColor[0]}, ${genreColor[1]}, ${genreColor[2]}, 0.9) 0%, rgba(${genreColor[0]}, ${genreColor[1]}, ${genreColor[2]}, 0.9) ${(currentTime / duration) * 100}%, rgba(255, 255, 255, 0.15) ${(currentTime / duration) * 100}%, rgba(255, 255, 255, 0.15) 100%)`,
            boxShadow: `0 0 8px rgba(${genreColor[0]}, ${genreColor[1]}, ${genreColor[2]}, 0.3)`
          }}
        />
        <span style={{ 
          fontSize: '11px', 
          color: '#888',
          minWidth: '35px',
          fontFamily: 'monospace'
        }}>
          {formatTime(duration)}
        </span>
      </div>

      {/* Visualization container */}
      <div
        ref={containerRef}
        style={{
          flex: 1,
          width: '100%',
          position: 'relative',
          minHeight: '80px'
        }}
      />
    </div>
    </>
  );
};

