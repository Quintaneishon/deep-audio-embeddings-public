import React, { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import ForceGraph3D from 'react-force-graph-3d';
import { ThreeAudioVisualizer } from './ThreeAudioVisualizer';

const STATIC_COLORS = [
  [255, 0, 0],
  [255, 105, 180],
  [138, 43, 226],
  [255, 215, 0],
  [0, 255, 255],
  [255, 140, 0],
  [34, 139, 34],
  [0, 0, 255],
  [128, 128, 128],
  [139, 69, 19]
];

const DEFAULT_COLOR = [200, 200, 200];
const HIGHLIGHT_COLOR = '#ffff00';
const HIGHLIGHT_NEIGHBOR_COLOR = '#ffffff';
const HIGHLIGHT_EDGE_COLOR = 'rgba(255,255,0,0.9)';
const DIM_NODE_OPACITY = 0.15;
const DIM_LINK_OPACITY = 0.03;

const generateRandomColor = () => [
  Math.floor(Math.random() * 256),
  Math.floor(Math.random() * 256),
  Math.floor(Math.random() * 256)
];

const rgbToHex = (r, g, b) =>
  '#' + [r, g, b].map(v => v.toString(16).padStart(2, '0')).join('');

const createGenreColorMap = async () => {
  try {
    const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://127.0.0.1:5000';
    const response = await fetch(`${API_BASE_URL}/tags`);
    const tags = await response.json();
    const colorMap = {};
    tags.forEach((tag, index) => {
      colorMap[tag.toLowerCase()] =
        index < STATIC_COLORS.length ? STATIC_COLORS[index] : generateRandomColor();
    });
    return colorMap;
  } catch (error) {
    console.error('Error fetching tags:', error);
    return {};
  }
};

const getColorForGenre = (genre, colorMap) => {
  if (!genre || !colorMap) return DEFAULT_COLOR;
  return colorMap[genre.toLowerCase()] || DEFAULT_COLOR;
};

export const ForceGraphVisualization = () => {
  const [graphData, setGraphData] = useState({ nodes: [], links: [] });
  const [genreColorMap, setGenreColorMap] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [showVisualizer, setShowVisualizer] = useState(false);
  const [currentAudioUrl, setCurrentAudioUrl] = useState(null);
  const [selectedPointInfo, setSelectedPointInfo] = useState(null);
  const [playingNodeId, setPlayingNodeId] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [highlightNodeId, setHighlightNodeId] = useState(null);
  const containerRef = useRef(null);
  const fgRef = useRef();

  // Build adjacency lookup: nodeId -> Set of neighbor nodeIds
  const neighborMap = useMemo(() => {
    const map = new Map();
    graphData.links.forEach(link => {
      const srcId = typeof link.source === 'object' ? link.source.id : link.source;
      const tgtId = typeof link.target === 'object' ? link.target.id : link.target;
      if (!map.has(srcId)) map.set(srcId, new Set());
      if (!map.has(tgtId)) map.set(tgtId, new Set());
      map.get(srcId).add(tgtId);
      map.get(tgtId).add(srcId);
    });
    return map;
  }, [graphData]);

  // Set of neighbor IDs for the highlighted node
  const highlightNeighbors = useMemo(() => {
    if (highlightNodeId === null) return new Set();
    return neighborMap.get(highlightNodeId) || new Set();
  }, [highlightNodeId, neighborMap]);

  // Search results
  const filteredSongs = useMemo(() => {
    if (!searchQuery.trim()) return [];
    const q = searchQuery.toLowerCase();
    return graphData.nodes
      .filter(n => n.name.toLowerCase().includes(q) || n.tag.toLowerCase().includes(q))
      .slice(0, 50);
  }, [searchQuery, graphData.nodes]);

  useEffect(() => {
    const load = async () => {
      try {
        setLoading(true);
        const colorMap = await createGenreColorMap();
        setGenreColorMap(colorMap);

        const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://127.0.0.1:5000';
        const res = await fetch(`${API_BASE_URL}/graph?red=vgg_contrastive&dataset=msd&k=5`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        console.log(`Graph loaded: ${data.nodes?.length} nodes, ${data.links?.length} links, k=${data.k}`);
        setGraphData({ nodes: data.nodes || [], links: data.links || [] });
        setError(null);
      } catch (err) {
        console.error('Error loading graph:', err);
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };
    load();
  }, []);

  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape') {
        if (showVisualizer) {
          setShowVisualizer(false);
          setCurrentAudioUrl(null);
          setSelectedPointInfo(null);
          setPlayingNodeId(null);
        }
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [showVisualizer]);

  const focusOnNode = useCallback((node) => {
    if (!fgRef.current || !node) return;
    const nx = node.x || 0;
    const ny = node.y || 0;
    const nz = node.z || 0;
    fgRef.current.cameraPosition(
      { x: nx, y: ny + 120, z: nz + 300 },
      { x: nx, y: ny, z: nz },
      2500
    );
  }, []);

  const handleSearchSelect = useCallback((searchNode) => {
    const liveNode = graphData.nodes.find(n => n.id === searchNode.id);
    if (!liveNode) return;
    setHighlightNodeId(liveNode.id);
    setSearchQuery('');
    focusOnNode(liveNode);
  }, [graphData.nodes, focusOnNode]);

  const handleNodeClick = useCallback((node) => {
    if (!node.audio) return;
    const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://127.0.0.1:5000';
    const audioUrl = `${API_BASE_URL}/audio/${node.audio}`;
    setCurrentAudioUrl(audioUrl);
    setSelectedPointInfo({
      name: node.name,
      tag: node.tag,
      position: [node.x || 0, node.y || 0, node.z || 0]
    });
    setPlayingNodeId(node.id);
    setShowVisualizer(true);
  }, []);

  const handleBackgroundClick = useCallback(() => {
    // background click does nothing — use Clear or new search to remove highlight
  }, []);

  const isHighlightActive = highlightNodeId !== null;

  const nodeColor = useCallback((node) => {
    // Playing state: highlight only the playing node, leave everything else normal
    if (playingNodeId !== null) {
      if (node.id === playingNodeId) return HIGHLIGHT_COLOR;
    }
    if (isHighlightActive) {
      if (node.id === highlightNodeId) return HIGHLIGHT_COLOR;
      if (highlightNeighbors.has(node.id)) return HIGHLIGHT_NEIGHBOR_COLOR;
      const c = getColorForGenre(node.tag, genreColorMap);
      return `rgba(${c[0]},${c[1]},${c[2]},${DIM_NODE_OPACITY})`;
    }
    const c = getColorForGenre(node.tag, genreColorMap);
    return rgbToHex(c[0], c[1], c[2]);
  }, [genreColorMap, isHighlightActive, highlightNodeId, highlightNeighbors, playingNodeId]);

  const nodeVal = useCallback((node) => {
    if (playingNodeId !== null) {
      return node.id === playingNodeId ? 6 : 1;
    }
    if (isHighlightActive) {
      if (node.id === highlightNodeId) return 6;
      if (highlightNeighbors.has(node.id)) return 2;
    }
    return 1;
  }, [isHighlightActive, highlightNodeId, highlightNeighbors, playingNodeId]);

  const linkColor = useCallback((link) => {
    const srcId = typeof link.source === 'object' ? link.source.id : link.source;
    const tgtId = typeof link.target === 'object' ? link.target.id : link.target;


    if (isHighlightActive) {
      const connected = srcId === highlightNodeId || tgtId === highlightNodeId;
      if (connected) return HIGHLIGHT_EDGE_COLOR;
      return `rgba(100,100,100,${DIM_LINK_OPACITY})`;
    }

    if (link.sameGenre) {
      const sourceNode = typeof link.source === 'object' ? link.source : null;
      if (sourceNode) {
        const c = getColorForGenre(sourceNode.tag, genreColorMap);
        return `rgba(${c[0]},${c[1]},${c[2]},0.6)`;
      }
      return 'rgba(255,255,255,0.4)';
    }
    return 'rgba(100,100,100,0.15)';
  }, [genreColorMap, isHighlightActive, highlightNodeId, playingNodeId]);

  const linkWidth = useCallback((link) => {
    if (isHighlightActive) {
      const srcId = typeof link.source === 'object' ? link.source.id : link.source;
      const tgtId = typeof link.target === 'object' ? link.target.id : link.target;
      if (srcId === highlightNodeId || tgtId === highlightNodeId) return 2;
      return 0.1;
    }
    if (link.sameGenre) return 0.8;
    return 0.3;
  }, [isHighlightActive, highlightNodeId, playingNodeId]);

  const nodeLabel = useCallback((node) => {
    const isTarget = node.id === highlightNodeId;
    const isNeighbor = highlightNeighbors.has(node.id);
    const badge = isTarget ? ' (selected)' : isNeighbor ? ' (neighbor)' : '';
    return `<div style="background:rgba(0,0,0,0.85);color:white;padding:6px 10px;border-radius:4px;font-size:13px">
      <b>${node.name}</b>${badge}<br/>Genre: ${node.tag}
    </div>`;
  }, [highlightNodeId, highlightNeighbors]);

  if (loading) {
    return (
      <div style={{
        width: '100%', height: '100%', display: 'flex',
        alignItems: 'center', justifyContent: 'center',
        backgroundColor: '#121212', color: 'white', flexDirection: 'column'
      }}>
        <div style={{
          width: 50, height: 50,
          border: '5px solid rgba(255,255,255,0.3)',
          borderTop: '5px solid white',
          borderRadius: '50%',
          animation: 'spin 1s linear infinite'
        }} />
        <p style={{ marginTop: 20 }}>Loading force graph...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div style={{
        width: '100%', height: '100%', display: 'flex',
        alignItems: 'center', justifyContent: 'center',
        backgroundColor: '#121212', color: 'white', flexDirection: 'column'
      }}>
        <div style={{ fontSize: 48, marginBottom: 16 }}>⚠</div>
        <h2>Error Loading Graph</h2>
        <p>{error}</p>
      </div>
    );
  }

  return (
    <div ref={containerRef} style={{ width: '100%', height: '100%', position: 'relative' }}>
      <ForceGraph3D
        ref={fgRef}
        graphData={graphData}
        nodeColor={nodeColor}
        nodeLabel={nodeLabel}
        nodeVal={nodeVal}
        nodeResolution={8}
        nodeOpacity={0.9}
        linkColor={linkColor}
        linkWidth={linkWidth}
        linkOpacity={0.6}
        onNodeClick={handleNodeClick}
        onBackgroundClick={handleBackgroundClick}
        backgroundColor="#121212"
        showNavInfo={false}
      />

      {/* Audio Visualizer Panel */}
      {showVisualizer && currentAudioUrl && selectedPointInfo && (
        <div style={{
          position: 'absolute',
          bottom: 10, right: 10,
          width: 340, height: 180,
          backgroundColor: 'rgba(0,0,0,0.9)',
          borderRadius: 8,
          zIndex: 10,
          overflow: 'hidden',
          boxShadow: '0 4px 20px rgba(0,0,0,0.5)'
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
              setPlayingNodeId(null);
            }}
          />
        </div>
      )}

      {/* Genre Legend */}
      {genreColorMap && (
        <div style={{
          position: 'absolute',
          top: 60, right: 10,
          backgroundColor: 'rgba(0,0,0,0.8)',
          color: 'white',
          padding: 12,
          borderRadius: 4,
          fontSize: 12,
          maxHeight: 'calc(100vh - 220px)',
          overflowY: 'auto',
          zIndex: 10
        }}>
          <div style={{ fontWeight: 'bold', marginBottom: 8 }}>Genres</div>
          {Object.entries(genreColorMap).map(([genre, color]) => (
            <div key={genre} style={{ display: 'flex', alignItems: 'center', marginBottom: 4 }}>
              <div style={{
                width: 12, height: 12,
                backgroundColor: `rgb(${color[0]},${color[1]},${color[2]})`,
                marginRight: 8,
                borderRadius: 2
              }} />
              <span>{genre}</span>
            </div>
          ))}
        </div>
      )}

      {/* Search Panel */}
      <div style={{
        position: 'absolute',
        bottom: 10, left: 10,
        backgroundColor: 'rgba(0,0,0,0.9)',
        color: 'white',
        padding: 12,
        borderRadius: 8,
        fontSize: 13,
        zIndex: 10,
        width: 300,
        maxHeight: 400,
        display: 'flex',
        flexDirection: 'column',
        boxShadow: '0 4px 12px rgba(0,0,0,0.5)'
      }}>
        <div style={{ marginBottom: 8, fontWeight: 'bold', fontSize: 14 }}>
          Search Songs
        </div>
        <input
          type="text"
          placeholder="Type song name or genre..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          style={{
            padding: '8px 12px',
            backgroundColor: 'rgba(255,255,255,0.1)',
            border: '1px solid rgba(255,255,255,0.3)',
            borderRadius: 4,
            color: 'white',
            fontSize: 13,
            outline: 'none',
            marginBottom: 8
          }}
        />

        {searchQuery && (
          <div style={{ fontSize: 11, opacity: 0.7, marginBottom: 8 }}>
            {filteredSongs.length} {filteredSongs.length === 1 ? 'result' : 'results'} found
          </div>
        )}

        {searchQuery && filteredSongs.length > 0 && (
          <div style={{ overflowY: 'auto', maxHeight: 260, marginTop: 4 }}>
            {filteredSongs.map((node) => {
              const c = getColorForGenre(node.tag, genreColorMap);
              return (
                <div
                  key={node.id}
                  onClick={() => handleSearchSelect(node)}
                  style={{
                    padding: 8, marginBottom: 4,
                    backgroundColor: 'rgba(255,255,255,0.05)',
                    borderRadius: 4, cursor: 'pointer',
                    transition: 'all 0.2s',
                    border: '1px solid transparent'
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.backgroundColor = 'rgba(255,255,255,0.15)';
                    e.currentTarget.style.borderColor = 'rgba(255,255,255,0.3)';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.backgroundColor = 'rgba(255,255,255,0.05)';
                    e.currentTarget.style.borderColor = 'transparent';
                  }}
                >
                  <div style={{
                    fontWeight: 500, marginBottom: 2,
                    overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap'
                  }}>
                    {node.name}
                  </div>
                  <div style={{
                    fontSize: 11, opacity: 0.7,
                    display: 'flex', alignItems: 'center', gap: 8
                  }}>
                    <span style={{
                      display: 'inline-block', width: 8, height: 8, minWidth: 8,
                      backgroundColor: `rgb(${c[0]},${c[1]},${c[2]})`,
                      borderRadius: '50%'
                    }} />
                    <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {node.tag}
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {searchQuery && filteredSongs.length === 0 && (
          <div style={{ padding: 16, textAlign: 'center', opacity: 0.6, fontSize: 12 }}>
            No songs found matching "{searchQuery}"
          </div>
        )}

        {highlightNodeId !== null && !searchQuery && (
          <div style={{
            marginTop: 4, padding: '8px 10px',
            backgroundColor: 'rgba(255,255,0,0.1)',
            borderRadius: 4, fontSize: 11,
            border: '1px solid rgba(255,255,0,0.3)'
          }}>
            Showing {highlightNeighbors.size} connections.
            <span
              onClick={() => setHighlightNodeId(null)}
              style={{ marginLeft: 8, cursor: 'pointer', textDecoration: 'underline', opacity: 0.8 }}
            >
              Clear
            </span>
          </div>
        )}
      </div>
    </div>
  );
};
