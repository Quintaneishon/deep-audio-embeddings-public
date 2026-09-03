import { useState, useEffect } from 'react';
import Plot from 'react-plotly.js';
import { Music } from 'lucide-react';

interface EmbeddingPoint {
  coords: number[];
  name: string;
  tag: string;
  audio: string;
}

interface PendingPoint {
  name: string;
  tag: string;
}

interface ScatterPlotProps {
  embeddings: EmbeddingPoint[];
  dimensiones: number;
  selectedSongs: string[];
  highlightActive: boolean;
  hasEmbeddings: boolean;
  onPairSelect?: (song1: string, song2: string) => void;
}

const COLOR_PALETTE = [
  '#6366f1', '#ef4444', '#3b82f6', '#ec4899', '#8b5cf6',
  '#f59e0b', '#10b981', '#06b6d4', '#dc2626', '#f97316', '#84cc16',
];

export function ScatterPlot({
  embeddings,
  dimensiones,
  selectedSongs,
  highlightActive,
  hasEmbeddings,
  onPairSelect,
}: ScatterPlotProps) {
  const [plotData, setPlotData] = useState<any[]>([]);
  const [plotLayout, setPlotLayout] = useState<any>({});
  const [pendingPoint, setPendingPoint] = useState<PendingPoint | null>(null);

  // Always group by tag (fixed default)
  const agruparPor = 'tag';

  // Clear pending selection when embeddings change
  useEffect(() => {
    setPendingPoint(null);
  }, [embeddings]);

  useEffect(() => {
    if (embeddings.length === 0) {
      setPlotData([]);
      return;
    }

    // Group embeddings by tag
    const grupos: Record<string, EmbeddingPoint[]> = {};
    embeddings.forEach((p) => {
      const key = p[agruparPor as keyof EmbeddingPoint] as string;
      if (!grupos[key]) grupos[key] = [];
      grupos[key].push(p);
    });

    const type = dimensiones === 3 ? 'scatter3d' : 'scatter';

    const traces = Object.keys(grupos).map((groupName, idx) => {
      const points = grupos[groupName];
      const color = COLOR_PALETTE[idx % COLOR_PALETTE.length];

      const isPending = (p: EmbeddingPoint) => p.name === pendingPoint?.name;
      const isHighlighted = (p: EmbeddingPoint) => selectedSongs.includes(p.name) && highlightActive;

      const getOpacity = (p: EmbeddingPoint) => {
        if (isPending(p)) return 1;
        if (isHighlighted(p)) return 1;
        if (pendingPoint) return 0.3;
        if (highlightActive) return 0.25;
        return 0.8;
      };

      const trace: any = {
        x: points.map((p) => p.coords[0]),
        y: points.map((p) => p.coords[1]),
        mode: 'markers',
        type,
        name: groupName,
        text: points.map((p) => `Name: ${p.name}<br>Genre: ${p.tag}`),
        hovertemplate: '%{text}<extra></extra>',
        marker: {
          size: points.map((p) =>
            isPending(p) ? 16 : isHighlighted(p) ? 16 : 8
          ),
          color: points.map((p) =>
            isPending(p) ? '#f59e0b' : isHighlighted(p) ? '#ffffff' : color
          ),
          opacity: points.map((p) => getOpacity(p)),
          line: {
            color: points.map((p) =>
              isPending(p) ? '#d97706' : isHighlighted(p) ? '#6366f1' : 'transparent'
            ),
            width: points.map((p) =>
              isPending(p) ? 2 : isHighlighted(p) ? 3 : 0
            ),
          },
        },
        customdata: points.map((p) => ({ name: p.name, audio: p.audio, tag: p.tag })),
      };

      if (dimensiones === 3) {
        trace.z = points.map((p) => p.coords[2] ?? 0);
      }

      return trace;
    });

    setPlotData(traces);

    // Build annotations when highlighting
    const annotations: any[] = [];

    if (highlightActive && selectedSongs.length > 0) {
      const hits = selectedSongs
        .map((name) => embeddings.find((e) => e.name === name))
        .filter(Boolean) as EmbeddingPoint[];

      hits.forEach((emb) => {
        annotations.push({
          x: emb.coords[0],
          y: emb.coords[1],
          text: emb.name,
          showarrow: true,
          arrowhead: 2,
          arrowsize: 1,
          arrowwidth: 1.5,
          arrowcolor: '#6366f1',
          ax: 0,
          ay: -40,
          font: { color: '#6366f1', size: 10, family: 'monospace' },
          bgcolor: '#eef2ff',
          bordercolor: '#6366f1',
          borderwidth: 1,
          borderpad: 4,
        });
      });
    }

    setPlotLayout({
      autosize: true,
      uirevision: 'static',
      paper_bgcolor: '#ffffff',
      plot_bgcolor: '#f9fafb',
      font: { family: 'monospace', color: '#6b7280' },
      xaxis: { gridcolor: '#f3f4f6', zerolinecolor: '#e5e7eb', showgrid: true },
      yaxis: { gridcolor: '#f3f4f6', zerolinecolor: '#e5e7eb', showgrid: true },
      legend: {
        bgcolor: '#f9fafb',
        bordercolor: '#e5e7eb',
        borderwidth: 1,
        font: { size: 11 },
      },
      annotations,
      margin: { l: 40, r: 40, t: 20, b: 40 },
      hovermode: 'closest',
    });
  }, [embeddings, dimensiones, selectedSongs, highlightActive, pendingPoint]);

  const handlePlotClick = (event: any) => {
    try {
      const point = event.points[0];
      if (!point) return;
      const { name: songName, tag } = point.customdata;
      if (!songName) return;

      if (pendingPoint === null) {
        // First click: select as pending
        setPendingPoint({ name: songName, tag: tag || '' });
      } else if (pendingPoint.name === songName) {
        // Same point clicked: deselect
        setPendingPoint(null);
      } else {
        // Second click on different point: trigger comparison
        onPairSelect?.(pendingPoint.name, songName);
        setPendingPoint(null);
      }
    } catch (err) {
      console.error('Error handling plot click:', err);
    }
  };

  if (!hasEmbeddings) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center text-center px-8">
        <div className="w-16 h-16 rounded-full bg-gray-100 flex items-center justify-center mb-4">
          <Music className="w-8 h-8 text-gray-400" />
        </div>
        <h3 className="text-gray-900 font-mono mb-2 text-lg font-medium">No Embeddings Loaded</h3>
        <p className="text-gray-500 text-sm font-mono max-w-md">
          Change the toolbar settings above to load and visualize audio embeddings.
        </p>
      </div>
    );
  }

  return (
    <div className="flex-1 bg-white relative">
      {pendingPoint ? (
        <div className="absolute top-2 left-2 bg-amber-50 border border-amber-300 text-amber-800 text-xs font-mono px-3 py-2 rounded shadow z-10 max-w-xs">
          <div className="font-semibold truncate">{pendingPoint.name}</div>
          {pendingPoint.tag && (
            <div className="text-amber-600 mt-0.5">{pendingPoint.tag}</div>
          )}
          <div className="text-amber-500 mt-1">Click another point to compare</div>
        </div>
      ) : (
        <div className="absolute top-2 left-2 text-gray-400 text-xs font-mono px-3 py-1 z-10">
          Click a point to select it
        </div>
      )}
      <Plot
        data={plotData}
        layout={plotLayout}
        config={{
          displayModeBar: true,
          displaylogo: false,
          modeBarButtonsToRemove: ['lasso2d', 'select2d'],
          responsive: true,
        }}
        style={{ width: '100%', height: '100%' }}
        useResizeHandler={true}
        onClick={handlePlotClick}
      />
    </div>
  );
}
