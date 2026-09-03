import { useEffect, useState, useRef } from 'react';
import { Sidebar } from '../components/Sidebar';
import { VisualizationToolbar } from '../components/VisualizationToolbar';
import { ScatterPlot } from '../components/ScatterPlot';
import { SongInfoPopup } from '../components/SongInfoPopup';
import { useMakeRequest } from '../hooks/useMakeRequest';

// Fixed defaults — not exposed in UI
const TIPO_GRAFICA = 'umap';

export const HomePage = () => {
  // Sidebar state
  const [listaCanciones, setListaCanciones] = useState([]);
  const [selectedSongs, setSelectedSongs] = useState([]);
  const [highlightActive, setHighlightActive] = useState(false);

  // Toolbar state
  const [modelsConfig, setModelsConfig] = useState({});
  const [architecture, setArchitecture] = useState('musicnn');
  const [dataset, setDataset] = useState('msd');
  const [dimensiones, setDimensiones] = useState(2);

  // Embedding state
  const [embeddings, setEmbeddings] = useState([]);
  const [hasEmbeddings, setHasEmbeddings] = useState(false);
  const [isLoadingEmbeddings, setIsLoadingEmbeddings] = useState(false);

  // Cross-validation comparison state
  const [showCompareModal, setShowCompareModal] = useState(false);
  const [isComparing, setIsComparing] = useState(false);
  const [compareResult, setCompareResult] = useState(null);
  const [compareError, setCompareError] = useState(null);
  const [compareTags, setCompareTags] = useState({ tag1: '', tag2: '' });

  const { obtenerAudios, obtenerEmbeddings, obtenerConfig, compararCanciones } = useMakeRequest();

  // Load songs and config on mount
  useEffect(() => {
    obtenerAudios().then(data => setListaCanciones(data || []));
    obtenerConfig()
      .then(({ models }) => {
        setModelsConfig(models);
        const archs = Object.keys(models);
        if (archs.length > 0) {
          const firstArch = archs[0];
          setArchitecture(firstArch);
          const firstDs = models[firstArch]?.datasets?.[0]?.value;
          if (firstDs) setDataset(firstDs);
        }
      })
      .catch(err => console.error('Failed to load config:', err));
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Auto-reload embeddings whenever toolbar changes
  const isFirstRender = useRef(true);
  useEffect(() => {
    if (isFirstRender.current) {
      isFirstRender.current = false;
      return;
    }
    if (!architecture || !dataset) return;

    setIsLoadingEmbeddings(true);
    setEmbeddings([]);
    setHasEmbeddings(false);
    setHighlightActive(false);

    obtenerEmbeddings(architecture, dataset, TIPO_GRAFICA, dimensiones)
      .then(resp => {
        setEmbeddings(resp.data || []);
        setHasEmbeddings(true);
      })
      .catch(err => console.error('Failed to load embeddings:', err))
      .finally(() => setIsLoadingEmbeddings(false));
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [architecture, dataset, dimensiones]);

  // Highlight the 2 selected songs in the plot (frontend only)
  const handleHighlight = () => {
    if (selectedSongs.length !== 2) return;
    setHighlightActive(true);
  };

  const handleSongsChange = (songs) => {
    setSelectedSongs(songs);
    if (highlightActive) setHighlightActive(false);
  };

  // Cross-validation: called when user clicks two points in the plot
  const handlePairSelect = (song1, song2) => {
    // Look up genre tags from the loaded embeddings
    const emb1 = embeddings.find(e => e.name === song1);
    const emb2 = embeddings.find(e => e.name === song2);
    setCompareTags({ tag1: emb1?.tag || '', tag2: emb2?.tag || '' });

    setCompareResult(null);
    setCompareError(null);
    setIsComparing(true);
    setShowCompareModal(true);

    compararCanciones(song1, song2, architecture, dataset)
      .then(result => {
        if (result.error) {
          setCompareError(result.error);
        } else {
          setCompareResult(result);
        }
      })
      .catch(err => setCompareError(err.message || 'Request failed'))
      .finally(() => setIsComparing(false));
  };

  return (
    <div className="flex h-screen bg-gray-50 overflow-hidden">
      <Sidebar
        listaCanciones={listaCanciones}
        selectedSongs={selectedSongs}
        onSongsChange={handleSongsChange}
        isLoading={false}
        progress={0}
        onCompare={handleHighlight}
      />

      <div className="flex-1 flex flex-col overflow-hidden">
        <VisualizationToolbar
          modelsConfig={modelsConfig}
          architecture={architecture}
          onArchitectureChange={(val) => {
            setArchitecture(val);
            const firstDs = modelsConfig[val]?.datasets?.[0]?.value;
            if (firstDs) setDataset(firstDs);
          }}
          dataset={dataset}
          onDatasetChange={setDataset}
          dimensiones={dimensiones}
          onDimensionesChange={setDimensiones}
        />

        {isLoadingEmbeddings && (
          <div className="px-6 py-2 bg-indigo-50 border-b border-indigo-100 text-indigo-600 text-xs font-mono">
            Loading embeddings...
          </div>
        )}

        <div className="flex-1 flex flex-col relative overflow-hidden">
          <ScatterPlot
            embeddings={embeddings}
            dimensiones={dimensiones}
            selectedSongs={selectedSongs}
            highlightActive={highlightActive}
            hasEmbeddings={hasEmbeddings}
            onPairSelect={handlePairSelect}
          />
        </div>
      </div>

      <SongInfoPopup
        isOpen={showCompareModal}
        onClose={() => setShowCompareModal(false)}
        compareResult={compareResult}
        isLoading={isComparing}
        error={compareError}
        tag1={compareTags.tag1}
        tag2={compareTags.tag2}
      />
    </div>
  );
};
