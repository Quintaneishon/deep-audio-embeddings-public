import { useEffect, useState, useRef } from "react";
import { SelectorGrafico } from "./SelectorGrafico";
import Plot from "react-plotly.js";
import { DeckGLVisualization } from "./DeckGLVisualization";
import "../styles/grafica.css";

export const Grafica = ({
    arquitectura,
    setArquitectura,
    dataset,
    setDataset,
    data,
    layout,
    tipoGrafica,
    setTipoGrafica,
    izq,
    embeddings,
    visualizar,
    setVisualizar,
    agruparPor,
    setAgruparPor,
    dimensiones,
    setDimensiones
}) => {
    const [plotData, setPlotData] = useState([]);
    const audioRef = useRef(null);
    const [currentPlayingPoint, setCurrentPlayingPoint] = useState(null);
    const [graficaCargando, setGraficaCargando] = useState(false);
    const [renderEngine, setRenderEngine] = useState("plotly");

    useEffect(() => {
        if (!data || data.length === 0 || !embeddings || embeddings.length === 0) {
            setPlotData(data);
            return;
        }

        setGraficaCargando(true);
        // Add hover text with name and genre information
        const updatedData = data.map(trace => {
            // For each point in this trace, find the corresponding embedding
            const hoverTexts = trace.x.map((x, i) => {
                const y = trace.y[i];
                const embedding = embeddings.find(e =>
                    Math.abs(e.coords[0] - x) < 0.0001 &&
                    Math.abs(e.coords[1] - y) < 0.0001
                );

                if (embedding) {
                    return `Name: ${embedding.name}<br>Genre: ${embedding.tag}`;
                }
                return `x: ${x.toFixed(2)}<br>y: ${y.toFixed(2)}`;
            });

            return {
                ...trace,
                text: hoverTexts,
                hovertemplate: '%{text}<extra></extra>'
            };
        });
        setPlotData(updatedData);
    }, [data, embeddings]);

    useEffect(() => {
        const handleKeyDown = (e) => {
            if (e.key === "Escape") {
                if (audioRef.current) {
                    audioRef.current.pause();
                    audioRef.current.currentTime = 0;
                }
                setCurrentPlayingPoint(null);
                console.log("Audio detenido con ESC");
            }
        };

        window.addEventListener("keydown", handleKeyDown);

        return () => {
            window.removeEventListener("keydown", handleKeyDown);
        };
    }, []);


    return (
        <div
            className="divGrafico"
            style={izq ? { borderRight: "1px solid black" } : {}}
        >
            <SelectorGrafico
                arquitectura={arquitectura}
                setArquitectura={setArquitectura}
                dataset={dataset}
                setDataset={setDataset}
                tipoGrafica={tipoGrafica}
                setTipoGrafica={setTipoGrafica}
                visualizar={visualizar}
                setVisualizar={setVisualizar}
                agruparPor={agruparPor}
                setAgruparPor={setAgruparPor}
                dimensiones={dimensiones}
                setDimensiones={setDimensiones}
                renderEngine={renderEngine}
                setRenderEngine={setRenderEngine}
            />
            {graficaCargando && <p>Cargando grafica...</p>}
            <div className="grafica">
                {renderEngine === "plotly" ? (
                    <Plot
                        data={plotData}
                        layout={layout}
                        style={{ width: '100%', height: '100%' }}
                        useResizeHandler={true}
                        config={{ responsive: true }}
                        onInitialized={() => {
                            setGraficaCargando(false);
                        }}
                        onUpdate={() => {
                            setGraficaCargando(false);
                        }}
                        onError={() => {
                            setGraficaCargando(false);
                        }}
                        onClick={(e) => {
                            try {
                                const x = e.points[0].x;
                                const y = e.points[0].y;
                                const z = e.points[0].z;
                                console.log(x, y, z);

                                // Find the embedding that matches the clicked point
                                const embedding = embeddings.find((emb) =>
                                    Math.abs(emb.coords[0] - x) < 0.0001 &&
                                    Math.abs(emb.coords[1] - y) < 0.0001 &&
                                    (dimensiones === 3 ? Math.abs(emb.coords[2] - z) < 0.0001 : true)
                                );
                                console.log(embedding)
                                if (embedding && embedding.audio) {
                                    // Create a unique identifier for the clicked point
                                    const pointId = `${embedding.name}_${x}_${y}${dimensiones === 3 ? `_${z}` : ''}`;

                                    // Check if clicking the same point that's currently playing
                                    if (currentPlayingPoint === pointId && audioRef.current) {
                                        // Stop the audio
                                        audioRef.current.pause();
                                        audioRef.current.currentTime = 0;
                                        setCurrentPlayingPoint(null);
                                        console.log(`Stopped: ${embedding.name}`);
                                        return;
                                    }

                                    // Stop current audio if playing a different track
                                    if (audioRef.current) {
                                        audioRef.current.pause();
                                        audioRef.current.currentTime = 0;
                                    }

                                    // Construct the audio URL
                                    const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://127.0.0.1:5000';
                                    const audioUrl = `${API_BASE_URL}/audio/${embedding.audio}`;
                                    console.log("Playing audio from:", audioUrl);

                                    // Play the new audio
                                    audioRef.current = new Audio(audioUrl);

                                    // Add event listener for when audio ends
                                    audioRef.current.addEventListener('ended', () => {
                                        setCurrentPlayingPoint(null);
                                    });

                                    audioRef.current.play().catch(err => {
                                        console.error("Error playing audio:", err);
                                        alert(`Error playing: ${embedding.name}`);
                                        setCurrentPlayingPoint(null);
                                    });

                                    setCurrentPlayingPoint(pointId);
                                    console.log(`Playing: ${embedding.name} (${embedding.tag})`);
                                }
                            } catch (err) {
                                console.error("Error handling click:", err);
                            }
                        }}
                    />
                ) : (
                    <DeckGLVisualization
                        embeddings={embeddings}
                        dimensiones={dimensiones}
                        agruparPor={agruparPor}
                    />
                )}
            </div>
        </div>
    );
};
