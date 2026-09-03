import { useEffect, useState } from 'react';
import Form from 'react-bootstrap/Form';
import { useMakeRequest } from '../hooks/useMakeRequest';
import '../styles/selectorgrafico.css';

export const SelectorGrafico = ({ arquitectura, setArquitectura, dataset, setDataset, tipoGrafica, setTipoGrafica, agruparPor, setAgruparPor, dimensiones, setDimensiones, renderEngine, setRenderEngine }) => {
    const [modelsConfig, setModelsConfig] = useState({});
    const { obtenerConfig } = useMakeRequest();

    useEffect(() => {
        obtenerConfig()
            .then(({ models }) => {
                setModelsConfig(models);
                const archs = Object.keys(models);
                if (archs.length > 0 && !models[arquitectura]) {
                    const firstArch = archs[0];
                    setArquitectura(firstArch);
                    setDataset(models[firstArch].datasets[0].value);
                }
            })
            .catch((err) => console.error('Failed to load config:', err));
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    const availableDatasets = modelsConfig[arquitectura]?.datasets ?? [];

    const handleArquitecturaChange = (e) => {
        const arch = e.target.value;
        setArquitectura(arch);
        const datasets = modelsConfig[arch]?.datasets ?? [];
        if (datasets.length > 0) setDataset(datasets[0].value);
    };

    return (
        <div className='selectorGrafico'>
            <div>
                <p>Arquitectura</p>
                <Form.Select
                    aria-label="Arquitectura"
                    value={arquitectura}
                    onChange={handleArquitecturaChange}
                >
                    {Object.entries(modelsConfig).map(([arch, { label }]) => (
                        <option key={arch} value={arch}>{label}</option>
                    ))}
                </Form.Select>
            </div>

            <div>
                <p>Dataset</p>
                <Form.Select
                    aria-label="Dataset"
                    value={dataset}
                    onChange={(e) => setDataset(e.target.value)}
                >
                    {availableDatasets.map(({ value, label }) => (
                        <option key={value} value={value}>{label}</option>
                    ))}
                </Form.Select>
            </div>

            <div>
                <p>Tipo Grafica</p>
                <Form.Select
                    aria-label="Grafica"
                    value={tipoGrafica}
                    onChange={(e) => setTipoGrafica(e.target.value)}
                >
                    <option value="tsne">t-SNE</option>
                    <option value="umap">UMAP</option>
                </Form.Select>
            </div>

            <div>
                <p>Agrupar por</p>
                <Form.Select
                    aria-label="Agrupar por"
                    value={agruparPor}
                    onChange={(e) => setAgruparPor(e.target.value)}
                >
                    <option value="tag">Tag</option>
                    <option value="name">Cancion</option>
                </Form.Select>
            </div>

            <div>
                <p>Dimension</p>
                <Form.Select
                    aria-label="Dimension"
                    value={String(dimensiones)}
                    onChange={(e) => setDimensiones(parseInt(e.target.value))}
                >
                    <option value="2">2D</option>
                    <option value="3">3D</option>
                </Form.Select>
            </div>

            <div>
                <p>Render Engine</p>
                <Form.Select
                    aria-label="Render Engine"
                    value={renderEngine}
                    onChange={(e) => setRenderEngine(e.target.value)}
                >
                    <option value="plotly">Plotly</option>
                    <option value="deckgl">Deck.gl (Point Cloud)</option>
                </Form.Select>
            </div>
        </div>
    );
};
