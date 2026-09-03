export const useMakeRequest = () => {
    const URL = process.env.REACT_APP_API_BASE_URL || "http://127.0.0.1:5000";

    const obtenerAudios = async () => {
        const resp = await fetch(URL + '/audios');
        return await resp.json();
    };

    const obtenerTags = async () => {
        const resp = await fetch(URL + '/tags');
        return await resp.json();
    };

    const obtenerEmbeddings = async (red, dataset, metodo, dimensions = 2) => {
        const url = `${URL}/embeddings?red=${red.toLowerCase()}&dataset=${dataset.toLowerCase()}&metodo=${metodo.toLowerCase()}&dimensions=${dimensions}`;
        const resp = await fetch(url);
        return await resp.json(); // { data: [...] }
    };

    const obtenerConfig = async () => {
        const resp = await fetch(URL + '/config');
        return await resp.json();
    };

    const compararCanciones = async (song1, song2, model, dataset) => {
        const url = `${URL}/compare?song1=${encodeURIComponent(song1)}&song2=${encodeURIComponent(song2)}&model=${encodeURIComponent(model)}&dataset=${encodeURIComponent(dataset)}`;
        const resp = await fetch(url);
        return await resp.json();
    };

    return {
        obtenerEmbeddings,
        obtenerAudios,
        obtenerTags,
        obtenerConfig,
        compararCanciones,
    };
};
