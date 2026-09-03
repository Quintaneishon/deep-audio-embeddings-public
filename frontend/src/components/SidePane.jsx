import Form from 'react-bootstrap/Form';
import '../styles/sidepane.css';
import Button from 'react-bootstrap/Button';
import ProgressBar from 'react-bootstrap/ProgressBar';

export const SidePane = ({ listaTags, tags, setTags, canciones, setCanciones, listaCanciones, cargarDatos, progreso }) => {

    const handleTagsChange = (e) => {
        const selectedValues = Array.from(e.target.selectedOptions, (option) => option.value);
       // setTags(selectedValues);
        console.log('Tags seleccionados:', selectedValues);
    };

    const handleCancionesChange = (e) => {
        const selectedValues = Array.from(e.target.selectedOptions, (option) => option.value);
        setCanciones(selectedValues);
        console.log('Canciones seleccionados:', selectedValues);
    };
    return (
        <div className='sidepane'>
            <div className='item'>
                <p>Tags</p>
                <Form.Select
                    multiple
                    aria-label="Tags"
                    value={tags}
                    onChange={handleTagsChange}
                >
                    {listaTags.map((tag, idx) => <option value={tag} key={idx}>{tag}</option>)}
                </Form.Select>
                {/* <Button style={{ marginTop: '10px' }} onClick={() => setTags([])}>Limpiar</Button> */}
            </div>

            <div className='item'>
                <p>Canciones</p>
                <Form.Select
                    multiple
                    aria-label="Cancion"
                    value={canciones}
                    onChange={handleCancionesChange}
                >
                    {listaCanciones.map((lCna, idx) => (<option value={lCna.id} key={idx}>{lCna.filename}</option>))}
                </Form.Select>
            </div>
            {progreso !== 0 && progreso !== 100 &&
                <div>
                    <p>Cargando...</p>
                    <ProgressBar now={progreso} />
                </div>
            }
            {(progreso === 0 || progreso === 100) && <Button style={{ marginTop: '10px' }} onClick={cargarDatos}>Obtener Embeddings y Taggrams</Button>}
        </div>
    );
};
