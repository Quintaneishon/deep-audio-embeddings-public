import { useState } from 'react';
import { EvalGame } from '../components/EvalGame';

export function EvalPage() {
  const [phase, setPhase] = useState('intro'); // 'intro' | 'playing' | 'done'

  const handleStart = () => {
    localStorage.removeItem('soundmatch_session_id');
    localStorage.removeItem('soundmatch_shown');
    setPhase('playing');
  };

  if (phase === 'done') {
    return (
      <div className="min-h-screen bg-gray-950 flex items-center justify-center">
        <div className="text-center space-y-4 max-w-md px-8">
          <div className="text-4xl">🎧</div>
          <h1 className="text-2xl font-mono text-white">Gracias</h1>
          <p className="text-gray-400 font-mono text-sm leading-relaxed">
            Completaste las 10 rondas. Tus respuestas han sido registradas y
            ayudarán a evaluar qué tan bien los modelos de IA capturan la
            percepción humana de similitud musical.
          </p>
          <button
            onClick={handleStart}
            className="mt-4 px-6 py-3 bg-indigo-600 hover:bg-indigo-500 text-white
                       font-mono text-sm rounded-lg transition-colors"
          >
            Iniciar nueva sesión
          </button>
        </div>
      </div>
    );
  }

  if (phase === 'playing') {
    return (
      <div className="min-h-screen bg-gray-950 py-10 px-4">
        <EvalGame onFinish={() => setPhase('done')} />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-950 flex items-center justify-center">
      <div className="text-center space-y-6 max-w-lg px-8">
        <h1 className="text-3xl font-mono text-white">SoundMatch</h1>

        <div className="text-left bg-gray-900 border border-indigo-900 rounded-lg p-4">
          <p className="text-gray-300 font-mono text-sm leading-relaxed">
            Este experimento forma parte de una investigación de maestría en la UNAM
            sobre representaciones de audio y similitud musical percibida.
          </p>
        </div>

        <p className="text-gray-400 font-mono text-sm leading-relaxed">
          Escucharás 10 rondas con una canción de referencia y dos opciones.
          En cada ronda elige cuál opción suena{' '}
          <span className="text-indigo-300 font-semibold">más parecida</span> a la referencia.
          No hay respuestas correctas — solo sigue tu intuición.
        </p>
        <p className="text-gray-500 font-mono text-xs">
          Cada canción dura 30 segundos. La evaluación toma aprox. 5 minutos.
        </p>
        <button
          onClick={handleStart}
          className="px-8 py-3 bg-indigo-600 hover:bg-indigo-500 text-white
                     font-mono rounded-lg transition-colors shadow-lg shadow-indigo-900/40"
        >
          Comenzar
        </button>
      </div>
    </div>
  );
}
