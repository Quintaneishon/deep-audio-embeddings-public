import { useState, useEffect, useCallback } from 'react';
import { AudioCard } from './AudioCard';

const API_BASE    = process.env.REACT_APP_API_BASE_URL || 'http://127.0.0.1:5000';
const TOTAL_ROUNDS = 10;

function getOrCreateSessionId() {
  let sid = localStorage.getItem('soundmatch_session_id');
  if (!sid) {
    sid = crypto.randomUUID();
    localStorage.setItem('soundmatch_session_id', sid);
  }
  return sid;
}

function loadShownIds() {
  try {
    const stored = JSON.parse(localStorage.getItem('soundmatch_shown') || '[]');
    if (!Array.isArray(stored)) return [];
    return stored.filter(id => Number.isInteger(id) && id > 0).slice(0, TOTAL_ROUNDS);
  }
  catch { return []; }
}

function saveShownIds(ids) {
  localStorage.setItem('soundmatch_shown', JSON.stringify(ids));
}

export function EvalGame({ onFinish }) {
  const [sessionId]    = useState(getOrCreateSessionId);
  const [shownIds,     setShownIds]     = useState(loadShownIds);
  const [triplet,      setTriplet]      = useState(null);
  const [roundIndex,   setRoundIndex]   = useState(() => loadShownIds().length);
  const [loading,      setLoading]      = useState(false);
  const [error,        setError]        = useState(null);
  const [roundStartMs, setRoundStartMs] = useState(null);

  const fetchNext = useCallback((currentShownIds) => {
    setLoading(true);
    setError(null);
    setTriplet(null);
    const param = currentShownIds.join(',');
    fetch(`${API_BASE}/eval/triplet?shown=${param}`)
      .then(async r => {
        const data = await r.json().catch(() => ({}));
        if (!r.ok) throw new Error(data.error || 'No se pudo cargar la ronda.');
        return data;
      })
      .then(data => {
        if (!data.triplet) { onFinish(); return; }
        setTriplet(data.triplet);
        setRoundStartMs(Date.now());
      })
      .catch(() => setError('No se pudo cargar la ronda. Verifica la conexión.'))
      .finally(() => setLoading(false));
  }, [onFinish]);

  // Load first round on mount
  useEffect(() => {
    if (roundIndex >= TOTAL_ROUNDS) { onFinish(); return; }
    fetchNext(shownIds);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleChoice = async (choice) => {
    if (!triplet || loading) return;

    const responseTimeMs = Date.now() - roundStartMs;
    const newShownIds    = [...shownIds, triplet.id];
    const newRound       = roundIndex + 1;

    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE}/eval/response`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id:       sessionId,
          triplet_id:       triplet.id,
          choice,
          response_time_ms: responseTimeMs,
          respondent_type:  'public',
        }),
      });
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        throw new Error(body.error || 'No se pudo guardar la respuesta.');
      }
    } catch {
      setError('No se pudo guardar la respuesta. Intenta de nuevo.');
      setLoading(false);
      return;
    }

    setShownIds(newShownIds);
    saveShownIds(newShownIds);
    setRoundIndex(newRound);

    if (newRound >= TOTAL_ROUNDS) {
      setLoading(false);
      onFinish();
    } else {
      fetchNext(newShownIds);
    }
  };

  // ── Render ──────────────────────────────────────────────────────────────────

  if (loading && !triplet) {
    return (
      <div className="flex items-center justify-center h-64 text-indigo-400 font-mono text-sm">
        Cargando...
      </div>
    );
  }

  if (error && !triplet) {
    return (
      <div className="text-red-400 font-mono text-sm text-center p-8 space-y-4">
        <p>{error}</p>
        <button
          onClick={() => fetchNext(shownIds)}
          className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg text-sm transition-colors"
        >
          Reintentar
        </button>
      </div>
    );
  }

  if (!triplet) return null;

  const progressPct = Math.round((roundIndex / TOTAL_ROUNDS) * 100);

  return (
    <div className="flex flex-col gap-6 w-full max-w-3xl mx-auto">

      {/* Round progress */}
      <div className="flex flex-col gap-1">
        <span className="text-gray-400 font-mono text-xs text-right">
          Ronda {roundIndex + 1} de {TOTAL_ROUNDS}
        </span>
        <div className="w-full h-1.5 bg-gray-700 rounded-full overflow-hidden">
          <div
            className="h-full bg-indigo-500 rounded-full transition-all duration-300"
            style={{ width: `${progressPct}%` }}
          />
        </div>
      </div>

      {/* Reference */}
      <div className="border border-indigo-900 rounded-xl p-4 bg-gray-900/50">
        <div className="text-xs font-mono text-indigo-400 uppercase tracking-widest mb-3">
          Referencia
        </div>
        <AudioCard
          label="Canción de referencia"
          filename={triplet.anchor_filename}
          onChoose={() => {}}
          disabled={false}
          showChooseButton={false}
        />
      </div>

      {/* Question */}
      <p className="text-gray-300 font-mono text-sm text-center">
        ¿Cuál opción suena <span className="text-indigo-300 font-semibold">más parecida</span> a la referencia?
      </p>

      {error && (
        <div className="text-red-300 bg-red-950/40 border border-red-900 rounded-lg p-3 font-mono text-sm text-center">
          {error} Tu selección no se registró; vuelve a elegir para reintentar.
        </div>
      )}

      {/* Options */}
      <div className="grid grid-cols-2 gap-4">
        <div className="flex flex-col gap-2">
          <span className="text-xs font-mono text-gray-500 uppercase tracking-widest text-center">
            Opción A
          </span>
          <AudioCard
            label="Opción A"
            filename={triplet.option_a_filename}
            onChoose={() => handleChoice('a')}
            disabled={loading}
          />
        </div>
        <div className="flex flex-col gap-2">
          <span className="text-xs font-mono text-gray-500 uppercase tracking-widest text-center">
            Opción B
          </span>
          <AudioCard
            label="Opción B"
            filename={triplet.option_b_filename}
            onChoose={() => handleChoice('b')}
            disabled={loading}
          />
        </div>
      </div>

    </div>
  );
}
