import { X } from 'lucide-react';
import { useEffect } from 'react';
import { AudioPlayer } from './AudioPlayer';

interface CompareResult {
  song1: string;
  song2: string;
  model: string;
  dataset: string;
  serra09_dist: number | null;
  mfcc_cosine_dist: number | null;
  emb_cosine_dist: number | null;
  key1: string;
  scale1: string;
  key2: string;
  scale2: string;
  key_cof_distance: number | null;
  emb_error: string | null;
}

interface SongInfoPopupProps {
  isOpen: boolean;
  onClose: () => void;
  compareResult: CompareResult | null;
  isLoading: boolean;
  error: string | null;
  tag1?: string;
  tag2?: string;
}

function formatDist(val: number | null | undefined): string {
  if (val == null) return '∞';
  return val.toFixed(4);
}

const API_BASE = process.env.REACT_APP_API_BASE_URL || 'http://127.0.0.1:5000';

export function SongInfoPopup({
  isOpen,
  onClose,
  compareResult,
  isLoading,
  error,
  tag1,
  tag2,
}: SongInfoPopupProps) {
  useEffect(() => {
    const handleEsc = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isOpen) onClose();
    };
    window.addEventListener('keydown', handleEsc);
    return () => window.removeEventListener('keydown', handleEsc);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  const songs = compareResult
    ? [
        { name: compareResult.song1, tag: tag1 || '—', key: compareResult.key1, scale: compareResult.scale1 },
        { name: compareResult.song2, tag: tag2 || '—', key: compareResult.key2, scale: compareResult.scale2 },
      ]
    : [];

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-black/50 z-40 backdrop-blur-sm"
        onClick={onClose}
      />

      {/* Popup */}
      <div className="fixed top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 z-50 w-full max-w-4xl max-h-[90vh] overflow-auto">
        <div className="bg-white rounded-2xl shadow-2xl border border-gray-200 m-4">

          {/* Header */}
          <div className="flex items-center justify-between p-6 border-b border-gray-200">
            <div>
              <h2 className="text-xl text-gray-900 font-mono">Song Comparison</h2>
              <p className="text-sm text-gray-500 font-mono mt-1">
                {compareResult
                  ? `${compareResult.model} / ${compareResult.dataset} · audio analysis and distance metrics`
                  : 'Audio analysis and distance metrics'}
              </p>
            </div>
            <button
              onClick={onClose}
              className="text-gray-400 hover:text-gray-600 transition-colors p-2 hover:bg-gray-100 rounded-lg"
            >
              <X className="w-5 h-5" />
            </button>
          </div>

          {/* Content */}
          <div className="p-6">
            {isLoading && (
              <div className="text-center py-12 text-indigo-600 font-mono text-sm">
                Computing distances…
              </div>
            )}

            {error && !isLoading && (
              <div className="text-red-600 text-sm bg-red-50 rounded-lg p-4 font-mono">
                {error}
              </div>
            )}

            {compareResult && !isLoading && (
              <>
                {/* Individual song cards */}
                <div className="grid grid-cols-2 gap-6 mb-6">
                  {songs.map((song, index) => (
                    <div
                      key={index}
                      className="bg-gradient-to-br from-indigo-50 to-purple-50 rounded-xl p-6 border border-indigo-100"
                    >
                      <div className="flex items-center gap-2 mb-4">
                        <div className="w-8 h-8 rounded-full bg-gradient-to-br from-indigo-500 to-purple-500 flex items-center justify-center text-white font-mono text-sm flex-shrink-0">
                          {index + 1}
                        </div>
                        <h3 className="text-sm text-gray-900 font-mono font-medium truncate" title={song.name}>
                          {song.name}
                        </h3>
                      </div>

                      <div className="space-y-3">
                        {/* Info */}
                        <div className="bg-white rounded-lg p-4 border border-indigo-100 space-y-3">
                          <div>
                            <div className="text-xs text-gray-500 font-mono uppercase tracking-wider mb-1">Genre</div>
                            <div className="text-sm text-gray-900 font-mono font-medium">{song.tag}</div>
                          </div>
                          <div>
                            <div className="text-xs text-gray-500 font-mono uppercase tracking-wider mb-1">Harmonic Key</div>
                            <div className="text-sm text-gray-900 font-mono font-medium">
                              {song.key && song.scale ? `${song.key} ${song.scale}` : '—'}
                            </div>
                          </div>
                        </div>

                        {/* Audio player */}
                        <div className="bg-white rounded-lg p-4 border border-indigo-100">
                          <div className="text-xs text-gray-500 font-mono uppercase tracking-wider mb-3">
                            Audio Player
                          </div>
                          <AudioPlayer
                            src={`${API_BASE}/audio/${song.name}`}
                            label={song.name}
                          />
                        </div>
                      </div>
                    </div>
                  ))}
                </div>

                {/* Distance metrics */}
                <div className="bg-gray-50 rounded-xl p-6 border border-gray-200">
                  <h4 className="text-sm text-gray-900 font-mono font-medium mb-4 uppercase tracking-wider">
                    Distance Metrics
                  </h4>
                  <div className="grid grid-cols-3 gap-4">
                    <div className="bg-white rounded-lg p-4 border border-gray-200">
                      <div className="text-xs text-gray-500 font-mono uppercase tracking-wider mb-2">Serra09 Distance</div>
                      <div className="text-2xl text-indigo-600 font-mono font-bold">
                        {formatDist(compareResult.serra09_dist)}
                      </div>
                      <div className="text-xs text-gray-400 font-mono mt-1">Harmonic similarity</div>
                    </div>
                    <div className="bg-white rounded-lg p-4 border border-gray-200">
                      <div className="text-xs text-gray-500 font-mono uppercase tracking-wider mb-2">Embedding Cosine</div>
                      <div className="text-2xl text-purple-600 font-mono font-bold">
                        {formatDist(compareResult.emb_cosine_dist)}
                      </div>
                      <div className="text-xs text-gray-400 font-mono mt-1">Neural similarity</div>
                    </div>
                    <div className="bg-white rounded-lg p-4 border border-gray-200">
                      <div className="text-xs text-gray-500 font-mono uppercase tracking-wider mb-2">MFCC Timbre</div>
                      <div className="text-2xl text-indigo-600 font-mono font-bold">
                        {formatDist(compareResult.mfcc_cosine_dist)}
                      </div>
                      <div className="text-xs text-gray-400 font-mono mt-1">Timbral similarity</div>
                    </div>
                  </div>

                  <div className="mt-4 text-xs text-gray-500 font-mono text-center">
                    Lower values indicate higher similarity between songs
                  </div>

                  {compareResult.emb_error && (
                    <div className="mt-3 text-xs text-amber-600 bg-amber-50 rounded p-2 font-mono">
                      Embedding note: {compareResult.emb_error}
                    </div>
                  )}
                </div>
              </>
            )}
          </div>

          {/* Footer */}
          <div className="flex items-center justify-end gap-3 p-6 border-t border-gray-200 bg-gray-50 rounded-b-2xl">
            <button
              onClick={onClose}
              className="px-4 py-2 bg-white border border-gray-300 text-gray-700 font-mono text-sm rounded-lg hover:bg-gray-50 transition-colors"
            >
              Close
            </button>
          </div>
        </div>
      </div>
    </>
  );
}
