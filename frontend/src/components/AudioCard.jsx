import { useState, useRef, useEffect } from 'react';
import { Play, Pause } from 'lucide-react';

const API_BASE = process.env.REACT_APP_API_BASE_URL || 'http://127.0.0.1:5000';

export function AudioCard({ label, filename, onChoose, disabled, showChooseButton = true }) {
  const audioRef    = useRef(null);
  const barRef      = useRef(null);
  const dragging    = useRef(false);
  const [isPlaying,   setIsPlaying]   = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration,    setDuration]    = useState(0);

  useEffect(() => {
    setIsPlaying(false);
    setCurrentTime(0);
    setDuration(0);
  }, [filename]);

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;
    const onMeta  = () => setDuration(audio.duration || 0);
    const onTime  = () => { if (!dragging.current) setCurrentTime(audio.currentTime); };
    const onEnded = () => { setIsPlaying(false); setCurrentTime(0); };
    audio.addEventListener('loadedmetadata', onMeta);
    audio.addEventListener('timeupdate', onTime);
    audio.addEventListener('ended', onEnded);
    return () => {
      audio.removeEventListener('loadedmetadata', onMeta);
      audio.removeEventListener('timeupdate', onTime);
      audio.removeEventListener('ended', onEnded);
      audio.pause();
    };
  }, [filename]);

  const togglePlay = () => {
    const audio = audioRef.current;
    if (!audio) return;
    if (isPlaying) { audio.pause(); setIsPlaying(false); }
    else           { audio.play().catch(() => setIsPlaying(false)); setIsPlaying(true); }
  };

  // Seek to position based on click/drag on the bar
  const seekTo = (clientX) => {
    const bar   = barRef.current;
    const audio = audioRef.current;
    if (!bar || !audio || !duration) return;
    const { left, width } = bar.getBoundingClientRect();
    const pct  = Math.max(0, Math.min(1, (clientX - left) / width));
    const time = pct * duration;
    audio.currentTime = time;
    setCurrentTime(time);
  };

  const onPointerDown = (e) => {
    dragging.current = true;
    barRef.current.setPointerCapture(e.pointerId);
    seekTo(e.clientX);
  };
  const onPointerMove = (e) => { if (dragging.current) seekTo(e.clientX); };
  const onPointerUp   = (e) => { dragging.current = false; };

  const progress = duration > 0 ? (currentTime / duration) * 100 : 0;
  const elapsed  = Math.round(currentTime);

  return (
    <div className={`bg-gray-800 border rounded-xl p-5 flex flex-col gap-4 transition-colors
                     ${disabled
                       ? 'border-gray-700 opacity-60 pointer-events-none'
                       : 'border-gray-600 hover:border-indigo-500'}`}>

      <div className="text-xs font-mono text-gray-400 truncate">{label}</div>

      {/* Play / progress */}
      <div className="flex flex-col items-center gap-3">
        <button
          onClick={togglePlay}
          className="w-14 h-14 rounded-full bg-indigo-600 hover:bg-indigo-500 flex items-center
                     justify-center text-white transition-colors shadow-lg"
        >
          {isPlaying ? <Pause size={22} /> : <Play size={22} className="ml-1" />}
        </button>

        {/* Draggable progress bar */}
        <div
          ref={barRef}
          className="w-full h-3 bg-gray-700 rounded-full overflow-hidden cursor-pointer"
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
        >
          <div
            className="h-full bg-indigo-500 rounded-full transition-none pointer-events-none"
            style={{ width: `${progress}%` }}
          />
        </div>
        <span className="text-xs font-mono text-gray-500">
          {elapsed}s{duration > 0 ? ` / ${Math.round(duration)}s` : ''}
        </span>
      </div>

      <audio
        ref={audioRef}
        src={`${API_BASE}/audio/${filename}`}
        preload="metadata"
        crossOrigin="anonymous"
      />

      {showChooseButton && (
        <button
          onClick={onChoose}
          className="w-full py-2.5 bg-indigo-600 hover:bg-indigo-500 text-white font-mono text-sm
                     rounded-lg transition-colors shadow-md shadow-indigo-900/40"
        >
          Elegir esta
        </button>
      )}
    </div>
  );
}
