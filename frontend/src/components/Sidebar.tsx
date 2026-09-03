import { useState } from 'react';
import { Music2, Search, X } from 'lucide-react';
import { Separator } from './ui/separator';

interface Song {
  id: string;
  filename: string;
}

interface SidebarProps {
  listaCanciones: Song[];
  selectedSongs: string[];
  onSongsChange: (songs: string[]) => void;
  isLoading?: boolean;
  progress?: number;
  onCompare: () => void;
}

export function Sidebar({
  listaCanciones,
  selectedSongs,
  onSongsChange,
  isLoading,
  progress,
  onCompare,
}: SidebarProps) {
  const [searchQuery, setSearchQuery] = useState('');

  const filteredSongs = listaCanciones.filter(s =>
    s.filename.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const toggleSong = (filename: string) => {
    if (selectedSongs.includes(filename)) {
      onSongsChange(selectedSongs.filter(s => s !== filename));
    } else if (selectedSongs.length < 2) {
      onSongsChange([...selectedSongs, filename]);
    }
  };

  const removeSong = (filename: string) => {
    onSongsChange(selectedSongs.filter(s => s !== filename));
  };

  return (
    <div className="w-[280px] h-screen bg-white border-r border-gray-200 flex flex-col">
      {/* Header */}
      <div className="p-6 flex items-center gap-3">
        <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-[#6366f1] to-[#8b5cf6] flex items-center justify-center">
          <Music2 className="w-5 h-5 text-white" />
        </div>
        <div>
          <h1 className="text-gray-900 font-mono tracking-tight text-base font-medium">Deep Audio</h1>
          <p className="text-gray-500 text-xs font-mono">Embeddings</p>
        </div>
      </div>

      <Separator className="bg-gray-200" />

      {/* Song Selection */}
      <div className="flex-1 overflow-auto px-4 py-4">
        <div className="mb-4">
          <label className="text-gray-600 text-xs font-mono uppercase tracking-wider mb-3 block">
            Select 2 Songs to Compare
          </label>

          {/* Selected Songs Slots */}
          <div className="mb-3 space-y-2">
            {[0, 1].map((index) => (
              <div
                key={index}
                className="bg-gray-50 border border-gray-200 rounded-lg px-3 py-2 min-h-[40px] flex items-center justify-between"
              >
                {selectedSongs[index] ? (
                  <>
                    <span className="text-indigo-600 font-mono text-xs truncate">
                      {selectedSongs[index]}
                    </span>
                    <button
                      onClick={() => removeSong(selectedSongs[index])}
                      className="ml-2 text-gray-400 hover:text-red-500 transition-colors flex-shrink-0"
                    >
                      <X className="w-4 h-4" />
                    </button>
                  </>
                ) : (
                  <span className="text-gray-400 font-mono text-xs">
                    Song {index + 1}
                  </span>
                )}
              </div>
            ))}
          </div>

          {/* Search */}
          <div className="relative mb-3">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search songs..."
              className="w-full bg-gray-50 border border-gray-200 text-gray-900 font-mono text-sm
                         px-3 py-2 pl-10 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500/30
                         placeholder:text-gray-400"
            />
          </div>

          {/* Song List */}
          <div className="bg-gray-50 border border-gray-200 rounded-lg overflow-hidden max-h-[360px] overflow-y-auto">
            {filteredSongs.length > 0 ? (
              filteredSongs.map((song) => {
                const isSelected = selectedSongs.includes(song.filename);
                const canSelect = selectedSongs.length < 2 || isSelected;
                return (
                  <div
                    key={song.id}
                    onClick={() => canSelect && toggleSong(song.filename)}
                    className={`
                      px-3 py-2 font-mono text-xs transition-colors
                      border-b border-gray-200 last:border-b-0
                      ${isSelected
                        ? 'bg-indigo-50 text-indigo-600'
                        : canSelect
                          ? 'text-gray-600 hover:bg-gray-100 cursor-pointer'
                          : 'text-gray-300 cursor-not-allowed opacity-50'
                      }
                    `}
                  >
                    {song.filename}
                  </div>
                );
              })
            ) : (
              <div className="px-3 py-4 text-center text-gray-400 font-mono text-xs">
                No songs found
              </div>
            )}
          </div>
        </div>
      </div>

      {/* CTA Button */}
      <div className="p-4 border-t border-gray-200">
        <button
          onClick={onCompare}
          disabled={selectedSongs.length !== 2}
          className="w-full px-4 py-3 bg-gradient-to-r from-[#6366f1] to-[#8b5cf6] text-white font-mono rounded-lg
                     hover:from-[#5558e3] hover:to-[#7c4ee0] transition-all disabled:opacity-50 disabled:cursor-not-allowed
                     shadow-lg shadow-[#6366f1]/20 text-sm"
        >
          Highlight in Plot
        </button>
        {selectedSongs.length !== 2 && (
          <p className="text-gray-500 text-xs font-mono text-center mt-2">
            Select exactly 2 songs
          </p>
        )}
      </div>
    </div>
  );
}
