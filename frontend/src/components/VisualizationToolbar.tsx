import { ChevronDown } from 'lucide-react';

interface ModelDataset {
  value: string;
  label: string;
}

interface ModelConfig {
  label: string;
  datasets: ModelDataset[];
}

interface VisualizationToolbarProps {
  modelsConfig: Record<string, ModelConfig>;
  architecture: string;
  onArchitectureChange: (value: string) => void;
  dataset: string;
  onDatasetChange: (value: string) => void;
  dimensiones: number;
  onDimensionesChange: (value: number) => void;
}

function DropdownField({
  label,
  value,
  onChange,
  options,
}: {
  label: string;
  value: string;
  onChange: (val: string) => void;
  options: { value: string; label: string }[];
}) {
  return (
    <div className="flex flex-col gap-1">
      <label className="text-gray-500 text-xs font-mono uppercase tracking-wider">
        {label}
      </label>
      <div className="relative">
        <select
          value={value}
          onChange={(e) => onChange(e.target.value)}
          className="appearance-none bg-gray-50 border border-gray-200 text-gray-900 font-mono text-sm
                     px-3 py-2 pr-8 rounded-lg cursor-pointer hover:border-indigo-400 transition-colors
                     focus:outline-none focus:ring-2 focus:ring-indigo-500/30"
        >
          {options.map((opt) => (
            <option key={opt.value} value={opt.value}>
              {opt.label}
            </option>
          ))}
        </select>
        <ChevronDown className="absolute right-2 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400 pointer-events-none" />
      </div>
    </div>
  );
}

export function VisualizationToolbar({
  modelsConfig,
  architecture,
  onArchitectureChange,
  dataset,
  onDatasetChange,
  dimensiones,
  onDimensionesChange,
}: VisualizationToolbarProps) {
  const availableDatasets = modelsConfig[architecture]?.datasets ?? [];

  const architectureOptions = Object.entries(modelsConfig).map(([key, cfg]) => ({
    value: key,
    label: cfg.label,
  }));

  return (
    <div className="bg-white border-b border-gray-200 px-6 py-4">
      <div className="flex items-end gap-4 flex-wrap">
        <DropdownField
          label="Architecture"
          value={architecture}
          onChange={onArchitectureChange}
          options={
            architectureOptions.length > 0
              ? architectureOptions
              : [{ value: architecture, label: architecture }]
          }
        />
        <DropdownField
          label="Dataset"
          value={dataset}
          onChange={onDatasetChange}
          options={
            availableDatasets.length > 0
              ? availableDatasets
              : [{ value: dataset, label: dataset }]
          }
        />
        <DropdownField
          label="Dimensions"
          value={String(dimensiones)}
          onChange={(val) => onDimensionesChange(parseInt(val))}
          options={[
            { value: '2', label: '2D' },
            { value: '3', label: '3D' },
          ]}
        />
      </div>
    </div>
  );
}
