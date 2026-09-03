"""
MERT-v1-95M with LoRA adapters for contrastive learning.

Architecture:
  MERT-v1-95M backbone (95M params, HuBERT-style transformer, frozen)
  └─ LoRA adapters on attention q_proj + v_proj  (~0.6M trainable)
  └─ Projection head: 768 → 768 → 128 + L2-norm (~600K trainable)

Total trainable: ~1.2M  vs  95M base  (< 1.3% of params)

Single-phase training — no freeze/unfreeze needed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel, Wav2Vec2FeatureExtractor
from peft import LoraConfig, get_peft_model

MERT_SAMPLE_RATE = 24000
MERT_MAX_SAMPLES = 30 * MERT_SAMPLE_RATE  # 720,000 samples


class MERTContrastiveLora(nn.Module):

    def __init__(
        self,
        model_name: str = 'm-a-p/MERT-v1-95M',
        projection_dim: int = 128,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        target_modules=None,
        device=None,
        cache_dir: str = None,
    ):
        super().__init__()

        self.projection_dim = projection_dim
        self.embedding_dim = 768   # MERT-v1-95M hidden size
        self.sample_rate = MERT_SAMPLE_RATE
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._device = device

        # ── Feature extractor (CPU preprocessing, no grad) ──────────────
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )

        # ── Backbone + LoRA ─────────────────────────────────────────────
        base = AutoModel.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True,
            output_hidden_states=False,
        )

        if target_modules is None:
            target_modules = ['q_proj', 'v_proj']

        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias='none',
        )
        self.backbone = get_peft_model(base, lora_cfg)
        self.backbone.to(device)

        # ── Projection head ─────────────────────────────────────────────
        self.projection_head = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, projection_dim),
        )
        self.projection_head.to(device)

    # ------------------------------------------------------------------ #
    # Preprocessing                                                        #
    # ------------------------------------------------------------------ #

    def _preprocess(self, audio: torch.Tensor) -> dict:
        """
        Normalize a batch of waveforms with the MERT feature extractor.
        audio: [B, T]  float32, already at 24 kHz, already padded/trimmed.
        """
        audio_list = audio.cpu().float().numpy().tolist()
        inputs = self.processor(
            audio_list,
            sampling_rate=self.sample_rate,
            return_tensors='pt',
            padding=False,
        )
        return {k: v.to(self._device) for k, v in inputs.items()}

    # ------------------------------------------------------------------ #
    # Forward                                                              #
    # ------------------------------------------------------------------ #

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        audio: [B, T]  waveform at 24 kHz
        returns: [B, projection_dim]  L2-normalised embeddings
        """
        inputs = self._preprocess(audio)
        outputs = self.backbone(**inputs)
        pooled = outputs.last_hidden_state.mean(dim=1)
        projected = self.projection_head(pooled)
        return F.normalize(projected, dim=-1)

    def forward_extract(self, audio: torch.Tensor) -> torch.Tensor:
        """
        audio: [B, T]  waveform at 24 kHz
        returns: embedding [B, projection_dim] L2-normalised
        """
        inputs = self._preprocess(audio)
        outputs = self.backbone(**inputs)
        pooled = outputs.last_hidden_state.mean(dim=1)
        projected = self.projection_head(pooled)
        return F.normalize(projected, dim=-1)

    # ------------------------------------------------------------------ #
    # Parameter helpers                                                    #
    # ------------------------------------------------------------------ #

    def get_trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # ------------------------------------------------------------------ #
    # Checkpoint helpers                                                   #
    # ------------------------------------------------------------------ #

    def state_dict_trainable(self) -> dict:
        """Return only trainable params (LoRA adapters + projection head).
        Small file (~8 MB). Use for per-epoch checkpoints."""
        return {k: v for k, v in self.state_dict().items()
                if any(k.startswith(pfx) for pfx in
                       ('backbone.base_model.model.encoder',
                        'projection_head'))
                and self.get_parameter(k).requires_grad
                }

    def save_adapters(self, directory: str):
        """Save LoRA adapters via PEFT (tiny files, easy to share)."""
        self.backbone.save_pretrained(directory)

    def load_adapters(self, directory: str):
        """Load LoRA adapters saved with save_adapters()."""
        from peft import PeftModel
        base = self.backbone.base_model.model
        self.backbone = PeftModel.from_pretrained(base, directory)
        self.backbone.to(self._device)
