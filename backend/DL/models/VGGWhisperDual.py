"""
Dual-backbone model: VGG_Res + Whisper encoder joint contrastive training.

Architecture:
  audio ──► VGG_Res backbone (MSD pretrained)   ──► 512-dim ─┐
         └► Whisper encoder (base, full output)  ──► 512-dim ─┴─► concat 1024-dim ──► projection 128-dim

Both backbones see the same raw 16kHz waveform.
VGG computes its mel-spectrogram internally.
Whisper receives pad-trimmed audio → log-mel → transformer encoder.

Training phases:
  Phase 1: Both backbones frozen, projection head trains only.
  Phase 2: Both backbones unfrozen with differential LR.

The combined embedding merges:
  - VGG convolutional features: local texture, timbre, rhythmic patterns
  - Whisper transformer features: long-range temporal structure, harmony
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from backend.DL.models.VGG import VGG_Res
from backend.DL.models.Whisper import WhisperEmbedding, log_mel_spectrogram, pad_or_trim


class VGGWhisperDual(nn.Module):
    """
    Dual-backbone contrastive model (VGG_Res + Whisper encoder).

    Architecture:
      1. VGG_Res backbone (MSD pretrained) → 512-dim embedding
      2. Whisper base encoder (full output, mean-pooled) → 512-dim
      3. Concatenation → 1024-dim
      4. Projection head: Linear(1024→512) → ReLU → Linear(512→128)
      5. L2 normalisation
    """

    VGG_DIM = 512
    WHISPER_DIM = 512  # Whisper base n_audio_state
    COMBINED_DIM = VGG_DIM + WHISPER_DIM  # 1024

    def __init__(
        self,
        projection_dim: int = 128,
        whisper_size: str = 'base',
        vgg_pretrained_weights: str = None,
        device=None,
    ):
        super().__init__()

        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.projection_dim = projection_dim
        self.whisper_size = whisper_size

        # ── VGG backbone ──────────────────────────────────────────────────
        self.vgg_backbone = VGG_Res(n_channels=128, n_class=50)
        if vgg_pretrained_weights is not None:
            state = torch.load(vgg_pretrained_weights, map_location=self.device, weights_only=True)
            self.vgg_backbone.load_state_dict(state)

        # ── Whisper encoder ───────────────────────────────────────────────
        _whisper_wrapper = WhisperEmbedding(model_name=whisper_size, device=self.device)
        self.whisper_model = _whisper_wrapper.whisper_model
        self.n_mels = _whisper_wrapper.n_mels
        del _whisper_wrapper

        # ── Projection head ───────────────────────────────────────────────
        self.projection_head = nn.Sequential(
            nn.Linear(self.COMBINED_DIM, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim),
        )
        for layer in self.projection_head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

        self.to(self.device)
        self._freeze_backbones()

    # ------------------------------------------------------------------
    # Freeze / unfreeze helpers
    # ------------------------------------------------------------------

    def _freeze_backbones(self):
        """Phase 1: freeze both backbones."""
        for p in self.vgg_backbone.parameters():
            p.requires_grad = False
        for p in self.whisper_model.parameters():
            p.requires_grad = False

    def _unfreeze_backbones(self):
        """Phase 2: unfreeze both backbones."""
        for p in self.vgg_backbone.parameters():
            p.requires_grad = True
        for p in self.whisper_model.parameters():
            p.requires_grad = True

    # ------------------------------------------------------------------
    # Parameter group helpers
    # ------------------------------------------------------------------

    def get_trainable_parameters(self):
        """Phase 1: only projection head parameters."""
        return self.projection_head.parameters()

    def get_all_parameters_grouped(self, backbone_lr: float = 1e-5, head_lr: float = 1e-3):
        """Phase 2: differential learning rates."""
        backbone_params = (
            list(self.vgg_backbone.parameters())
            + list(self.whisper_model.parameters())
        )
        return [
            {'params': backbone_params, 'lr': backbone_lr},
            {'params': list(self.projection_head.parameters()), 'lr': head_lr},
        ]

    # ------------------------------------------------------------------
    # Backbone embeddings
    # ------------------------------------------------------------------

    def _vgg_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """VGG_Res → [B, 512]. Identical to VGGContrastive._backbone_embedding."""
        with torch.amp.autocast('cuda', enabled=False):
            x = self.vgg_backbone.spec(x.float())
            x = self.vgg_backbone.to_db(x)
            x = x.unsqueeze(1)
            x = self.vgg_backbone.spec_bn(x)

        x = self.vgg_backbone.layer1(x)
        x = self.vgg_backbone.layer2(x)
        x = self.vgg_backbone.layer3(x)
        x = self.vgg_backbone.layer4(x)
        x = self.vgg_backbone.layer5(x)
        x = self.vgg_backbone.layer6(x)
        x = self.vgg_backbone.layer7(x)
        x = x.squeeze(2)
        if x.size(-1) != 1:
            x = nn.MaxPool1d(x.size(-1))(x)
        x = x.squeeze(2)  # [B, 512]
        x = self.vgg_backbone.dense1(x)
        x = self.vgg_backbone.bn(x)
        return self.vgg_backbone.relu(x)  # [B, 512]

    def _whisper_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Whisper encoder → [B, 512].

        Audio-to-mel conversion goes through CPU/numpy (not differentiable).
        Gradients flow through conv1, conv2, and transformer blocks in phase 2.
        """
        B = x.shape[0]
        mel_list = []
        for i in range(B):
            audio_padded = pad_or_trim(x[i].cpu().numpy())
            mel = log_mel_spectrogram(audio_padded, n_mels=self.n_mels)
            mel_list.append(mel)
        mel = torch.stack(mel_list).to(self.device)  # [B, n_mels, n_frames]

        encoder = self.whisper_model.encoder
        h = F.gelu(encoder.conv1(mel))
        h = F.gelu(encoder.conv2(h))
        h = h.permute(0, 2, 1)  # [B, n_ctx, n_state]
        h = (h + encoder.positional_embedding).to(h.dtype)

        for block in encoder.blocks:
            h = block(h)
        h = encoder.ln_post(h)

        return h.mean(dim=1)  # [B, 512]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Contrastive training forward pass.

        Returns:
            [B, projection_dim] L2-normalised embeddings
        """
        vgg_feat = self._vgg_embedding(x)           # [B, 512]
        whisper_feat = self._whisper_embedding(x)   # [B, 512]
        combined = torch.cat([vgg_feat, whisper_feat], dim=1)  # [B, 1024]
        z = self.projection_head(combined)
        return F.normalize(z, p=2, dim=1)  # [B, projection_dim]

    def forward_extract(self, x: torch.Tensor) -> torch.Tensor:
        """Inference forward pass (identical to forward)."""
        return self.forward(x)
