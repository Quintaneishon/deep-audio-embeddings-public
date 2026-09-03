"""
MusiCNN-based contrastive learning model.

Extends the pretrained MusiCNN (MSD) by adding a trainable projection head
for supervised contrastive learning. Supports two-phase training:
  Phase 1: Freeze backbone, train only the projection head.
  Phase 2: Unfreeze backbone with a lower learning rate.

Memory note: the largest intermediate tensor is [B, 2097, T] where T depends
on audio duration (T ≈ 625 for 10 s, T ≈ 1875 for 30 s). Use
use_grad_checkpointing=True in Phase 2 to trade compute for ~3-4x less VRAM.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from backend.DL.models.MusiCNN import Musicnn


class MusiCNNContrastive(nn.Module):
    """
    MusiCNN backbone with a contrastive projection head.

    Architecture:
      1. MusiCNN backbone (pretrained on MSD) up to penultimate dense layer
      2. Projection head: 500 -> 500 -> 128
      3. L2 normalisation

    Inference: forward_extract() returns the 500-dim backbone embedding
    (dense1), NOT the projection head. The projection head is only used
    during training (forward()). This follows the SimCLR convention:
    the projection head improves training but the backbone generalises better.
    """

    def __init__(self, projection_dim=128, pretrained_weights=None, device=None):
        super().__init__()

        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.projection_dim = projection_dim
        self.dataset = 'msd'
        self.use_grad_checkpointing = False

        # MSD variant: backend_channel=512, dense_channel=500
        self.backbone = Musicnn(dataset='msd')
        self.embedding_dim = 500

        if pretrained_weights is not None:
            state = torch.load(pretrained_weights, map_location=self.device, weights_only=True)
            self.backbone.load_state_dict(state)

        # Projection head: embedding_dim -> embedding_dim -> projection_dim
        self.projection_head = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, projection_dim),
        )
        for layer in self.projection_head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

        self.to(self.device)
        self._freeze_backbone()

    # ------------------------------------------------------------------
    # Freeze / unfreeze helpers
    # ------------------------------------------------------------------

    def enable_grad_checkpointing(self):
        """Enable gradient checkpointing for phase 2 — trades compute for ~3x less VRAM."""
        self.use_grad_checkpointing = True

    def disable_grad_checkpointing(self):
        self.use_grad_checkpointing = False

    def _freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def _unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    # ------------------------------------------------------------------
    # Parameter group helpers
    # ------------------------------------------------------------------

    def get_trainable_parameters(self):
        """Return only the projection head parameters (phase 1)."""
        return self.projection_head.parameters()

    def get_all_parameters_grouped(self, backbone_lr=1e-5, head_lr=1e-3):
        """Return param groups with differential learning rates (phase 2)."""
        return [
            {'params': self.backbone.parameters(), 'lr': backbone_lr},
            {'params': self.projection_head.parameters(), 'lr': head_lr},
        ]

    # ------------------------------------------------------------------
    # Forward passes
    # ------------------------------------------------------------------

    def _spectrogram(self, x):
        """Spectrogram must stay in FP32 — mel underflows to 0 in FP16."""
        with torch.amp.autocast('cuda', enabled=False):
            x = self.backbone.spec(x.float())
            x = self.backbone.to_db(x)
            x = x.unsqueeze(1)
            x = self.backbone.spec_bn(x)
        return x

    def _frontend(self, x):
        """Pons parallel front-end → [B, 561, T]."""
        out = torch.cat([layer(x) for layer in self.backbone.layers], dim=1)
        return out.squeeze(2)

    def _backend(self, out):
        """Pons residual backend + global pooling → [B, 4194]."""
        length = out.size(2)
        res1 = self.backbone.layer1(out)
        res2 = self.backbone.layer2(res1) + res1
        res3 = self.backbone.layer3(res2) + res2
        # [B, 2097, T] — the largest tensor; freed immediately after pooling
        cat = torch.cat([out, res1, res2, res3], 1)
        del out, res1, res2, res3
        mp   = nn.MaxPool1d(length)(cat)
        avgp = nn.AvgPool1d(length)(cat)
        del cat
        return torch.cat([mp, avgp], dim=1).squeeze(2)  # [B, 4194]

    def _backbone_embedding(self, x):
        """Full backbone pass → 500-dim dense1 embedding."""
        if self.use_grad_checkpointing and self.training:
            # Gradient checkpointing: recompute activations on backward pass
            # instead of storing them. Saves ~3-4x VRAM at ~30% compute cost.
            spec = self._spectrogram(x)
            frontend = checkpoint(self._frontend, spec, use_reentrant=False)
            pooled   = checkpoint(self._backend,  frontend, use_reentrant=False)
        else:
            spec     = self._spectrogram(x)
            frontend = self._frontend(spec)
            pooled   = self._backend(frontend)
        return self.backbone.relu(self.backbone.bn(self.backbone.dense1(pooled)))

    def forward(self, x):
        """Training forward — returns L2-normalised projection head output [B, 128]."""
        z = self.projection_head(self._backbone_embedding(x))
        return F.normalize(z, p=2, dim=1)

    def forward_extract(self, x):
        """
        Inference forward — returns 500-dim backbone embedding (dense1), NOT the
        projection head. This follows SimCLR: the projection head is discarded at
        inference because the backbone representation generalises better.
        """
        return self._backbone_embedding(x)  # [B, 500]
