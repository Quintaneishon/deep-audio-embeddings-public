"""
VGG-based contrastive learning model.

Extends the pretrained VGG_Res (MSD) by adding a trainable projection head
for multi-label supervised contrastive learning (Jaccard-weighted positives).
Supports two-phase training:
  Phase 1: Freeze backbone, train only the projection head.
  Phase 2: Unfreeze backbone with differential learning rates.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.DL.models.VGG import VGG_Res


class VGGContrastive(nn.Module):
    """
    VGG_Res backbone with a contrastive projection head.

    Architecture:
      1. VGG_Res backbone (pretrained on MSD) up to the dense1 embedding layer
      2. Projection head: Linear(512 → 256) → ReLU → Linear(256 → 128)
      3. L2 normalisation
    """

    # VGG_Res default: n_channels=128 → dense1 outputs n_channels*4 = 512
    EMBEDDING_DIM = 512

    def __init__(self, projection_dim=128, pretrained_weights=None, device=None):
        super().__init__()

        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.projection_dim = projection_dim
        self.embedding_dim = self.EMBEDDING_DIM
        self.dataset = 'msd'

        self.backbone = VGG_Res(n_channels=128, n_class=50)

        if pretrained_weights is not None:
            state = torch.load(pretrained_weights, map_location=self.device, weights_only=True)
            self.backbone.load_state_dict(state)

        # Projection head: 512 → 256 → projection_dim
        self.projection_head = nn.Sequential(
            nn.Linear(self.embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, projection_dim),
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

    def _freeze_backbone(self):
        """Freeze all backbone parameters (phase 1)."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def _unfreeze_backbone(self):
        """Unfreeze all backbone parameters (phase 2)."""
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

    def _backbone_embedding(self, x):
        """
        Run VGG_Res backbone up to the dense1 embedding (512-dim).

        Mel-spectrogram preprocessing runs in FP32 even under AMP to prevent
        log10(0) = -Inf from propagating when quiet mel bands underflow to 0.
        """
        with torch.amp.autocast('cuda', enabled=False):
            x = self.backbone.spec(x.float())
            x = self.backbone.to_db(x)
            x = x.unsqueeze(1)
            x = self.backbone.spec_bn(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.backbone.layer5(x)
        x = self.backbone.layer6(x)
        x = self.backbone.layer7(x)
        x = x.squeeze(2)  # squeeze freq dim → [B, 512, T]

        if x.size(-1) != 1:
            x = nn.MaxPool1d(x.size(-1))(x)
        x = x.squeeze(2)  # → [B, 512]

        x = self.backbone.dense1(x)
        x = self.backbone.bn(x)
        return self.backbone.relu(x)  # [B, 512]

    def forward(self, x):
        """
        Contrastive training forward pass.

        Returns:
            embeddings: [batch, projection_dim] L2-normalised
        """
        embedding = self._backbone_embedding(x)
        z = self.projection_head(embedding)
        return F.normalize(z, p=2, dim=1)

    def forward_extract(self, x):
        """
        Inference forward pass returning the projected embedding.

        Returns:
            projected: [batch, projection_dim] L2-normalised
        """
        embedding = self._backbone_embedding(x)
        z = self.projection_head(embedding)
        return F.normalize(z, p=2, dim=1)
