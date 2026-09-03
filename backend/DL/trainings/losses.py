"""
Loss functions for contrastive learning.

Implements Supervised Contrastive Loss (SupCon) for learning embeddings
with genre supervision.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Learning Loss.

    Based on: "Supervised Contrastive Learning" (Khosla et al., NeurIPS 2020)
    https://arxiv.org/abs/2004.11362

    This loss pulls together embeddings from the same class (positive pairs)
    and pushes apart embeddings from different classes (negative pairs).

    The embeddings are expected to be L2 normalized, so that cosine similarity
    equals the dot product.
    """

    def __init__(self, temperature=0.07, contrast_mode='all'):
        """
        Initialize Supervised Contrastive Loss.

        Args:
            temperature: Temperature parameter for scaling similarities (default: 0.07)
            contrast_mode: 'all' uses all samples as contrast, 'one' uses one positive
        """
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode

    def forward(self, features, labels):
        """
        Compute supervised contrastive loss.

        Args:
            features: [batch_size, embedding_dim] - L2 normalized embeddings
            labels: [batch_size] - integer class labels

        Returns:
            loss: Scalar tensor with the contrastive loss
        """
        device = features.device
        features.shape[0]

        # Ensure features are normalized
        features = F.normalize(features, p=2, dim=1)

        # Compute similarity matrix: cosine similarity = dot product (since normalized)
        # [batch_size, batch_size]
        similarity_matrix = torch.matmul(features, features.T)

        # Scale by temperature
        similarity_matrix = similarity_matrix / self.temperature

        # Create mask for positive pairs (same label)
        # [batch_size, batch_size]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # Mask out self-contrast (diagonal)
        logits_mask = torch.ones_like(mask).to(device)
        logits_mask.fill_diagonal_(0)

        # Apply mask to remove self-contrast
        mask = mask * logits_mask

        # Compute log probabilities
        # Subtract max for numerical stability
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()

        # Compute log-sum-exp of negative pairs
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        # Compute mean of log-likelihood over positive pairs
        # Handle case where a class has only one sample in the batch
        mask_sum = mask.sum(1)
        mask_sum = torch.clamp(mask_sum, min=1.0)  # Avoid division by zero

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum

        # Loss is negative log-likelihood
        loss = -mean_log_prob_pos
        loss = loss.mean()

        return loss


class HybridJaccardBPMLoss(nn.Module):
    """
    Contrastive loss with continuous positive-pair weights derived from
    Jaccard genre similarity and BPM Gaussian similarity.

    w_ij = alpha * jaccard(genres_i, genres_j)
         + (1 - alpha) * exp(-|bpm_i - bpm_j|^2 / (2 * sigma^2))

    Pairs with w_ij > 0 act as soft positives; the weight modulates how
    strongly each positive pair contributes to the loss.
    """

    def __init__(self, temperature=0.07, alpha=0.7, bpm_sigma=10.0):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.bpm_sigma = bpm_sigma

    def forward(self, features, genre_vectors, bpms):
        """
        Args:
            features:      [B, D]  L2-normalised embeddings.
            genre_vectors: [B, G]  multi-hot float genre vectors.
            bpms:          [B]     BPM values (float).
        """
        device = features.device
        B = features.shape[0]

        features = F.normalize(features, p=2, dim=1)

        # Pairwise Jaccard similarity [B, B]
        intersection = torch.matmul(genre_vectors, genre_vectors.T)
        row_sum = genre_vectors.sum(dim=1, keepdim=True)
        union = row_sum + row_sum.T - intersection
        jaccard = intersection / (union + 1e-8)

        # Pairwise BPM Gaussian similarity [B, B]
        bpm_diff = bpms.unsqueeze(1) - bpms.unsqueeze(0)
        bpm_sim = torch.exp(-bpm_diff.pow(2) / (2 * self.bpm_sigma ** 2))

        # Blended weight matrix
        weights = self.alpha * jaccard + (1.0 - self.alpha) * bpm_sim

        # Self-contrast mask (exclude diagonal)
        logits_mask = torch.ones(B, B, device=device)
        logits_mask.fill_diagonal_(0)
        weights = weights * logits_mask

        # Cosine similarity scaled by temperature
        sim = torch.matmul(features, features.T) / self.temperature

        # Numerical stability
        logits_max, _ = sim.max(dim=1, keepdim=True)
        logits = sim - logits_max.detach()

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        # Weighted mean over positive pairs
        weight_sum = weights.sum(1)
        weight_sum = torch.clamp(weight_sum, min=1e-8)

        mean_log_prob = (weights * log_prob).sum(1) / weight_sum

        return -mean_log_prob.mean()


class MultiLabelSupConLoss(nn.Module):
    """
    Supervised Contrastive Loss for multi-label (multi-hot) genre labels.

    Positive-pair weight is the Jaccard similarity between genre sets:
        w_ij = |genres_i ∩ genres_j| / |genres_i ∪ genres_j|

    Pairs with w_ij > 0 act as soft positives; pairs with w_ij = 0 are negatives.
    Self-contrast (diagonal) is always excluded.

    Args:
        temperature: Temperature scaling for cosine similarities (default: 0.07)
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, D]  L2-normalised embeddings
            labels:   [B, C]  multi-hot float genre vectors

        Returns:
            Scalar loss tensor
        """
        device = features.device
        B = features.shape[0]

        features = F.normalize(features, p=2, dim=1)

        # Pairwise Jaccard similarity from multi-hot labels [B, B]
        labels = labels.float().to(device)
        intersection = torch.matmul(labels, labels.T)
        row_sum = labels.sum(dim=1, keepdim=True)
        union = row_sum + row_sum.T - intersection
        weights = intersection / (union + 1e-8)

        # Exclude self-contrast
        logits_mask = torch.ones(B, B, device=device)
        logits_mask.fill_diagonal_(0)
        weights = weights * logits_mask

        # Cosine similarities scaled by temperature
        sim = torch.matmul(features, features.T) / self.temperature

        # Numerical stability
        logits_max, _ = sim.max(dim=1, keepdim=True)
        logits = sim - logits_max.detach()

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        # Weighted mean log-likelihood over positive pairs
        weight_sum = weights.sum(1).clamp(min=1e-8)
        mean_log_prob = (weights * log_prob).sum(1) / weight_sum

        return -mean_log_prob.mean()


class MultiSignalLoss(nn.Module):
    """
    Contrastive loss weighted by multiple structural audio signals.

    w_ij = α·jaccard(genres_i, genres_j)          -- genre overlap (soft, noisy)
         + β·gaussian(|bpm_i  - bpm_j|, σ_bpm)    -- tempo proximity
         + γ·jaccard(instruments_i, instruments_j) -- instrument overlap
         + ε·(time_sig_i == time_sig_j)            -- same meter (binary)
         + ζ·gaussian(|sq_i  - sq_j|,  σ_sq)      -- structural similarity

    Weights must sum to 1.0.  Mood is intentionally excluded (too subjective).

    Default weights (tunable as hyperparameters):
        α=0.15  β=0.30  γ=0.25  ε=0.10  ζ=0.20
    """

    def __init__(
        self,
        temperature: float = 0.07,
        # Signal weights
        w_genre: float       = 0.15,
        w_bpm: float         = 0.30,
        w_instrument: float  = 0.25,
        w_timesig: float     = 0.10,
        w_squareness: float  = 0.20,
        # Gaussian bandwidths
        bpm_sigma: float        = 10.0,   # BPM — ~1 sub-genre width
        squareness_sigma: float = 0.15,   # squareness score in [0,1]
    ):
        super().__init__()
        total = w_genre + w_bpm + w_instrument + w_timesig + w_squareness
        assert abs(total - 1.0) < 1e-4, f"Weights must sum to 1.0, got {total:.4f}"

        self.temperature      = temperature
        self.w_genre          = w_genre
        self.w_bpm            = w_bpm
        self.w_instrument     = w_instrument
        self.w_timesig        = w_timesig
        self.w_squareness     = w_squareness
        self.bpm_sigma        = bpm_sigma
        self.squareness_sigma = squareness_sigma

    @staticmethod
    def _jaccard(vecs: torch.Tensor) -> torch.Tensor:
        """Pairwise Jaccard similarity for multi-hot vectors. [B, B]"""
        inter = torch.matmul(vecs, vecs.T)
        row_sum = vecs.sum(dim=1, keepdim=True)
        union = row_sum + row_sum.T - inter
        return inter / (union + 1e-8)

    @staticmethod
    def _gaussian(values: torch.Tensor, sigma: float) -> torch.Tensor:
        """Pairwise Gaussian similarity for a 1-D signal. [B, B]"""
        diff = values.unsqueeze(1) - values.unsqueeze(0)
        return torch.exp(-diff.pow(2) / (2 * sigma ** 2))

    def forward(
        self,
        features: torch.Tensor,            # [B, D]  L2-normalised embeddings
        genre_vectors: torch.Tensor,        # [B, G]  multi-hot genres
        instrument_vectors: torch.Tensor,   # [B, I]  multi-hot instruments
        bpms: torch.Tensor,                 # [B]     BPM float
        time_signatures: torch.Tensor,      # [B]     int (3 or 4)
        squareness_scores: torch.Tensor,    # [B]     float in [0, 1]
    ) -> torch.Tensor:
        device = features.device
        B = features.shape[0]

        features = F.normalize(features, p=2, dim=1)

        # Move all inputs to the same device
        genre_vectors      = genre_vectors.to(device)
        instrument_vectors = instrument_vectors.to(device)
        bpms               = bpms.to(device)
        time_signatures    = time_signatures.to(device).float()
        squareness_scores  = squareness_scores.to(device)

        # ── Individual similarity matrices ─────────────────────────────────
        genre_sim   = self._jaccard(genre_vectors)                        # [B, B]
        instr_sim   = self._jaccard(instrument_vectors)                   # [B, B]
        bpm_sim     = self._gaussian(bpms, self.bpm_sigma)                # [B, B]
        sq_sim      = self._gaussian(squareness_scores, self.squareness_sigma)  # [B, B]

        # Time signature: 1 if same, 0 otherwise
        timesig_sim = torch.eq(                                           # [B, B]
            time_signatures.unsqueeze(1),
            time_signatures.unsqueeze(0)
        ).float()

        # ── Weighted combination ───────────────────────────────────────────
        weights = (
            self.w_genre      * genre_sim
            + self.w_bpm      * bpm_sim
            + self.w_instrument * instr_sim
            + self.w_timesig  * timesig_sim
            + self.w_squareness * sq_sim
        )

        # Self-contrast mask
        logits_mask = torch.ones(B, B, device=device)
        logits_mask.fill_diagonal_(0)
        weights = weights * logits_mask

        # ── Contrastive loss ───────────────────────────────────────────────
        sim = torch.matmul(features, features.T) / self.temperature
        logits_max, _ = sim.max(dim=1, keepdim=True)
        logits = sim - logits_max.detach()

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        weight_sum = weights.sum(1).clamp(min=1e-8)
        mean_log_prob = (weights * log_prob).sum(1) / weight_sum

        return -mean_log_prob.mean()


class AudioPureLoss(nn.Module):
    """
    Contrastive loss driven exclusively by audio-derived signals.

    No genre or instrument labels are used. Similarity is defined by:
      - BPM:         Gaussian similarity  (tempo compatibility)
      - Squareness:  Gaussian similarity  (rhythmic regularity)
      - Time sig:    Binary match         (same rhythmic meter)

    Weights must sum to 1.0.

    Default weights optimised for audio-content similarity + DJ set selection:
        bpm_weight=0.55  squareness_weight=0.30  time_sig_weight=0.15
    """

    def __init__(
        self,
        temperature: float = 0.07,
        bpm_weight: float = 0.55,
        squareness_weight: float = 0.30,
        time_sig_weight: float = 0.15,
        bpm_sigma: float = 10.0,
        squareness_sigma: float = 0.15,
    ):
        super().__init__()
        total = bpm_weight + squareness_weight + time_sig_weight
        assert abs(total - 1.0) < 1e-4, f"Weights must sum to 1.0, got {total:.4f}"
        self.temperature      = temperature
        self.bpm_weight       = bpm_weight
        self.squareness_weight = squareness_weight
        self.time_sig_weight  = time_sig_weight
        self.bpm_sigma        = bpm_sigma
        self.squareness_sigma = squareness_sigma

    @staticmethod
    def _gaussian(values: torch.Tensor, sigma: float) -> torch.Tensor:
        """Pairwise Gaussian similarity for a 1-D signal. Returns [B, B]."""
        diff = values.unsqueeze(1) - values.unsqueeze(0)
        return torch.exp(-diff.pow(2) / (2 * sigma ** 2))

    def forward(
        self,
        features: torch.Tensor,          # [B, D]  L2-normalised embeddings
        bpms: torch.Tensor,              # [B]     BPM float
        time_signatures: torch.Tensor,   # [B]     int (3 or 4)
        squareness_scores: torch.Tensor, # [B]     float in [0, 1]
    ) -> torch.Tensor:
        device = features.device
        B = features.shape[0]

        features         = F.normalize(features, p=2, dim=1)
        bpms             = bpms.float().to(device)
        squareness_scores = squareness_scores.float().to(device)
        time_signatures  = time_signatures.float().to(device)

        # Pairwise similarity matrices [B, B]
        bpm_sim = self._gaussian(bpms,             self.bpm_sigma)
        sq_sim  = self._gaussian(squareness_scores, self.squareness_sigma)
        ts_sim  = torch.eq(
            time_signatures.unsqueeze(1),
            time_signatures.unsqueeze(0),
        ).float()

        weights = (
            self.bpm_weight       * bpm_sim
            + self.squareness_weight * sq_sim
            + self.time_sig_weight   * ts_sim
        )

        # Exclude self-contrast (diagonal)
        logits_mask = torch.ones(B, B, device=device)
        logits_mask.fill_diagonal_(0)
        weights = weights * logits_mask

        # Cosine similarity logits
        sim = torch.matmul(features, features.T) / self.temperature
        logits_max, _ = sim.max(dim=1, keepdim=True)
        logits = sim - logits_max.detach()

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        weight_sum = weights.sum(1).clamp(min=1e-8)
        mean_log_prob = (weights * log_prob).sum(1) / weight_sum

        return -mean_log_prob.mean()


class DualBackboneLoss(nn.Module):
    """
    Contrastive loss for the VGGWhisperDual model.

    Uses available audio-derived signals with genre as structural anchor:

      w_ij = w_genre  · jaccard(genres_i, genres_j)          -- anti-collapse anchor (0.25)
           + w_bpm    · gaussian(|bpm_i - bpm_j|, σ_bpm)     -- tempo proximity    (0.35)
           + w_timesig· (time_sig_i == time_sig_j)            -- meter match        (0.25)
           + w_sq     · gaussian(|sq_i - sq_j|, σ_sq)        -- beat regularity    (0.15)

    The genre term (0.25) acts as a structural regulariser to prevent embedding
    space collapse. The remaining 0.75 weight is split across pure audio signals.

    Notes:
      - time_sig is used as a proxy for danceability (4/4 = more danceable).
      - Replace time_sig with a proper danceability signal once computed in cache.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        w_genre: float = 0.25,
        w_bpm: float = 0.35,
        w_timesig: float = 0.25,
        w_squareness: float = 0.15,
        bpm_sigma: float = 10.0,
        squareness_sigma: float = 0.15,
    ):
        super().__init__()
        total = w_genre + w_bpm + w_timesig + w_squareness
        assert abs(total - 1.0) < 1e-4, f"Weights must sum to 1.0, got {total:.4f}"

        self.temperature = temperature
        self.w_genre = w_genre
        self.w_bpm = w_bpm
        self.w_timesig = w_timesig
        self.w_squareness = w_squareness
        self.bpm_sigma = bpm_sigma
        self.squareness_sigma = squareness_sigma

    @staticmethod
    def _jaccard(vecs: torch.Tensor) -> torch.Tensor:
        inter = torch.matmul(vecs, vecs.T)
        row_sum = vecs.sum(dim=1, keepdim=True)
        union = row_sum + row_sum.T - inter
        return inter / (union + 1e-8)

    @staticmethod
    def _gaussian(values: torch.Tensor, sigma: float) -> torch.Tensor:
        diff = values.unsqueeze(1) - values.unsqueeze(0)
        return torch.exp(-diff.pow(2) / (2 * sigma ** 2))

    def forward(
        self,
        features: torch.Tensor,
        genre_vectors: torch.Tensor,
        bpms: torch.Tensor,
        time_signatures: torch.Tensor,
        squareness_scores: torch.Tensor,
    ) -> torch.Tensor:
        device = features.device
        B = features.shape[0]

        features = F.normalize(features, p=2, dim=1)

        genre_vectors     = genre_vectors.float().to(device)
        bpms              = bpms.to(device)
        time_signatures   = time_signatures.to(device).float()
        squareness_scores = squareness_scores.to(device)

        genre_sim  = self._jaccard(genre_vectors)
        bpm_sim    = self._gaussian(bpms, self.bpm_sigma)
        sq_sim     = self._gaussian(squareness_scores, self.squareness_sigma)
        timesig_sim = torch.eq(
            time_signatures.unsqueeze(1),
            time_signatures.unsqueeze(0)
        ).float()

        weights = (
            self.w_genre        * genre_sim
            + self.w_bpm        * bpm_sim
            + self.w_timesig    * timesig_sim
            + self.w_squareness * sq_sim
        )

        logits_mask = torch.ones(B, B, device=device)
        logits_mask.fill_diagonal_(0)
        weights = weights * logits_mask

        sim = torch.matmul(features, features.T) / self.temperature
        logits_max, _ = sim.max(dim=1, keepdim=True)
        logits = sim - logits_max.detach()

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        weight_sum = weights.sum(1).clamp(min=1e-8)
        mean_log_prob = (weights * log_prob).sum(1) / weight_sum

        return -mean_log_prob.mean()
