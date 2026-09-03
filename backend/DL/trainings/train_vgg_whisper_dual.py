"""
VGG + Whisper Dual-Backbone Contrastive Training
=================================================
Trains a joint VGG_Res + Whisper encoder model with a hybrid loss that uses
genre as a structural anchor (prevents collapse) while weighting pure audio
signals more heavily.

Loss weights:
  Genre (Jaccard)      0.25  -- anti-collapse anchor, not the objective
  BPM (Gaussian)       0.35  -- tempo compatibility
  Time sig (binary)    0.25  -- meter match (proxy for danceability)
  Squareness (Gauss)   0.15  -- beat regularity

The goal: an embedding space where genre is a *consequence* of proximity,
not the *cause*. Similar-sounding songs cluster together regardless of label.

Phase 1 (warm-up): both backbones frozen, only projection head trains.
Phase 2 (fine-tune): both backbones unfrozen with differential LR.

Usage:
    cd /path/to/deep-audio-embeddings   # project root, NOT backend/
    source .venv/bin/activate
    python -m backend.DL.trainings.train_vgg_whisper_dual \\
        --cuda-number 1 \\
        --batch-size 8 \\
        --warmup-epochs 10 \\
        --finetune-epochs 40

Note: batch_size 8 recommended due to Whisper's memory footprint (~2x VGG alone).
"""

import os
import argparse
import json
import math
from pathlib import Path
from datetime import datetime

import torch
import torch.optim as optim
from backend.DL.trainings.checkpointing import atomic_torch_save
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from backend.DL.models.VGGWhisperDual import VGGWhisperDual
from backend.DL.trainings.losses import DualBackboneLoss
from backend.DL.trainings.dataset_multisignal import create_dataloaders_multisignal
from backend import config


# ======================================================================
# Trainer
# ======================================================================

class VGGWhisperDualTrainer:
    """Two-phase trainer for the VGG+Whisper dual-backbone contrastive model."""

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        device,
        checkpoint_dir,
        log_dir,
        warmup_epochs=10,
        finetune_epochs=40,
        head_lr=1e-3,
        backbone_lr=1e-5,
        weight_decay=1e-4,
        scheduler_type="plateau",
        early_stopping_patience=8,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        self.warmup_epochs = warmup_epochs
        self.finetune_epochs = finetune_epochs
        self.num_epochs = warmup_epochs + finetune_epochs
        self.head_lr = head_lr
        self.backbone_lr = backbone_lr
        self.weight_decay = weight_decay
        self.scheduler_type = scheduler_type
        self.early_stopping_patience = early_stopping_patience

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.train_losses = []
        self.val_losses = []
        self.current_phase = 1

        self._init_phase1()

    # ------------------------------------------------------------------
    # Phase management
    # ------------------------------------------------------------------

    def _init_phase1(self):
        self.current_phase = 1
        self.model._freeze_backbones()
        self.optimizer = optim.Adam(
            self.model.get_trainable_parameters(),
            lr=self.head_lr,
            weight_decay=self.weight_decay,
        )
        self.scheduler = self._make_scheduler(self.optimizer)
        print("[Phase 1] Both backbones frozen — training projection head only")

    def _init_phase2(self):
        self.current_phase = 2
        self.model._unfreeze_backbones()
        del self.optimizer
        if self.scheduler:
            del self.scheduler
        torch.cuda.empty_cache()

        param_groups = self.model.get_all_parameters_grouped(
            backbone_lr=self.backbone_lr,
            head_lr=self.head_lr,
        )
        for pg in param_groups:
            pg["weight_decay"] = self.weight_decay

        self.optimizer = optim.Adam(param_groups)
        self.scheduler = self._make_scheduler(self.optimizer)
        self.patience_counter = 0
        print(
            f"[Phase 2] Backbones unfrozen — "
            f"backbone_lr={self.backbone_lr}, head_lr={self.head_lr}"
        )

    def _make_scheduler(self, optimizer):
        if self.scheduler_type == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=5
            )
        if self.scheduler_type == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.num_epochs, eta_min=1e-6
            )
        if self.scheduler_type == "step":
            return optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        return None

    # ------------------------------------------------------------------
    # GPU memory logging
    # ------------------------------------------------------------------

    def _log_gpu_memory(self, epoch):
        if not torch.cuda.is_available():
            return
        idx = self.device.index or 0
        alloc    = torch.cuda.memory_allocated(idx) / 1e9
        reserved = torch.cuda.memory_reserved(idx) / 1e9
        total    = torch.cuda.get_device_properties(idx).total_memory / 1e9
        print(f"  GPU mem: {alloc:.2f}GB alloc / {reserved:.2f}GB reserved / {total:.2f}GB total")
        self.writer.add_scalar("GPU/MemoryAllocatedGB", alloc, epoch)
        self.writer.add_scalar("GPU/MemoryReservedGB",  reserved, epoch)

    # ------------------------------------------------------------------
    # Train / validate
    # ------------------------------------------------------------------

    def train_epoch(self):
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch + 1}/{self.num_epochs} [P{self.current_phase}]",
        )
        # Batch: (audio, genre_vec, instrument_vec, bpm, time_sig, squareness)
        for batch_idx, (audio, genre, _instr, bpm, time_sig, squareness) in enumerate(pbar):
            audio      = audio.to(self.device)
            genre      = genre.to(self.device)
            bpm        = bpm.to(self.device)
            time_sig   = time_sig.to(self.device)
            squareness = squareness.to(self.device)

            self.optimizer.zero_grad()
            embeddings = self.model(audio)
            loss = self.criterion(embeddings, genre, bpm, time_sig, squareness)

            batch_loss = loss.item()
            if math.isnan(batch_loss) or math.isinf(batch_loss):
                print(f"\n  NaN/Inf loss at batch {batch_idx} — aborting epoch")
                return float("nan")

            loss.backward()
            trainable = [p for p in self.model.parameters() if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            self.optimizer.step()

            epoch_loss += batch_loss
            num_batches += 1
            pbar.set_postfix({"loss": f"{batch_loss:.4f}"})

            global_step = self.current_epoch * len(self.train_loader) + batch_idx
            self.writer.add_scalar("Train/BatchLoss", batch_loss, global_step)

        avg_loss = epoch_loss / num_batches
        self.train_losses.append(avg_loss)
        return avg_loss

    def validate(self):
        self.model.eval()
        epoch_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for audio, genre, _instr, bpm, time_sig, squareness in tqdm(
                self.val_loader, desc="Validation"
            ):
                audio      = audio.to(self.device)
                genre      = genre.to(self.device)
                bpm        = bpm.to(self.device)
                time_sig   = time_sig.to(self.device)
                squareness = squareness.to(self.device)

                embeddings = self.model(audio)
                loss = self.criterion(embeddings, genre, bpm, time_sig, squareness)

                epoch_loss += loss.item()
                num_batches += 1

        avg_loss = epoch_loss / num_batches
        self.val_losses.append(avg_loss)
        return avg_loss

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, filename="checkpoint.pth", is_best=False):
        checkpoint = {
            "epoch":                self.current_epoch,
            "phase":                self.current_phase,
            "model_state_dict":     self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "best_val_loss":        self.best_val_loss,
            "train_losses":         self.train_losses,
            "val_losses":           self.val_losses,
            "model_config": {
                "projection_dim": self.model.projection_dim,
                "whisper_size":   self.model.whisper_size,
                "combined_dim":   self.model.COMBINED_DIM,
            },
        }
        atomic_torch_save(checkpoint, self.checkpoint_dir / filename)
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            atomic_torch_save(checkpoint, best_path)
            print(f"  Best model saved: {best_path}")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def train(self):
        print("\n" + "=" * 60)
        print("VGG + Whisper Dual-Backbone Contrastive Training")
        print("=" * 60)
        print(f"Device:           {self.device}")
        print(f"Warm-up epochs:   {self.warmup_epochs}")
        print(f"Fine-tune epochs: {self.finetune_epochs}")
        print(f"Train batches:    {len(self.train_loader)}")
        print(f"Val batches:      {len(self.val_loader)}")
        total     = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Parameters:       {trainable:,} trainable / {total:,} total (phase 1)")
        print("=" * 60 + "\n")

        for epoch in range(self.current_epoch, self.num_epochs):
            self.current_epoch = epoch

            if self.current_phase == 1 and epoch >= self.warmup_epochs:
                self._init_phase2()

            train_loss = self.train_epoch()
            if math.isnan(train_loss):
                print("Training aborted due to NaN loss.")
                break

            val_loss = self.validate()

            print(
                f"Epoch {epoch + 1}/{self.num_epochs} "
                f"[P{self.current_phase}]  "
                f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}"
            )
            self._log_gpu_memory(epoch)

            self.writer.add_scalar("Loss/Train", train_loss, epoch)
            self.writer.add_scalar("Loss/Val",   val_loss,   epoch)

            if self.scheduler:
                if isinstance(
                    self.scheduler,
                    optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            self.save_checkpoint(
                filename=f"epoch_{epoch + 1:03d}.pth",
                is_best=is_best,
            )

            if self.patience_counter >= self.early_stopping_patience:
                print(f"\nEarly stopping after {epoch + 1} epochs")
                break

            print()

        print("\n" + "=" * 60)
        print("Training Complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Checkpoint dir: {self.checkpoint_dir}")
        print("=" * 60 + "\n")
        self.writer.close()


# ======================================================================
# CLI entry-point
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train VGG+Whisper dual-backbone contrastive model"
    )

    # Dataset
    parser.add_argument("--data-root", type=str, default=config.MTG_JAMENDO_ROOT)
    parser.add_argument("--split", type=str, default="split-0")
    parser.add_argument(
        "--structure-cache",
        type=str,
        default=str(
            Path(__file__).resolve().parents[1] / "cache" / "structure_features_cache.json"
        ),
        help="Path to precomputed structure features cache (JSON)",
    )

    # Model
    parser.add_argument(
        "--vgg-pretrained-weights",
        type=str,
        default=config.MODEL_WEIGHTS["vgg"]["msd"],
        help="Path to pretrained VGG MSD weights (.pth) for backbone initialisation",
    )
    parser.add_argument(
        "--whisper-size",
        type=str,
        default="base",
        choices=["tiny", "base", "small"],
        help="Whisper model size (base recommended; small uses more memory)",
    )
    parser.add_argument("--projection-dim", type=int, default=config.N_DIM)

    # Training phases
    parser.add_argument("--warmup-epochs",   type=int,   default=10,
                        help="Epochs with frozen backbones (phase 1)")
    parser.add_argument("--finetune-epochs", type=int,   default=40,
                        help="Epochs with unfrozen backbones (phase 2)")
    parser.add_argument("--lr",              type=float, default=1e-3,
                        help="Learning rate for projection head")
    parser.add_argument("--backbone-lr",     type=float, default=1e-5,
                        help="Learning rate for backbones in phase 2")
    parser.add_argument("--weight-decay",    type=float, default=1e-4)
    parser.add_argument("--early-stopping",  type=int,   default=8)
    parser.add_argument("--scheduler",       type=str,   default="plateau",
                        choices=["plateau", "cosine", "step", "none"])

    # Loss weights (must sum to 1.0)
    parser.add_argument("--w-genre",      type=float, default=0.25,
                        help="Genre Jaccard weight (anti-collapse anchor)")
    parser.add_argument("--w-bpm",        type=float, default=0.35,
                        help="BPM Gaussian weight")
    parser.add_argument("--w-timesig",    type=float, default=0.25,
                        help="Time signature binary weight (danceability proxy)")
    parser.add_argument("--w-squareness", type=float, default=0.15,
                        help="Squareness Gaussian weight")
    parser.add_argument("--temperature",  type=float, default=0.07)
    parser.add_argument("--bpm-sigma",    type=float, default=10.0)
    parser.add_argument("--sq-sigma",     type=float, default=0.15)

    # Data loading
    parser.add_argument("--batch-size",   type=int,   default=8,
                        help="Batch size (8 recommended; Whisper doubles memory vs VGG-only)")
    parser.add_argument("--num-workers",  type=int,   default=4)
    parser.add_argument("--duration",     type=float, default=30.0)

    # Output
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--log-dir",        type=str, default=None)
    parser.add_argument("--resume",         type=str, default=None)
    parser.add_argument("--cuda-number",    type=str, default=None)

    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name  = f"vgg_whisper_dual_{timestamp}"

    if args.checkpoint_dir is None:
        args.checkpoint_dir = os.path.join(
            config.CONTRASTIVE_TRAINING["checkpoint_dir"], run_name
        )
    if args.log_dir is None:
        args.log_dir = os.path.join(
            config.CONTRASTIVE_TRAINING["log_dir"], run_name
        )

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.cuda_number is not None:
        device = torch.device(f"cuda:{args.cuda_number}")
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        idx = device.index if device.index is not None else 0
        print(f"GPU: {torch.cuda.get_device_name(idx)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(idx).total_memory / 1e9:.1f} GB\n")

    # Validate loss weights
    total_w = args.w_genre + args.w_bpm + args.w_timesig + args.w_squareness
    if abs(total_w - 1.0) > 1e-4:
        parser.error(
            f"Loss weights must sum to 1.0, got {total_w:.4f} "
            f"(genre={args.w_genre} bpm={args.w_bpm} "
            f"timesig={args.w_timesig} sq={args.w_squareness})"
        )

    # Paths
    data_root = Path(args.data_root).expanduser()
    splits_dir = data_root / 'data' / 'splits' / args.split
    audio_dir  = data_root / 'songs'
    cache_path = Path(args.structure_cache)

    for path in [splits_dir, audio_dir, cache_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required path not found: {path}")

    train_tsv = splits_dir / "autotagging_genre-train.tsv"
    val_tsv   = splits_dir / "autotagging_genre-validation.tsv"
    test_tsv  = splits_dir / "autotagging_genre-test.tsv"

    # Dataloaders
    print("Creating dataloaders (multi-signal, 16kHz)...")
    train_loader, val_loader, _test_loader, dataset_info = create_dataloaders_multisignal(
        train_tsv=str(train_tsv),
        val_tsv=str(val_tsv),
        test_tsv=str(test_tsv),
        audio_dir=str(audio_dir),
        features_cache_path=str(cache_path),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sample_rate=config.SAMPLE_RATE,
        duration=args.duration,
    )
    print(f"\nDataset sizes:")
    print(f"  Train: {dataset_info['train_size']} | Val: {dataset_info['val_size']} | Test: {dataset_info['test_size']}")

    # Save training config
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.checkpoint_dir) / "training_config.json", "w") as f:
        json.dump(vars(args), f, indent=2, default=str)

    # Model
    print(f"\nInitialising VGGWhisperDual (VGG-MSD + Whisper-{args.whisper_size})...")
    model = VGGWhisperDual(
        projection_dim=args.projection_dim,
        whisper_size=args.whisper_size,
        vgg_pretrained_weights=args.vgg_pretrained_weights,
        device=device,
    )
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters:    {total:,}")
    print(f"  Trainable (phase 1): {trainable:,}")

    # Loss
    criterion = DualBackboneLoss(
        temperature=args.temperature,
        w_genre=args.w_genre,
        w_bpm=args.w_bpm,
        w_timesig=args.w_timesig,
        w_squareness=args.w_squareness,
        bpm_sigma=args.bpm_sigma,
        squareness_sigma=args.sq_sigma,
    )
    print(f"\nDualBackboneLoss:")
    print(f"  genre={args.w_genre}  bpm={args.w_bpm}  "
          f"timesig={args.w_timesig}  squareness={args.w_squareness}")
    print(f"  temperature={args.temperature}  bpm_sigma={args.bpm_sigma}  "
          f"sq_sigma={args.sq_sigma}")

    # Trainer
    trainer = VGGWhisperDualTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        warmup_epochs=args.warmup_epochs,
        finetune_epochs=args.finetune_epochs,
        head_lr=args.lr,
        backbone_lr=args.backbone_lr,
        weight_decay=args.weight_decay,
        scheduler_type=args.scheduler,
        early_stopping_patience=args.early_stopping,
    )

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Resumed from: {args.resume}")

    trainer.train()

    print(f"\nCheckpoints: {args.checkpoint_dir}")
    print(f"TensorBoard: tensorboard --logdir {args.log_dir}")


if __name__ == "__main__":
    main()
