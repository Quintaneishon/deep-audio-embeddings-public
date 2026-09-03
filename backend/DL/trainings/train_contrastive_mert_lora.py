"""
Contrastive training for MERT-v1-95M with LoRA adapters + MultiSignalLoss.

Single-phase training — LoRA adapters and projection head train jointly
from epoch 1. No frozen/unfrozen transition needed.

Positive-pair weights:
    w = w_genre * Jaccard(genre) + w_bpm * Gaussian(BPM)
      + w_instrument * Jaccard(instrument) + w_timesig * (time_sig ==)
      + w_squareness * Gaussian(squareness)

Requires:
    pip install peft
    Precomputed features cache (precompute_structure_features.py)
    MTG-Jamendo audio at 24 kHz — set --sample-rate 24000 (default)
"""

import os
import math
import json
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from backend.DL.models.MERTContrastiveLora import MERTContrastiveLora
from backend.DL.trainings.losses import MultiSignalLoss
from backend.DL.trainings.dataset_multisignal import create_dataloaders_multisignal
from backend.DL.trainings.checkpointing import atomic_torch_save
from backend import config


class MERTLoraTrainer:
    """Single-phase contrastive trainer for MERT + LoRA."""

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        device,
        checkpoint_dir,
        log_dir,
        num_epochs=50,
        lr=1e-3,
        weight_decay=1e-4,
        scheduler_type='plateau',
        early_stopping_patience=10,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        self.num_epochs = num_epochs
        self.early_stopping_patience = early_stopping_patience

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir)

        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.train_losses = []
        self.val_losses = []

        self.scaler = torch.amp.GradScaler('cuda')

        self.optimizer = optim.Adam(
            model.get_trainable_parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        self.scheduler = self._make_scheduler(self.optimizer, scheduler_type, num_epochs)

    # ------------------------------------------------------------------ #

    def _make_scheduler(self, optimizer, scheduler_type, num_epochs):
        if scheduler_type == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5)
        if scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=num_epochs, eta_min=1e-6)
        if scheduler_type == 'step':
            return optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        return None

    def _log_gpu_memory(self, epoch):
        if not torch.cuda.is_available():
            return
        idx = self.device.index or 0
        alloc    = torch.cuda.memory_allocated(idx) / 1e9
        reserved = torch.cuda.memory_reserved(idx) / 1e9
        total    = torch.cuda.get_device_properties(idx).total_memory / 1e9
        print(f"  GPU mem: {alloc:.2f}GB alloc / {reserved:.2f}GB reserved / {total:.2f}GB total")
        self.writer.add_scalar('GPU/MemoryAllocatedGB', alloc, epoch)
        self.writer.add_scalar('GPU/MemoryReservedGB', reserved, epoch)

    # ------------------------------------------------------------------ #

    def train_epoch(self):
        self.model.train()
        epoch_loss, num_batches = 0.0, 0

        pbar = tqdm(
            self.train_loader,
            desc=f'Epoch {self.current_epoch + 1}/{self.num_epochs} [MERT-LoRA]',
        )
        for batch_idx, (audio, genre_vecs, instr_vecs, bpms, time_sigs, squareness) in enumerate(pbar):
            audio      = audio.to(self.device)
            genre_vecs = genre_vecs.to(self.device)
            instr_vecs = instr_vecs.to(self.device)
            bpms       = bpms.to(self.device)
            time_sigs  = time_sigs.to(self.device)
            squareness = squareness.to(self.device)

            self.optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                embeddings = self.model(audio)
            loss = self.criterion(
                embeddings.float(), genre_vecs, instr_vecs, bpms, time_sigs, squareness
            )

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            trainable = self.model.get_trainable_parameters()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            batch_loss = loss.item()
            if math.isnan(batch_loss) or math.isinf(batch_loss):
                print(f"\n  NaN/Inf loss at batch {batch_idx} — aborting epoch")
                return float('nan')

            epoch_loss  += batch_loss
            num_batches += 1
            pbar.set_postfix({'loss': f'{batch_loss:.4f}'})

            global_step = self.current_epoch * len(self.train_loader) + batch_idx
            self.writer.add_scalar('Train/BatchLoss', batch_loss, global_step)

        avg = epoch_loss / num_batches
        self.train_losses.append(avg)
        return avg

    def validate(self):
        self.model.eval()
        epoch_loss, num_batches = 0.0, 0

        with torch.no_grad():
            for audio, genre_vecs, instr_vecs, bpms, time_sigs, squareness in tqdm(
                self.val_loader, desc='Validation'
            ):
                audio      = audio.to(self.device)
                genre_vecs = genre_vecs.to(self.device)
                instr_vecs = instr_vecs.to(self.device)
                bpms       = bpms.to(self.device)
                time_sigs  = time_sigs.to(self.device)
                squareness = squareness.to(self.device)

                with torch.amp.autocast('cuda'):
                    embeddings = self.model(audio)
                loss = self.criterion(
                    embeddings.float(), genre_vecs, instr_vecs, bpms, time_sigs, squareness
                )
                epoch_loss  += loss.item()
                num_batches += 1

        avg = epoch_loss / num_batches
        self.val_losses.append(avg)
        return avg

    # ------------------------------------------------------------------ #

    def save_checkpoint(self, filename='checkpoint.pth', is_best=False):
        """Save trainable params only (LoRA adapters + head) — small files."""
        checkpoint = {
            'epoch': self.current_epoch,
            'trainable_state_dict': {
                k: v for k, v in self.model.state_dict().items()
                if any(n == k for n, p in self.model.named_parameters() if p.requires_grad)
            },
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'model_config': {
                'projection_dim': self.model.projection_dim,
                'embedding_dim': self.model.embedding_dim,
                'lora_r': self.model.lora_r,
                'lora_alpha': self.model.lora_alpha,
            },
        }
        atomic_torch_save(checkpoint, self.checkpoint_dir / filename)
        if is_best:
            atomic_torch_save(checkpoint, self.checkpoint_dir / 'best_model.pth')
            print(f"  Best model saved: {self.checkpoint_dir / 'best_model.pth'}")

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)

        # Load only trainable params (LoRA + head) into the model
        model_state = self.model.state_dict()
        model_state.update(checkpoint['trainable_state_dict'])
        self.model.load_state_dict(model_state)

        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint.get('scheduler_state_dict') and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        self.current_epoch = checkpoint['epoch'] + 1
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses  = checkpoint.get('train_losses', [])
        self.val_losses    = checkpoint.get('val_losses', [])
        print(f"Checkpoint loaded: {checkpoint_path}  (epoch {self.current_epoch})")

    # ------------------------------------------------------------------ #

    def train(self):
        w = self.criterion
        print("\n" + "=" * 60)
        print("MERT-v1-95M + LoRA  Contrastive Training")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Trainable params: {self.model.trainable_param_count():,}  "
              f"/ {self.model.total_param_count():,} total")
        print(f"Epochs: {self.num_epochs}  |  Train batches: {len(self.train_loader)}"
              f"  |  Val batches: {len(self.val_loader)}")
        print(f"Loss weights — genre:{w.w_genre} bpm:{w.w_bpm} "
              f"instrument:{w.w_instrument} timesig:{w.w_timesig} "
              f"squareness:{w.w_squareness}")
        print("=" * 60 + "\n")

        for epoch in range(self.current_epoch, self.num_epochs):
            self.current_epoch = epoch

            train_loss = self.train_epoch()
            if math.isnan(train_loss):
                print(f"\nTraining aborted: NaN at epoch {epoch + 1}")
                break

            val_loss = self.validate()

            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            lrs = [pg['lr'] for pg in self.optimizer.param_groups]
            lr_str = ', '.join(f'{lr:.2e}' for lr in lrs)
            self._log_gpu_memory(epoch)
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            print(f"  Train: {train_loss:.4f}  Val: {val_loss:.4f}  LR: {lr_str}")

            self.writer.add_scalar('Train/EpochLoss', train_loss, epoch)
            self.writer.add_scalar('Val/EpochLoss',   val_loss,   epoch)

            self.save_checkpoint(filename=f'checkpoint_epoch_{epoch + 1}.pth')

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint(is_best=True)
                print(f"  New best val loss: {val_loss:.4f}")
            else:
                self.patience_counter += 1
                print(f"  Patience: {self.patience_counter}/{self.early_stopping_patience}")

            if self.patience_counter >= self.early_stopping_patience:
                print(f"\nEarly stopping after {epoch + 1} epochs")
                break
            print()

        print("\n" + "=" * 60)
        print(f"Training Complete! Best val loss: {self.best_val_loss:.4f}")
        print("=" * 60)
        self.writer.close()


# ======================================================================
# CLI
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Train MERT-v1-95M + LoRA with MultiSignalLoss'
    )

    # Dataset
    parser.add_argument('--data-root', type=str, default=config.MTG_JAMENDO_ROOT)
    parser.add_argument('--split', type=str, default='split-0')
    parser.add_argument('--features-cache', type=str, required=True,
                        help='Path to precomputed structural features JSON')
    parser.add_argument('--sample-rate', type=int, default=24000,
                        help='Audio sample rate. MERT requires 24000 Hz (default).')

    # Model
    parser.add_argument('--model-name', type=str, default='m-a-p/MERT-v1-95M')
    parser.add_argument('--projection-dim', type=int,
                        default=config.CONTRASTIVE_TRAINING['projection_dim'])
    parser.add_argument('--cache-dir', type=str, default=None,
                        help='HuggingFace cache directory for MERT weights')

    # LoRA
    parser.add_argument('--lora-r', type=int, default=8,
                        help='LoRA rank (higher = more params, more capacity). Default: 8')
    parser.add_argument('--lora-alpha', type=int, default=16,
                        help='LoRA alpha scaling. Default: 16')
    parser.add_argument('--lora-dropout', type=float, default=0.1)
    parser.add_argument('--lora-targets', type=str, nargs='+',
                        default=['q_proj', 'v_proj'],
                        help='Attention modules to apply LoRA to')

    # Training
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size. 24kHz audio is larger — default 16.')
    parser.add_argument('--num-epochs', type=int,
                        default=config.CONTRASTIVE_TRAINING['num_epochs'])
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--temperature', type=float,
                        default=config.CONTRASTIVE_TRAINING['temperature'])
    parser.add_argument('--num-workers', type=int,
                        default=config.CONTRASTIVE_TRAINING['num_workers'])
    parser.add_argument('--duration', type=float, default=30.0)

    # MultiSignalLoss weights (must sum to 1.0)
    parser.add_argument('--w-genre',      type=float, default=0.15)
    parser.add_argument('--w-bpm',        type=float, default=0.30)
    parser.add_argument('--w-instrument', type=float, default=0.25)
    parser.add_argument('--w-timesig',    type=float, default=0.10)
    parser.add_argument('--w-squareness', type=float, default=0.20)
    parser.add_argument('--bpm-sigma',        type=float, default=10.0)
    parser.add_argument('--squareness-sigma', type=float, default=0.15)

    # Optimisation
    parser.add_argument('--weight-decay', type=float,
                        default=config.CONTRASTIVE_TRAINING['weight_decay'])
    parser.add_argument('--scheduler', type=str,
                        default=config.CONTRASTIVE_TRAINING['scheduler'],
                        choices=['plateau', 'cosine', 'step', 'none'])
    parser.add_argument('--early-stopping', type=int,
                        default=config.CONTRASTIVE_TRAINING['early_stopping_patience'])

    # Infrastructure
    parser.add_argument('--checkpoint-dir', type=str, default=None)
    parser.add_argument('--log-dir', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--cuda-number', type=str, default=None)

    args = parser.parse_args()

    # Validate loss weights
    total_w = args.w_genre + args.w_bpm + args.w_instrument + args.w_timesig + args.w_squareness
    if abs(total_w - 1.0) > 1e-3:
        parser.error(f'Loss weights must sum to 1.0, got {total_w:.4f}')

    # Default dirs
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.checkpoint_dir is None:
        args.checkpoint_dir = os.path.join(
            config.CONTRASTIVE_TRAINING['checkpoint_dir'],
            f'mert_lora_multisignal_{ts}')
    if args.log_dir is None:
        args.log_dir = os.path.join(
            config.CONTRASTIVE_TRAINING['log_dir'],
            f'mert_lora_multisignal_{ts}')

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.cuda_number is not None:
        device = torch.device(f'cuda:{args.cuda_number}')
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(device.index)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(device.index).total_memory / 1e9:.2f} GB\n")

    # Paths
    data_root = Path(args.data_root).expanduser()
    splits_dir = data_root / 'data' / 'splits' / args.split
    audio_dir  = data_root / 'songs'
    train_tsv  = splits_dir / 'autotagging_genre-train.tsv'
    val_tsv    = splits_dir / 'autotagging_genre-validation.tsv'
    test_tsv   = splits_dir / 'autotagging_genre-test.tsv'

    for p in [train_tsv, val_tsv, test_tsv, audio_dir]:
        if not p.exists():
            raise FileNotFoundError(f"Required path not found: {p}")
    if not Path(args.features_cache).exists():
        raise FileNotFoundError(
            f"Features cache not found: {args.features_cache}\n"
            "Run precompute_structure_features.py first.")

    # Dataloaders (sample_rate=24000 for MERT)
    print(f"Creating dataloaders (batch={args.batch_size}, sr={args.sample_rate})...")
    train_loader, val_loader, _, dataset_info = create_dataloaders_multisignal(
        train_tsv=str(train_tsv),
        val_tsv=str(val_tsv),
        test_tsv=str(test_tsv),
        audio_dir=str(audio_dir),
        features_cache_path=args.features_cache,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sample_rate=args.sample_rate,
        duration=args.duration,
    )
    print(f"\nDataset: {dataset_info['train_size']} train | "
          f"{dataset_info['val_size']} val | "
          f"{dataset_info['num_genres']} genres | "
          f"{dataset_info['num_instruments']} instruments")

    # Save dataset info
    info_path = Path(args.checkpoint_dir) / 'dataset_info.json'
    info_path.parent.mkdir(parents=True, exist_ok=True)
    with open(info_path, 'w') as f:
        json.dump({k: v for k, v in dataset_info.items()
                   if k not in ('genre_to_idx', 'instrument_to_idx')}, f, indent=2)

    # Model
    model = MERTContrastiveLora(
        model_name=args.model_name,
        projection_dim=args.projection_dim,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_targets,
        device=device,
        cache_dir=args.cache_dir,
    )

    # Loss
    criterion = MultiSignalLoss(
        temperature=args.temperature,
        w_genre=args.w_genre,
        w_bpm=args.w_bpm,
        w_instrument=args.w_instrument,
        w_timesig=args.w_timesig,
        w_squareness=args.w_squareness,
        bpm_sigma=args.bpm_sigma,
        squareness_sigma=args.squareness_sigma,
    )

    # Trainer
    trainer = MERTLoraTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        num_epochs=args.num_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        scheduler_type=args.scheduler,
        early_stopping_patience=args.early_stopping,
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)

    config_path = Path(args.checkpoint_dir) / 'training_config.json'
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)

    trainer.train()

    print(f"\nCheckpoints: {args.checkpoint_dir}")
    print(f"TensorBoard: tensorboard --logdir {args.log_dir}")


if __name__ == '__main__':
    main()
