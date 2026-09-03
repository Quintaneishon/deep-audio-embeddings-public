"""
Two-phase contrastive training for MusiCNN (MSD) with MultiSignalLoss.

Positive-pair weights combine five structural signals:
    w = w_genre * Jaccard(genre) + w_bpm * Gaussian(BPM)
      + w_instrument * Jaccard(instrument) + w_timesig * (time_sig ==)
      + w_squareness * Gaussian(squareness)

Phase 1 (warm-up): backbone frozen, projection head only.
Phase 2 (fine-tune): backbone unfrozen with lower LR.

Requires a precomputed structural features cache produced by:
    python precompute_structure_features.py --tsv ... --audio ... --output features_cache.json
"""

import os
import argparse
import json
import math
from pathlib import Path
from datetime import datetime

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from backend.DL.models.MusiCNNContrastive import MusiCNNContrastive
from backend.DL.trainings.losses import MultiSignalLoss
from backend.DL.trainings.dataset_multisignal import create_dataloaders_multisignal
from backend.DL.trainings.checkpointing import atomic_torch_save
from backend import config


class MultiSignalTrainer:
    """Two-phase trainer for MusiCNN + MultiSignalLoss."""

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
        warmup_epochs=10,
        head_lr=1e-3,
        backbone_lr=1e-5,
        weight_decay=1e-4,
        scheduler_type='plateau',
        early_stopping_patience=10,
        phase2_train_loader=None,
    ):
        self.model = model
        self.val_loader = val_loader
        self.criterion = criterion
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        self.num_epochs = num_epochs
        self.warmup_epochs = warmup_epochs
        self.head_lr = head_lr
        self.backbone_lr = backbone_lr
        self.weight_decay = weight_decay
        self.scheduler_type = scheduler_type
        self.early_stopping_patience = early_stopping_patience

        # Phase-specific loaders (phase 2 may use smaller batch to fit VRAM)
        self.phase1_train_loader = train_loader
        self.phase2_train_loader = phase2_train_loader or train_loader
        self.train_loader = self.phase1_train_loader

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir)

        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.train_losses = []
        self.val_losses = []
        self.current_phase = 1

        self.scaler = torch.amp.GradScaler('cuda')
        self._init_phase1()

    # ------------------------------------------------------------------
    # Phase management
    # ------------------------------------------------------------------

    def _init_phase1(self):
        self.current_phase = 1
        self.train_loader = self.phase1_train_loader
        self.model._freeze_backbone()
        self.optimizer = optim.Adam(
            self.model.get_trainable_parameters(),
            lr=self.head_lr,
            weight_decay=self.weight_decay,
        )
        self.scheduler = self._make_scheduler(self.optimizer)
        print(f"[Phase 1] Backbone frozen — projection head only | "
              f"batch={self.train_loader.batch_size}")

    def _init_phase2(self):
        self.current_phase = 2
        self.train_loader = self.phase2_train_loader
        self.model._unfreeze_backbone()
        # Gradient checkpointing: saves ~3-4x VRAM during unfrozen phase
        # at the cost of ~30% slower backward pass. Mandatory for safety.
        self.model.enable_grad_checkpointing()

        del self.optimizer
        if self.scheduler:
            del self.scheduler
        torch.cuda.empty_cache()

        param_groups = self.model.get_all_parameters_grouped(
            backbone_lr=self.backbone_lr,
            head_lr=self.head_lr,
        )
        for pg in param_groups:
            pg['weight_decay'] = self.weight_decay

        self.optimizer = optim.Adam(param_groups)
        self.scheduler = self._make_scheduler(self.optimizer)
        self.patience_counter = 0
        print(f"[Phase 2] Backbone unfrozen — backbone LR={self.backbone_lr} "
              f"head LR={self.head_lr} | batch={self.train_loader.batch_size}")

    def _make_scheduler(self, optimizer):
        if self.scheduler_type == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5)
        if self.scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.num_epochs, eta_min=1e-6)
        if self.scheduler_type == 'step':
            return optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        return None

    def _gpu_memory_info(self):
        if not torch.cuda.is_available():
            return None, None, None
        idx = self.device.index or 0
        alloc    = torch.cuda.memory_allocated(idx) / 1e9
        reserved = torch.cuda.memory_reserved(idx) / 1e9
        total    = torch.cuda.get_device_properties(idx).total_memory / 1e9
        return alloc, reserved, total

    def _log_gpu_memory(self, epoch):
        alloc, reserved, total = self._gpu_memory_info()
        if total is None:
            return
        pct = reserved / total * 100
        flag = '  ⚠️  HIGH' if pct > 85 else ''
        print(f"  GPU mem: {alloc:.2f}GB alloc / {reserved:.2f}GB reserved / {total:.2f}GB total  ({pct:.0f}%){flag}")
        self.writer.add_scalar('GPU/MemoryAllocatedGB', alloc, epoch)
        self.writer.add_scalar('GPU/MemoryReservedGB', reserved, epoch)
        self.writer.add_scalar('GPU/MemoryUsedPct', pct, epoch)

    # ------------------------------------------------------------------
    # Train / validate
    # ------------------------------------------------------------------

    def train_epoch(self):
        self.model.train()
        epoch_loss  = 0.0
        num_batches = 0
        oom_skips   = 0

        pbar = tqdm(
            self.train_loader,
            desc=f'Epoch {self.current_epoch + 1}/{self.num_epochs} [P{self.current_phase}]',
        )
        for batch_idx, (audio, genre_vecs, instr_vecs, bpms, time_sigs, squareness) in enumerate(pbar):
            try:
                audio      = audio.to(self.device)
                genre_vecs = genre_vecs.to(self.device)
                instr_vecs = instr_vecs.to(self.device)
                bpms       = bpms.to(self.device)
                time_sigs  = time_sigs.to(self.device)
                squareness = squareness.to(self.device)

                self.optimizer.zero_grad(set_to_none=True)  # frees grad memory faster

                with torch.amp.autocast('cuda'):
                    embeddings = self.model(audio)
                loss = self.criterion(
                    embeddings.float(), genre_vecs, instr_vecs, bpms, time_sigs, squareness
                )

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                trainable = [p for p in self.model.parameters() if p.requires_grad]
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                batch_loss = loss.item()

                # Explicit cleanup before next batch
                del audio, genre_vecs, instr_vecs, bpms, time_sigs, squareness
                del embeddings, loss

                if math.isnan(batch_loss) or math.isinf(batch_loss):
                    print(f"\n  NaN/Inf loss at batch {batch_idx} — aborting epoch")
                    return float('nan')

                epoch_loss  += batch_loss
                num_batches += 1
                global_step  = self.current_epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Train/BatchLoss', batch_loss, global_step)

                # Periodic cache flush + memory warning
                if batch_idx % 50 == 0:
                    torch.cuda.empty_cache()
                    _, reserved, total = self._gpu_memory_info()
                    if total and reserved / total > 0.90:
                        print(f"\n  ⚠️  GPU memory at {reserved/total*100:.0f}% — "
                              f"consider reducing --batch-size or --duration")
                    pbar.set_postfix({
                        'loss': f'{batch_loss:.4f}',
                        'gpu%': f'{reserved/total*100:.0f}%' if total else '?',
                        'oom_skip': oom_skips,
                    })
                else:
                    pbar.set_postfix({'loss': f'{batch_loss:.4f}', 'oom_skip': oom_skips})

            except torch.cuda.OutOfMemoryError:
                # OOM: skip this batch instead of crashing the server
                oom_skips += 1
                self.optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                print(f"\n  ⚠️  OOM at batch {batch_idx} (skipped, total skips={oom_skips}). "
                      f"Reduce --batch-size or --duration if this keeps happening.")
                if oom_skips >= 5:
                    print("  ✋  Too many OOM skips — aborting epoch to protect the server.")
                    return float('nan')

        if num_batches == 0:
            return float('nan')
        avg_loss = epoch_loss / num_batches
        self.train_losses.append(avg_loss)
        if oom_skips:
            print(f"  Epoch finished with {oom_skips} OOM-skipped batches out of "
                  f"{num_batches + oom_skips} total.")
        return avg_loss

    def validate(self):
        self.model.eval()
        epoch_loss  = 0.0
        num_batches = 0

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
                epoch_loss += loss.item()
                num_batches += 1
                del audio, genre_vecs, instr_vecs, bpms, time_sigs, squareness
                del embeddings, loss

        torch.cuda.empty_cache()
        avg_loss = epoch_loss / num_batches
        self.val_losses.append(avg_loss)
        return avg_loss

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, filename='checkpoint.pth', is_best=False):
        checkpoint = {
            'epoch': self.current_epoch,
            'phase': self.current_phase,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'model_config': {
                'dataset': self.model.dataset,
                'projection_dim': self.model.projection_dim,
                'embedding_dim': self.model.embedding_dim,
            },
        }
        atomic_torch_save(checkpoint, self.checkpoint_dir / filename)
        if is_best:
            atomic_torch_save(checkpoint, self.checkpoint_dir / 'best_model.pth')
            print(f"  Best model saved: {self.checkpoint_dir / 'best_model.pth'}")

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        saved_phase = checkpoint.get('phase', 1)
        if saved_phase == 2 and self.current_phase == 1:
            self._init_phase2()

        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint.get('scheduler_state_dict') and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        self.current_epoch = checkpoint['epoch'] + 1
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        print(f"Checkpoint loaded: {checkpoint_path}  "
              f"(epoch {self.current_epoch}, phase {self.current_phase})")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def train(self):
        print("\n" + "=" * 60)
        print("MusiCNN MultiSignalLoss Contrastive Training (Two-Phase)")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Total epochs: {self.num_epochs}  |  Warm-up: {self.warmup_epochs}")
        print(f"Phase-1 train batches: {len(self.phase1_train_loader)} "
              f"(bs={self.phase1_train_loader.batch_size})")
        print(f"Phase-2 train batches: {len(self.phase2_train_loader)} "
              f"(bs={self.phase2_train_loader.batch_size})")
        w = self.criterion
        print(f"Loss weights — genre:{w.w_genre} bpm:{w.w_bpm} "
              f"instrument:{w.w_instrument} timesig:{w.w_timesig} "
              f"squareness:{w.w_squareness}")
        print("=" * 60 + "\n")

        for epoch in range(self.current_epoch, self.num_epochs):
            self.current_epoch = epoch

            if self.current_phase == 1 and epoch >= self.warmup_epochs:
                self._init_phase2()

            train_loss = self.train_epoch()
            if math.isnan(train_loss):
                print(f"\nTraining aborted: NaN loss at epoch {epoch + 1}")
                break

            val_loss = self.validate()
            torch.cuda.empty_cache()

            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            lrs = [pg['lr'] for pg in self.optimizer.param_groups]
            lr_str = ', '.join(f'{lr:.2e}' for lr in lrs)
            self._log_gpu_memory(epoch)
            print(f"\nEpoch {epoch + 1}/{self.num_epochs} [Phase {self.current_phase}]")
            print(f"  Train: {train_loss:.4f}  Val: {val_loss:.4f}  LR: {lr_str}")

            self.writer.add_scalar('Train/EpochLoss', train_loss, epoch)
            self.writer.add_scalar('Val/EpochLoss', val_loss, epoch)
            for i, lr in enumerate(lrs):
                self.writer.add_scalar(f'Train/LR_group{i}', lr, epoch)

            self.save_checkpoint(filename=f'checkpoint_epoch_{epoch + 1}.pth')

            is_best = val_loss < self.best_val_loss
            if is_best:
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
        description='Train MusiCNN with MultiSignalLoss (genre+BPM+instrument+timesig+squareness)'
    )

    # Dataset
    parser.add_argument('--data-root', type=str, default=config.MTG_JAMENDO_ROOT)
    parser.add_argument('--split', type=str, default='split-0')
    parser.add_argument('--features-cache', type=str, required=True,
                        help='Path to precomputed structural features JSON '
                             '(output of precompute_structure_features.py)')

    # Model
    parser.add_argument('--pretrained-weights', type=str, required=True,
                        help='Path to pretrained MusiCNN MSD weights (.pth)')
    parser.add_argument('--projection-dim', type=int,
                        default=config.CONTRASTIVE_TRAINING['projection_dim'])

    # Training
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for phase 1 (frozen backbone). Default: 16. '
                             'Lower to 8 if GPU < 10 GB.')
    parser.add_argument('--phase2-batch-size', type=int, default=8,
                        help='Batch size for phase 2 (unfrozen backbone). Default: 8. '
                             'Lower to 4 if OOM warnings appear.')
    parser.add_argument('--num-epochs', type=int,
                        default=config.CONTRASTIVE_TRAINING['num_epochs'])
    parser.add_argument('--warmup-epochs', type=int, default=10)
    parser.add_argument('--lr', type=float,
                        default=config.CONTRASTIVE_TRAINING['learning_rate'])
    parser.add_argument('--backbone-lr', type=float, default=1e-5)
    parser.add_argument('--temperature', type=float,
                        default=config.CONTRASTIVE_TRAINING['temperature'])
    parser.add_argument('--num-workers', type=int, default=2,
                        help='DataLoader workers. Default: 2. Each worker prefetches '
                             '2 batches — with 4 workers that is 8 batches pinned in RAM.')
    parser.add_argument('--duration', type=float, default=10.0,
                        help='Audio segment length in seconds. Default: 10.0. '
                             'The [B, 2097, T] backbone tensor scales linearly with T: '
                             '10 s → T≈625 (~3x less VRAM than 30 s). '
                             'Use 15 s max if GPU >= 16 GB.')

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

    # Checkpointing
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
            f'musicnn_multisignal_{ts}')
    if args.log_dir is None:
        args.log_dir = os.path.join(
            config.CONTRASTIVE_TRAINING['log_dir'],
            f'musicnn_multisignal_{ts}')

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

    for path in [train_tsv, val_tsv, test_tsv, audio_dir]:
        if not path.exists():
            raise FileNotFoundError(f"Required path not found: {path}")
    if not Path(args.features_cache).exists():
        raise FileNotFoundError(
            f"Features cache not found: {args.features_cache}\n"
            "Run precompute_structure_features.py first.")

    # Dataloaders — phase 1 (larger batch) and phase 2 (smaller batch)
    print(f"Creating phase-1 dataloaders (batch={args.batch_size})...")
    p1_train, val_loader, _, dataset_info = create_dataloaders_multisignal(
        train_tsv=str(train_tsv), val_tsv=str(val_tsv), test_tsv=str(test_tsv),
        audio_dir=str(audio_dir),
        features_cache_path=args.features_cache,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sample_rate=config.SAMPLE_RATE,
        duration=args.duration,
    )

    if args.phase2_batch_size != args.batch_size:
        print(f"Creating phase-2 dataloaders (batch={args.phase2_batch_size})...")
        p2_train, _, _, _ = create_dataloaders_multisignal(
            train_tsv=str(train_tsv), val_tsv=str(val_tsv), test_tsv=str(test_tsv),
            audio_dir=str(audio_dir),
            features_cache_path=args.features_cache,
            batch_size=args.phase2_batch_size,
            num_workers=args.num_workers,
            sample_rate=config.SAMPLE_RATE,
            duration=args.duration,
        )
    else:
        p2_train = p1_train

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
    print("\nInitializing MusiCNNContrastive...")
    model = MusiCNNContrastive(
        projection_dim=args.projection_dim,
        pretrained_weights=args.pretrained_weights,
        device=device,
    )
    total   = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total:,}  |  Phase-1 trainable: {trainable:,}")

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
    trainer = MultiSignalTrainer(
        model=model,
        train_loader=p1_train,
        val_loader=val_loader,
        criterion=criterion,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        num_epochs=args.num_epochs,
        warmup_epochs=args.warmup_epochs,
        head_lr=args.lr,
        backbone_lr=args.backbone_lr,
        weight_decay=args.weight_decay,
        scheduler_type=args.scheduler,
        early_stopping_patience=args.early_stopping,
        phase2_train_loader=p2_train,
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
