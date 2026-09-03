"""
Two-phase contrastive training for MusiCNN (MSD).

Phase 1 (warm-up): backbone frozen, only projection head trains.
Phase 2 (fine-tune): backbone unfrozen with lower LR, projection head keeps higher LR.
"""

import os
import argparse
import json
from pathlib import Path
from datetime import datetime

import math

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from backend.DL.models.MusiCNNContrastive import MusiCNNContrastive
from backend.DL.trainings.losses import SupConLoss
from backend.DL.trainings.dataset import create_dataloaders
from backend.DL.trainings.checkpointing import atomic_torch_save
from backend import config


class ContrastiveTrainer:
    """Two-phase trainer for MusiCNN contrastive learning."""

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
    ):
        self.model = model
        self.train_loader = train_loader
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

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir)

        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.train_losses = []
        self.val_losses = []
        self.current_phase = 1

        self._init_phase1()

    # ------------------------------------------------------------------
    # Phase management
    # ------------------------------------------------------------------

    def _init_phase1(self):
        """Frozen backbone, train projection head only."""
        self.current_phase = 1
        self.model._freeze_backbone()

        self.optimizer = optim.Adam(
            self.model.get_trainable_parameters(),
            lr=self.head_lr,
            weight_decay=self.weight_decay,
        )
        self.scheduler = self._make_scheduler(self.optimizer)
        print("[Phase 1] Backbone frozen – training projection head only")

    def _init_phase2(self):
        """Unfreeze backbone with differential LR."""
        self.current_phase = 2
        self.model._unfreeze_backbone()

        # Release Phase-1 optimizer state and any cached allocations before
        # the backbone gradients roughly double VRAM usage.
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
        print(f"[Phase 2] Backbone unfrozen – backbone LR={self.backbone_lr}, head LR={self.head_lr}")

    def _make_scheduler(self, optimizer):
        if self.scheduler_type == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5
            )
        if self.scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.num_epochs, eta_min=1e-6
            )
        if self.scheduler_type == 'step':
            return optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        return None

    def _log_gpu_memory(self, epoch):
        if not torch.cuda.is_available():
            return
        idx = self.device.index or 0
        alloc = torch.cuda.memory_allocated(idx) / 1e9
        reserved = torch.cuda.memory_reserved(idx) / 1e9
        total = torch.cuda.get_device_properties(idx).total_memory / 1e9
        print(f"  GPU mem: {alloc:.2f}GB alloc / {reserved:.2f}GB reserved / {total:.2f}GB total")
        self.writer.add_scalar('GPU/MemoryAllocatedGB', alloc, epoch)
        self.writer.add_scalar('GPU/MemoryReservedGB', reserved, epoch)

    # ------------------------------------------------------------------
    # Train / validate
    # ------------------------------------------------------------------

    def train_epoch(self):
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch + 1}/{self.num_epochs} [P{self.current_phase}]')
        for batch_idx, (audio, labels) in enumerate(pbar):
            audio = audio.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            embeddings = self.model(audio)
            loss = self.criterion(embeddings, labels)

            batch_loss = loss.item()
            if math.isnan(batch_loss) or math.isinf(batch_loss):
                print(f"\n  NaN/Inf loss at batch {batch_idx} — aborting epoch")
                return float('nan')

            loss.backward()
            trainable = [p for p in self.model.parameters() if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            self.optimizer.step()

            epoch_loss += batch_loss
            num_batches += 1
            pbar.set_postfix({'loss': f'{batch_loss:.4f}'})

            global_step = self.current_epoch * len(self.train_loader) + batch_idx
            self.writer.add_scalar('Train/BatchLoss', batch_loss, global_step)

        avg_loss = epoch_loss / num_batches
        self.train_losses.append(avg_loss)
        return avg_loss

    def validate(self):
        self.model.eval()
        epoch_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for audio, labels in tqdm(self.val_loader, desc='Validation'):
                audio = audio.to(self.device)
                labels = labels.to(self.device)

                embeddings = self.model(audio)
                loss = self.criterion(embeddings, labels)

                epoch_loss += loss.item()
                num_batches += 1

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
            best_path = self.checkpoint_dir / 'best_model.pth'
            atomic_torch_save(checkpoint, best_path)
            print(f"  Best model saved: {best_path}")

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        saved_phase = checkpoint.get('phase', 1)
        if saved_phase == 2 and self.current_phase == 1:
            self._init_phase2()

        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch'] + 1
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])

        print(f"Checkpoint loaded: {checkpoint_path}  (epoch {self.current_epoch}, phase {self.current_phase})")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def train(self):
        print("\n" + "=" * 60)
        print("Starting MusiCNN Contrastive Training (Two-Phase)")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Total epochs: {self.num_epochs}  |  Warm-up epochs: {self.warmup_epochs}")
        print(f"Train batches: {len(self.train_loader)}  |  Val batches: {len(self.val_loader)}")
        print("=" * 60 + "\n")

        for epoch in range(self.current_epoch, self.num_epochs):
            self.current_epoch = epoch

            # Transition from phase 1 to phase 2
            if self.current_phase == 1 and epoch >= self.warmup_epochs:
                self._init_phase2()

            train_loss = self.train_epoch()
            if math.isnan(train_loss):
                print(f"\nTraining aborted: NaN loss at epoch {epoch + 1}")
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
            print(f"\nEpoch {epoch + 1}/{self.num_epochs} [Phase {self.current_phase}]")
            print(f"  Train Loss: {train_loss:.4f}  |  Val Loss: {val_loss:.4f}  |  LR: {lr_str}")

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
                print(f"  New best validation loss: {val_loss:.4f}")
            else:
                self.patience_counter += 1
                print(f"  Patience: {self.patience_counter}/{self.early_stopping_patience}")

            if self.patience_counter >= self.early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break

            print()

        print("\n" + "=" * 60)
        print("Training Complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print("=" * 60 + "\n")
        self.writer.close()


# ======================================================================
# CLI entry-point
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description='Train MusiCNN Contrastive (Two-Phase)')

    # Dataset
    parser.add_argument('--data-root', type=str, default=config.MTG_JAMENDO_ROOT)
    parser.add_argument('--split', type=str, default='split-0')

    # Model
    parser.add_argument('--pretrained-weights', type=str, required=True,
                        help='Path to pretrained MusiCNN MSD weights (.pth)')
    parser.add_argument('--projection-dim', type=int,
                        default=config.CONTRASTIVE_TRAINING['projection_dim'])

    # Training
    parser.add_argument('--batch-size', type=int,
                        default=config.CONTRASTIVE_TRAINING['batch_size'])
    parser.add_argument('--num-epochs', type=int,
                        default=config.CONTRASTIVE_TRAINING['num_epochs'])
    parser.add_argument('--warmup-epochs', type=int, default=10,
                        help='Epochs to train with frozen backbone (phase 1)')
    parser.add_argument('--lr', type=float,
                        default=config.CONTRASTIVE_TRAINING['learning_rate'],
                        help='Learning rate for projection head')
    parser.add_argument('--backbone-lr', type=float, default=1e-5,
                        help='Learning rate for backbone in phase 2')
    parser.add_argument('--temperature', type=float,
                        default=config.CONTRASTIVE_TRAINING['temperature'])
    parser.add_argument('--num-workers', type=int,
                        default=config.CONTRASTIVE_TRAINING['num_workers'])
    parser.add_argument('--duration', type=float,
                        default=config.CONTRASTIVE_TRAINING['audio_duration'])
    parser.add_argument('--balanced-sampling', action='store_true',
                        default=config.CONTRASTIVE_TRAINING['balanced_sampling'])

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

    # Default directories
    if args.checkpoint_dir is None:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.checkpoint_dir = os.path.join(
            config.CONTRASTIVE_TRAINING['checkpoint_dir'],
            f'musicnn_contrastive_{ts}',
        )
    if args.log_dir is None:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.log_dir = os.path.join(
            config.CONTRASTIVE_TRAINING['log_dir'],
            f'musicnn_contrastive_{ts}',
        )

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.cuda_number is not None:
        device = torch.device(f'cuda:{args.cuda_number}')
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(device.index)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(device.index).total_memory / 1e9:.2f} GB\n")

    # Data
    data_root = Path(args.data_root).expanduser()
    splits_dir = data_root / 'data' / 'splits' / args.split
    audio_dir = data_root / 'songs'

    train_tsv = splits_dir / 'autotagging_genre-train.tsv'
    val_tsv = splits_dir / 'autotagging_genre-validation.tsv'
    test_tsv = splits_dir / 'autotagging_genre-test.tsv'

    for path in [train_tsv, val_tsv, test_tsv, audio_dir]:
        if not path.exists():
            raise FileNotFoundError(f"Required path not found: {path}")

    print("Creating dataloaders...")
    train_loader, val_loader, test_loader, dataset_info = create_dataloaders(
        train_tsv=str(train_tsv),
        val_tsv=str(val_tsv),
        test_tsv=str(test_tsv),
        audio_dir=str(audio_dir),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sample_rate=config.SAMPLE_RATE,
        duration=args.duration,
        balanced_sampling=args.balanced_sampling,
    )

    print(f"\nDataset Info:")
    print(f"  Number of classes: {dataset_info['num_classes']}")
    print(f"  Train size: {dataset_info['train_size']}")
    print(f"  Val size: {dataset_info['val_size']}")
    print(f"  Test size: {dataset_info['test_size']}")

    # Save dataset info
    info_path = Path(args.checkpoint_dir) / 'dataset_info.json'
    info_path.parent.mkdir(parents=True, exist_ok=True)
    with open(info_path, 'w') as f:
        json.dump({
            'num_classes': dataset_info['num_classes'],
            'genre_to_idx': dataset_info['genre_to_idx'],
            'idx_to_genre': dataset_info['idx_to_genre'],
            'train_size': dataset_info['train_size'],
            'val_size': dataset_info['val_size'],
            'test_size': dataset_info['test_size'],
        }, f, indent=2)

    # Model
    print("\nInitializing MusiCNNContrastive model...")
    model = MusiCNNContrastive(
        projection_dim=args.projection_dim,
        pretrained_weights=args.pretrained_weights,
        device=device,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters (phase 1): {trainable_params:,}")

    # Loss
    criterion = SupConLoss(temperature=args.temperature)

    # Trainer
    trainer = ContrastiveTrainer(
        model=model,
        train_loader=train_loader,
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
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Save training config
    config_path = Path(args.checkpoint_dir) / 'training_config.json'
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)

    trainer.train()

    print(f"\nCheckpoints saved to: {args.checkpoint_dir}")
    print(f"TensorBoard logs saved to: {args.log_dir}")
    print(f"\nTo view training progress, run:")
    print(f"  tensorboard --logdir {args.log_dir}")


if __name__ == '__main__':
    main()
