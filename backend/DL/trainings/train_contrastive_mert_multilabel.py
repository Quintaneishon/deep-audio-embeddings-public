"""
Training script for MERT-based multi-label supervised contrastive learning.

Frozen MERT-v1-95M backbone at layer 6 + trainable projection head.
Uses MultiLabelSupConLoss with Jaccard-weighted positive pairs from
all genre tags per track.

GPU memory notes:
  - MERT-v1-95M backbone: ~380 MB
  - Backbone is fully frozen (no gradients stored through it)
  - Default batch_size=4 is safe for a 24 GB GPU on cuda:0
  - Increase to 8 if memory allows, decrease to 2 if OOM

Gradient accumulation:
  - Default --grad-accum-steps=4 gives effective batch = 4 × 4 = 16
  - Smooths out noisy gradients from small batches without extra GPU memory

Usage:
    cd /path/to/deep-audio-embeddings
    source venv/bin/activate
    python -m backend.DL.trainings.train_contrastive_mert_multilabel \\
        --cuda-number 0 --batch-size 4 --grad-accum-steps 4
"""

import os
import argparse
import json
from pathlib import Path
from datetime import datetime

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from backend.DL.models.MERTContrastiveMultiLabel import MERTContrastiveMultiLabel
from backend.DL.trainings.checkpointing import atomic_torch_save
from backend.DL.trainings.losses import MultiLabelSupConLoss
from backend.DL.trainings.dataset import create_dataloaders_multilabel as create_dataloaders
from backend import config

MERT_SAMPLE_RATE = 24000  # MERT requires 24kHz — different from Whisper's 16kHz


class MERTContrastiveTrainer:
    """Trainer for MERT contrastive multilabel learning."""

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        checkpoint_dir,
        log_dir,
        num_epochs=50,
        early_stopping_patience=10,
        grad_accum_steps=4,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        self.num_epochs = num_epochs
        self.early_stopping_patience = early_stopping_patience
        self.grad_accum_steps = grad_accum_steps

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(log_dir=self.log_dir)

        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.train_losses = []
        self.val_losses = []

    def train_epoch(self):
        self.model.train()

        # Ensure projection head is trainable (backbone stays frozen)
        for param in self.model.projection_head.parameters():
            param.requires_grad = True

        epoch_loss = 0.0
        num_batches = 0

        self.optimizer.zero_grad()

        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch + 1}/{self.num_epochs}')
        for batch_idx, (audio, labels) in enumerate(pbar):
            audio = audio.to(self.device)
            labels = labels.to(self.device)

            embeddings = self.model(audio)
            # Scale loss by accumulation steps so effective gradient magnitude
            # matches a single large-batch update
            loss = self.criterion(embeddings, labels) / self.grad_accum_steps

            if not torch.isfinite(loss):
                print(f"  WARNING: non-finite loss at batch {batch_idx} (epoch {self.current_epoch + 1}), skipping")
                self.optimizer.zero_grad()
                continue

            loss.backward()

            if (batch_idx + 1) % self.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.projection_head.parameters(), max_norm=1.0
                )
                self.optimizer.step()
                self.optimizer.zero_grad()

            # Track the un-scaled loss for logging
            scaled_loss = loss.item() * self.grad_accum_steps
            epoch_loss += scaled_loss
            num_batches += 1
            pbar.set_postfix({'loss': f'{scaled_loss:.4f}'})

            global_step = self.current_epoch * len(self.train_loader) + batch_idx
            self.writer.add_scalar('Train/BatchLoss', scaled_loss, global_step)

        # Handle leftover batches not divisible by grad_accum_steps
        remainder = len(self.train_loader) % self.grad_accum_steps
        if remainder != 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.projection_head.parameters(), max_norm=1.0
            )
            self.optimizer.step()
            self.optimizer.zero_grad()

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

    def save_checkpoint(self, filename='checkpoint.pth', is_best=False):
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'model_config': {
                'model_name': self.model.model_name,
                'projection_dim': self.model.projection_dim,
                'backbone_layer': self.model.backbone_layer,
                'sample_rate': self.model.sample_rate,
            },
        }

        checkpoint_path = self.checkpoint_dir / filename
        atomic_torch_save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            atomic_torch_save(checkpoint, best_path)
            print(f"Best model saved: {best_path}")

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.current_epoch = checkpoint['epoch'] + 1
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']

        print(f"Checkpoint loaded: {checkpoint_path}")
        print(f"  Resuming from epoch {self.current_epoch}")
        print(f"  Best validation loss: {self.best_val_loss:.4f}")

    def train(self):
        print("\n" + "=" * 60)
        print("MERT Contrastive Multi-Label Training")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Backbone: {self.model.model_name} (layer {self.model.backbone_layer}, FROZEN)")
        print(f"Projection dim: {self.model.projection_dim}")
        print(f"Epochs: {self.num_epochs}")
        print(f"Batch size: {self.train_loader.batch_size} × {self.grad_accum_steps} accum = {self.train_loader.batch_size * self.grad_accum_steps} effective")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"Parameters: {trainable:,} trainable / {total:,} total")
        print("=" * 60 + "\n")

        for epoch in range(self.current_epoch, self.num_epochs):
            self.current_epoch = epoch

            train_loss = self.train_epoch()
            val_loss = self.validate()

            if self.scheduler:
                self.scheduler.step(val_loss)

            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}")

            if self.scheduler:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"  LR: {current_lr:.6f}")
                self.writer.add_scalar('Train/LearningRate', current_lr, epoch)

            self.writer.add_scalar('Train/EpochLoss', train_loss, epoch)
            self.writer.add_scalar('Val/EpochLoss', val_loss, epoch)

            self.save_checkpoint(filename=f'checkpoint_epoch_{epoch + 1}.pth')

            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint(is_best=True)
                print(f"  *** New best: {val_loss:.4f} ***")
            else:
                self.patience_counter += 1
                print(f"  Patience: {self.patience_counter}/{self.early_stopping_patience}")

            if self.patience_counter >= self.early_stopping_patience:
                print(f"\nEarly stopping after {epoch + 1} epochs")
                break

            print()

        print("\n" + "=" * 60)
        print("Training Complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print("=" * 60 + "\n")

        self.writer.close()


def main():
    parser = argparse.ArgumentParser(
        description='Train MERT Contrastive Model (Multi-Label, Layer 6)'
    )

    # Dataset
    parser.add_argument('--data-root', type=str, default=config.MTG_JAMENDO_ROOT)
    parser.add_argument('--split', type=str, default='split-0')

    # Model
    parser.add_argument('--model-name', type=str, default='m-a-p/MERT-v1-95M')
    parser.add_argument('--projection-dim', type=int, default=128)
    parser.add_argument('--backbone-layer', type=int, default=6,
                        help='Which MERT layer to use as backbone output (0-11, default 6)')

    # Training — small batches to protect GPU memory
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size (keep ≤8 to avoid OOM on 24GB GPU)')
    parser.add_argument('--grad-accum-steps', type=int, default=4,
                        help='Gradient accumulation steps. Effective batch = batch_size × grad_accum_steps (default: 4 → effective 16)')
    parser.add_argument('--num-epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--duration', type=float, default=30.0)
    parser.add_argument('--balanced-sampling', action='store_true', default=False)

    # Optimizer
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--scheduler', type=str, default='plateau',
                        choices=['plateau', 'cosine', 'step', 'none'])
    parser.add_argument('--early-stopping', type=int, default=10)

    # Checkpoints
    parser.add_argument('--checkpoint-dir', type=str, default=None)
    parser.add_argument('--log-dir', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--cuda-number', type=str, default=None,
                        help='CUDA device index (e.g. 1 for cuda:1)')

    args = parser.parse_args()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f'mert_contrastive_multilabel_{timestamp}'

    if args.checkpoint_dir is None:
        args.checkpoint_dir = os.path.join(
            config.CONTRASTIVE_TRAINING['checkpoint_dir'], run_name
        )
    if args.log_dir is None:
        args.log_dir = os.path.join(
            config.CONTRASTIVE_TRAINING['log_dir'], run_name
        )

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.cuda_number is not None:
        device = torch.device(f'cuda:{args.cuda_number}')
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        idx = device.index if device.index is not None else 0
        print(f"GPU: {torch.cuda.get_device_name(idx)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(idx).total_memory / 1e9:.1f} GB\n")

    # Data paths
    data_root = Path(args.data_root).expanduser()
    splits_dir = data_root / 'data' / 'splits' / args.split
    audio_dir = data_root / 'songs'

    train_tsv = splits_dir / 'autotagging_genre-train.tsv'
    val_tsv = splits_dir / 'autotagging_genre-validation.tsv'
    test_tsv = splits_dir / 'autotagging_genre-test.tsv'

    for path in [train_tsv, val_tsv, test_tsv, audio_dir]:
        if not path.exists():
            raise FileNotFoundError(f"Required path not found: {path}")

    print("Creating dataloaders (multi-label, 24kHz for MERT)...")
    train_loader, val_loader, test_loader, dataset_info = create_dataloaders(
        train_tsv=str(train_tsv),
        val_tsv=str(val_tsv),
        test_tsv=str(test_tsv),
        audio_dir=str(audio_dir),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sample_rate=MERT_SAMPLE_RATE,  # 24kHz — critical for MERT
        duration=args.duration,
        balanced_sampling=args.balanced_sampling,
    )

    print(f"\nDataset Info:")
    print(f"  Classes: {dataset_info['num_classes']}")
    print(f"  Train: {dataset_info['train_size']} | Val: {dataset_info['val_size']} | Test: {dataset_info['test_size']}")

    # Save dataset info
    dataset_info_path = Path(args.checkpoint_dir) / 'dataset_info.json'
    dataset_info_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dataset_info_path, 'w') as f:
        json.dump({
            'num_classes': dataset_info['num_classes'],
            'genre_to_idx': dataset_info['genre_to_idx'],
            'idx_to_genre': dataset_info['idx_to_genre'],
            'train_size': dataset_info['train_size'],
            'val_size': dataset_info['val_size'],
            'test_size': dataset_info['test_size'],
            'multi_label': True,
            'sample_rate': MERT_SAMPLE_RATE,
        }, f, indent=2)

    print(f"\nInitializing MERT model (layer {args.backbone_layer})...")
    model = MERTContrastiveMultiLabel(
        model_name=args.model_name,
        projection_dim=args.projection_dim,
        backbone_layer=args.backbone_layer,
        device=device,
    )

    criterion = MultiLabelSupConLoss(temperature=args.temperature)

    optimizer = optim.Adam(
        model.get_trainable_parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = None
    if args.scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
    elif args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.num_epochs, eta_min=1e-6
        )
    elif args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    trainer = MERTContrastiveTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        num_epochs=args.num_epochs,
        early_stopping_patience=args.early_stopping,
        grad_accum_steps=args.grad_accum_steps,
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Save training config
    with open(Path(args.checkpoint_dir) / 'training_config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    trainer.train()

    print(f"\nCheckpoints: {args.checkpoint_dir}")
    print(f"TensorBoard: tensorboard --logdir {args.log_dir}")


if __name__ == '__main__':
    main()
