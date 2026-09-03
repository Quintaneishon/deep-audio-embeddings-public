"""
Training script for Whisper-based multi-label supervised contrastive learning.

Uses all genre tags per track (multi-hot labels) with a Jaccard-weighted
contrastive loss instead of single-label SupCon.
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

from backend.DL.models.WhisperContrastive import WhisperContrastive
from backend.DL.trainings.losses import MultiLabelSupConLoss
from backend.DL.trainings.dataset import create_dataloaders_multilabel as create_dataloaders
from backend.DL.trainings.checkpointing import atomic_torch_save
from backend import config


class ContrastiveTrainer:
    """Trainer class for supervised contrastive learning."""

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
        early_stopping_patience=10
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

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(log_dir=self.log_dir)

        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.train_losses = []
        self.val_losses = []

    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()

        for param in self.model.projection_head.parameters():
            param.requires_grad = True

        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch + 1}/{self.num_epochs}')
        for batch_idx, (audio, labels) in enumerate(pbar):
            audio = audio.to(self.device)
            labels = labels.to(self.device)

            embeddings = self.model(audio)
            loss = self.criterion(embeddings, labels)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.projection_head.parameters(), max_norm=1.0)
            self.optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': loss.item()})

            global_step = self.current_epoch * len(self.train_loader) + batch_idx
            self.writer.add_scalar('Train/BatchLoss', loss.item(), global_step)

        avg_loss = epoch_loss / num_batches
        self.train_losses.append(avg_loss)
        return avg_loss

    def validate(self):
        """Validate the model."""
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
                'n_audio_state': self.model.n_audio_state
            }
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
        """Main training loop."""
        print("\n" + "="*60)
        print("Starting Multi-Label Contrastive Training")
        print("="*60)
        print(f"Device: {self.device}")
        print(f"Epochs: {self.num_epochs}")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        print("="*60 + "\n")

        for epoch in range(self.current_epoch, self.num_epochs):
            self.current_epoch = epoch

            train_loss = self.train_epoch()
            val_loss = self.validate()

            if self.scheduler:
                self.scheduler.step(val_loss)

            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")

            if self.scheduler:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"  Learning Rate: {current_lr:.6f}")
                self.writer.add_scalar('Train/LearningRate', current_lr, epoch)

            self.writer.add_scalar('Train/EpochLoss', train_loss, epoch)
            self.writer.add_scalar('Val/EpochLoss', val_loss, epoch)

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

        print("\n" + "="*60)
        print("Training Complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print("="*60 + "\n")

        self.writer.close()


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description='Train Whisper Contrastive Model (Multi-Label)'
    )

    # Dataset arguments
    parser.add_argument('--data-root', type=str,
                        default=config.MTG_JAMENDO_ROOT,
                        help='Root directory of MTG-Jamendo dataset')
    parser.add_argument('--split', type=str, default='split-0',
                        help='Data split to use')

    # Model arguments
    parser.add_argument('--model-name', type=str,
                        default=config.CONTRASTIVE_TRAINING['model_name'],
                        choices=['tiny', 'base', 'small', 'medium'],
                        help='Whisper model size')
    parser.add_argument('--projection-dim', type=int,
                        default=config.CONTRASTIVE_TRAINING['projection_dim'],
                        help='Dimension of projection head output')

    # Training arguments
    parser.add_argument('--batch-size', type=int,
                        default=config.CONTRASTIVE_TRAINING['batch_size'],
                        help='Batch size for training')
    parser.add_argument('--num-epochs', type=int,
                        default=config.CONTRASTIVE_TRAINING['num_epochs'],
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float,
                        default=config.CONTRASTIVE_TRAINING['learning_rate'],
                        help='Learning rate')
    parser.add_argument('--temperature', type=float,
                        default=config.CONTRASTIVE_TRAINING['temperature'],
                        help='Temperature for contrastive loss')
    parser.add_argument('--num-workers', type=int,
                        default=config.CONTRASTIVE_TRAINING['num_workers'],
                        help='Number of data loading workers')
    parser.add_argument('--duration', type=float,
                        default=config.CONTRASTIVE_TRAINING['audio_duration'],
                        help='Audio duration in seconds')
    parser.add_argument('--balanced-sampling', action='store_true',
                        default=config.CONTRASTIVE_TRAINING['balanced_sampling'],
                        help='Use balanced sampling for training')

    # Optimization arguments
    parser.add_argument('--weight-decay', type=float,
                        default=config.CONTRASTIVE_TRAINING['weight_decay'],
                        help='Weight decay for optimizer')
    parser.add_argument('--scheduler', type=str,
                        default=config.CONTRASTIVE_TRAINING['scheduler'],
                        choices=['plateau', 'cosine', 'step', 'none'],
                        help='Learning rate scheduler')
    parser.add_argument('--early-stopping', type=int,
                        default=config.CONTRASTIVE_TRAINING['early_stopping_patience'],
                        help='Early stopping patience')

    # Checkpoint arguments
    parser.add_argument('--checkpoint-dir', type=str,
                        default=None,
                        help='Directory to save checkpoints')
    parser.add_argument('--log-dir', type=str,
                        default=None,
                        help='Directory for TensorBoard logs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--cuda-number', type=str, default=None,
                        help='CUDA number to use')

    args = parser.parse_args()

    if args.checkpoint_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.checkpoint_dir = os.path.join(
            config.CONTRASTIVE_TRAINING['checkpoint_dir'],
            f'whisper_contrastive_multilabel_{timestamp}'
        )

    if args.log_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.log_dir = os.path.join(
            config.CONTRASTIVE_TRAINING['log_dir'],
            f'whisper_contrastive_multilabel_{timestamp}'
        )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.cuda_number is not None:
        device = torch.device(f'cuda:{args.cuda_number}')
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(device.index)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(device.index).total_memory / 1e9:.2f} GB\n")

    data_root = Path(args.data_root).expanduser()
    splits_dir = data_root / 'data' / 'splits' / args.split
    audio_dir = data_root / 'songs'

    train_tsv = splits_dir / 'autotagging_genre-train.tsv'
    val_tsv = splits_dir / 'autotagging_genre-validation.tsv'
    test_tsv = splits_dir / 'autotagging_genre-test.tsv'

    for path in [train_tsv, val_tsv, test_tsv, audio_dir]:
        if not path.exists():
            raise FileNotFoundError(f"Required path not found: {path}")

    print("Creating dataloaders (multi-label)...")
    train_loader, val_loader, test_loader, dataset_info = create_dataloaders(
        train_tsv=str(train_tsv),
        val_tsv=str(val_tsv),
        test_tsv=str(test_tsv),
        audio_dir=str(audio_dir),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sample_rate=config.SAMPLE_RATE,
        duration=args.duration,
        balanced_sampling=args.balanced_sampling
    )

    print(f"\nDataset Info:")
    print(f"  Number of classes: {dataset_info['num_classes']}")
    print(f"  Train size: {dataset_info['train_size']}")
    print(f"  Val size: {dataset_info['val_size']}")
    print(f"  Test size: {dataset_info['test_size']}")

    dataset_info_path = Path(args.checkpoint_dir) / 'dataset_info.json'
    dataset_info_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dataset_info_path, 'w') as f:
        info_to_save = {
            'num_classes': dataset_info['num_classes'],
            'genre_to_idx': dataset_info['genre_to_idx'],
            'idx_to_genre': dataset_info['idx_to_genre'],
            'train_size': dataset_info['train_size'],
            'val_size': dataset_info['val_size'],
            'test_size': dataset_info['test_size'],
            'multi_label': True,
        }
        json.dump(info_to_save, f, indent=2)

    print("\nInitializing model...")
    model = WhisperContrastive(
        model_name=args.model_name,
        projection_dim=args.projection_dim,
        device=device
    )

    criterion = MultiLabelSupConLoss(temperature=args.temperature)

    optimizer = optim.Adam(
        model.get_trainable_parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
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
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=10, gamma=0.5
        )

    trainer = ContrastiveTrainer(
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
        early_stopping_patience=args.early_stopping
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)

    config_path = Path(args.checkpoint_dir) / 'training_config.json'
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)

    trainer.train()

    print(f"\nCheckpoints saved to: {args.checkpoint_dir}")
    print(f"TensorBoard logs saved to: {args.log_dir}")
    print("\nTo view training progress, run:")
    print(f"  tensorboard --logdir {args.log_dir}")


if __name__ == '__main__':
    main()
