import os
from pathlib import Path

from dotenv import load_dotenv

# Get the project root directory (where this config.py file is located)
PROJECT_ROOT = Path(__file__).parent.absolute()

# Load .env file (placed alongside this config.py at backend/.env)
load_dotenv(PROJECT_ROOT / '.env')

# Directory paths (absolute paths)
CACHE_DIR = str(PROJECT_ROOT / 'db')
DATABASE_PATH = str(PROJECT_ROOT / 'db' / 'data_thesis.db')
DATABASE_EVAL_PATH = str(PROJECT_ROOT / 'db' / 'data_eval_remote.db')
AUDIO_DIR = str(PROJECT_ROOT / 'data' / 'audio_fma')
CSV_PATH = AUDIO_DIR + '/selected_songs.csv'

# External dataset paths — read from .env, fallback to placeholder.
# Override these in backend/.env (see .env.example).
FMA_METADATA_DIR = os.environ.get('FMA_METADATA_DIR', 'backend')
FMA_SMALL_DIR = os.environ.get('FMA_SMALL_DIR', '/path/to/fma_small')
MTG_JAMENDO_ROOT = os.environ.get('MTG_JAMENDO_ROOT', '/path/to/mtg-jamendo-dataset')
MY_TRACKS_DIR = os.environ.get('MY_TRACKS_DIR', '')

# Compute device for inference and training. Override with CUDA_DEVICE in .env.
CUDA_DEVICE = os.environ.get('CUDA_DEVICE', 'cpu')

# Public deployments are read-only by default. Enable these deliberately only
# while collecting a study or when using the local upload/delete UI.
EVAL_COLLECTION_OPEN = os.environ.get('EVAL_COLLECTION_OPEN', 'false').lower() == 'true'
MUTATIONS_ENABLED = os.environ.get('MUTATIONS_ENABLED', 'false').lower() == 'true'
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get(
        'ALLOWED_ORIGINS',
        'http://localhost:3000,http://127.0.0.1:3000',
    ).split(',')
    if origin.strip()
]

AUDIO_OUTPUT_DIR = str(PROJECT_ROOT / 'data' / 'audio_fma')
MODEL_DIR = str(PROJECT_ROOT / 'ML' / 'models')
WEIGHTS_DIR = str(PROJECT_ROOT / 'DL'  / 'weights')
REPORTS_DIR = str(PROJECT_ROOT / 'reports')

# Model weight paths (absolute paths)
MODEL_WEIGHTS = {
    'musicnn': {
        'msd': str(Path(WEIGHTS_DIR) / 'msd' / 'musicnn.pth'),
        'mtat': str(Path(WEIGHTS_DIR) / 'mtat' / 'musicnn.pth')
    },
    'vgg': {
        'msd': str(Path(WEIGHTS_DIR) / 'msd' / 'vgg.pth'),
        'mtat': str(Path(WEIGHTS_DIR) / 'mtat' / 'vgg.pth')
    },
    'vggish': {
        'pretrained': str(Path(WEIGHTS_DIR) / 'vggish' / 'vggish-10086976.pth') # This is not necessary, we use the pretrained weights from the model
    },
    'whisper': {
        'base': 'base'
    },
    'mert': {
        '95m': str(Path(WEIGHTS_DIR) / 'mert')  # Cache directory for HuggingFace models
    },
    'whisper_contrastive': {
        'base': str(Path(WEIGHTS_DIR) / 'whisper_contrastive' / 'best_model.pth')
    },
    'whisper_contrastive_multilabel': {
        'base': str(Path(WEIGHTS_DIR) / 'whisper_contrastive_multilabel' / 'best_model.pth')
    },
    'whisper_contrastive_hybrid': {
        'base': str(Path(WEIGHTS_DIR) / 'whisper_contrastive_hybrid' / 'best_model.pth')
    },
    'musicnn_contrastive_hybrid': {
        'msd': str(Path(WEIGHTS_DIR) / 'musicnn_contrastive_hybrid' / 'best_model.pth')
    },
    'musicnn_contrastive': {
        'msd': str(Path(WEIGHTS_DIR) / 'musicnn_contrastive' / 'best_model.pth')
    },
    # Future models — registered once checkpoints are published to weights/
    'musicnn_multisignal': {
        'msd': str(Path(WEIGHTS_DIR) / 'musicnn_multisignal' / 'best_model.pth')
    },
    'mert_lora': {
        '95m': str(Path(WEIGHTS_DIR) / 'mert_lora_multisignal' / 'best_model.pth')
    },
    'mert_contrastive_multilabel': {
        '95m': str(Path(WEIGHTS_DIR) / 'mert_contrastive_multilabel' / 'best_model.pth')
    },
    'vgg_contrastive': {
        'msd': str(Path(WEIGHTS_DIR) / 'vgg_contrastive_multilabel' / 'best_model.pth')
    },
    'vgg_contrastive_proj512': {
        'msd': str(Path(WEIGHTS_DIR) / 'vgg_contrastive_proj512' / 'best_model.pth')
    },
    'vgg_audio_pure': {
        'msd': str(Path(WEIGHTS_DIR) / 'vgg_audio_pure' / 'best_model.pth')
    },
    'vgg_whisper_dual': {
        'msd': str(Path(WEIGHTS_DIR) / 'vgg_whisper_dual' / 'best_model.pth')
    },
}

# Human-readable display names for the UI
MODEL_DISPLAY_NAMES = {
    'musicnn': 'MusiCNN',
    'vgg': 'VGG',
    'whisper': 'Whisper',
    'vggish': 'VGGish',
    'mert': 'MERT',
    'whisper_contrastive': 'Whisper Contrastive (SupCon)',
    'whisper_contrastive_multilabel': 'Whisper Contrastive (MultiLabel)',
    'whisper_contrastive_hybrid': 'Whisper Contrastive (Hybrid)',
    'musicnn_contrastive': 'MusiCNN Contrastive (SupCon)',
    'musicnn_contrastive_hybrid': 'MusiCNN Contrastive (Hybrid)',
    'musicnn_multisignal': 'MusiCNN Contrastive (MultiSignal)',
    'mert_lora': 'MERT LoRA (MultiSignal)',
    'mert_contrastive_multilabel': 'MERT Contrastive (MultiLabel, Layer 6)',
    'vgg_contrastive': 'VGG Contrastive (MultiLabel)',
    'vgg_contrastive_proj512': 'VGG Contrastive (MultiLabel, proj512)',
    'vgg_audio_pure': 'VGG Audio-Pure (Contrastive)',
    'vgg_whisper_dual': 'VGG+Whisper Dual Backbone (Hybrid Loss)',
}

DATASET_DISPLAY_NAMES = {
    'msd': 'MSD',
    'mtat': 'MTAT',
    'pretrained': 'Pretrained',
    'base': 'Base',
    '95m': '95M',
    'mtgjamendo': 'MTG-Jamendo',
}

# HuggingFace model identifiers for MERT
MERT_MODEL_IDS = {
    '95m': 'm-a-p/MERT-v1-95M',
    '330m': 'm-a-p/MERT-v1-330M'
}

VGGISH_WEIGHTS_URL = "https://github.com/harritaylor/torchvggish/releases/download/v0.1/vggish-10086976.pth"
PCA_PARAMS_URL = "https://github.com/harritaylor/torchvggish/releases/download/v0.1/vggish_pca_params-970ea276.pth"

# Preprocessing settings
SAMPLE_RATE = 16000  # Hz
N_TAGS = 50  # Number of tags
N_DIM = 128  # Number of dimensions

# ============================================================================
# Contrastive Learning Configuration
# ============================================================================

# Dataset paths for MTG-Jamendo (root is read from .env above)
MTG_JAMENDO_AUDIO_DIR = str(Path(MTG_JAMENDO_ROOT) / 'songs')
MTG_JAMENDO_SPLITS_DIR = str(Path(MTG_JAMENDO_ROOT) / 'data' / 'splits' / 'split-0')

# Training hyperparameters
CONTRASTIVE_TRAINING = {
    # Model configuration
    'model_name': 'base',  # Whisper model size: 'tiny', 'base', 'small', 'medium'
    'projection_dim': N_DIM,  # Use shared dimension setting

    # Training parameters
    'batch_size': 16,  # Batch size (adjust based on GPU memory)
    'num_epochs': 50,  # Maximum number of epochs
    'learning_rate': 1e-3,  # Initial learning rate
    'weight_decay': 1e-4,  # L2 regularization

    # Loss function
    'temperature': 0.07,  # Temperature for contrastive loss (0.07 is standard)

    # Data loading
    'num_workers': 4,  # Number of parallel data loading workers
    'audio_duration': 30.0,  # Audio duration in seconds
    'sample_rate': SAMPLE_RATE,  # Use shared sample rate setting
    'balanced_sampling': False,  # Use weighted sampling for class balance

    # Optimization
    'scheduler': 'plateau',  # LR scheduler: 'plateau', 'cosine', 'step', 'none'
    'early_stopping_patience': 10,  # Epochs to wait before early stopping

    # Checkpointing
    'checkpoint_dir': str(PROJECT_ROOT / 'DL' / 'trainings' / 'checkpoints'),
    'log_dir': str(PROJECT_ROOT / 'DL' / 'trainings' / 'logs'),
    'save_frequency': 1,  # Save checkpoint every N epochs
}
