import os
import torch
import logging
from typing import Dict, Any, Optional, List


logger = logging.getLogger(__name__)

# --- Device Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Directory Structure ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
PREPROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'preprocessed')
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
LOG_DIR = os.path.join(BASE_DIR, 'logs')

# Create Directories
os.makedirs(PREPROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# --- Data Files ---
TRAIN_DATA_FILENAME = 'full_dataset_40k.csv'
TRAIN_DATA = os.path.join(PREPROCESSED_DATA_DIR, TRAIN_DATA_FILENAME)

# --- Logging ---
TRAINING_LOG_FILENAME = 'deezymatch_training.log'
LOG_PATH = os.path.join(LOG_DIR, TRAINING_LOG_FILENAME)

# --- Tokenizer Configuration ---
TOKENIZER_PATH = os.path.join(CHECKPOINT_DIR, 'bpe_tokenizer.json')
MAX_LEN = 32

# --- Model Configuration ---
MODEL_CONFIG = {
    'embedding_dim': 128,
    'rnn_hidden_dim': 256,
    'num_layers': 1,
    'dropout': 0.5,
    'padding_idx': 0,
    'ff_hidden_dim': 2048,
    'final_embedding_dim': 1024
}

# --- Training Configuration ---
TRAINING_CONFIG = {
    'num_epochs': 50,
    'batch_size': 2048,
    'learning_rate': 1e-4,
    'weight_decay': 0.01,
    'temperature': 0.1,
    'clip_grad_norm': 1.0,
    'use_shuffled_positives': True,
    'shuffle_probability': 0.80,
    'checkpoint_dir': CHECKPOINT_DIR,
}

def get_checkpoint_path(epoch: Optional[int] = None, checkpoint_dir: str = CHECKPOINT_DIR) -> Optional[str]:
    """ Returns path to specified epoch checkpoint or latest if None. """
    if not os.path.isdir(checkpoint_dir):
        return None

    if epoch is not None:
        path = f"{checkpoint_dir}/encoder_epoch_{epoch}.pth"
        return path if os.path.exists(path) else None

    try:
        checkpoints = [
            f for f in os.listdir(checkpoint_dir) 
            if f.startswith("encoder_epoch_") and f.endswith(".pth")
        ]
        if checkpoints:
            latest = max(checkpoints, key=lambda f: int(f.split('_')[-1][:-4]))
            return f"{checkpoint_dir}/{latest}"
    except Exception:
        pass
    
    return None

# --- Log initialization ---
logger.info("\n--- DeezyMatch Configuration ---")
logger.info(f"Device: {DEVICE}")
logger.info(f"Base directory: {BASE_DIR}")
logger.info(f"Model config: {MODEL_CONFIG}")
logger.info(f"Training config: {TRAINING_CONFIG}")
logger.info("--- Configuration loaded ---")