import os
import logging
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from tokenizers import Tokenizer
from sklearn.model_selection import KFold

from DeezyMatch import utils
from DeezyMatch.dataset import ContrastiveProductDataset
from DeezyMatch.model import DeezyMatchEncoder
import config

# Setup logging
utils.setup_logging(log_file=config.LOG_PATH)
logger = logging.getLogger(__name__)

def log_config(key, value, width=30):
    logger.info(f"{key:<{width}} : {value}")

def symmetric_infonce_loss(anchor_emb, positive_emb, temperature=0.1):
    """Compute symmetric InfoNCE loss using cosine similarity."""
    sim_matrix = torch.matmul(anchor_emb, positive_emb.T) / temperature
    batch_size = anchor_emb.size(0)
    labels = torch.arange(batch_size, device=anchor_emb.device)
    
    loss_anchor = F.cross_entropy(sim_matrix, labels)
    loss_positive = F.cross_entropy(sim_matrix.T, labels)
    return (loss_anchor + loss_positive) / 2

def main():
    # === Configuration ===
    device = torch.device(config.DEVICE)
    logger.info("="*60)
    logger.info(f"{'TRAINING CONFIGURATION (CROSS-VALIDATION)':^60}")
    log_config("Device", device)
    log_config("Batch Size", config.TRAINING_CONFIG['batch_size'])
    logger.info("="*60)

    # === Load data ===
    try:
        df = pd.read_csv(config.TRAIN_DATA, low_memory=False)
        logger.info(f"Loaded {len(df):,} total samples")
    except Exception as e:
        logger.error(f"Data loading failed: {e}")
        return

    # === Initialize tokenizer ===
    try:
        tokenizer = Tokenizer.from_file(config.TOKENIZER_PATH)
        tokenizer.enable_padding(
            pad_id=tokenizer.token_to_id("<pad>"),
            pad_token="<pad>",
            length=config.MAX_LEN
        )
        tokenizer.enable_truncation(max_length=config.MAX_LEN)
    except Exception as e:
        logger.error(f"Tokenizer initialization failed: {e}")
        return

    # === K-Fold setup ===
    k_folds = 5
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(df)):
        logger.info("="*60)
        logger.info(f"Fold {fold + 1}/{k_folds}")
        logger.info("="*60)

        # === Prepare datasets ===
        train_subset = df.iloc[train_idx].reset_index(drop=True)
        val_subset = df.iloc[val_idx].reset_index(drop=True)

        train_dataset = ContrastiveProductDataset(
            df=train_subset,
            tokenizer=tokenizer,
            max_len=config.MAX_LEN,
            use_shuffled_positives=config.TRAINING_CONFIG['use_shuffled_positives'],
            shuffle_probability=config.TRAINING_CONFIG['shuffle_probability']
        )

        val_dataset = ContrastiveProductDataset(
            df=val_subset,
            tokenizer=tokenizer,
            max_len=config.MAX_LEN,
            use_shuffled_positives=False  # Validation shouldn't shuffle
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.TRAINING_CONFIG['batch_size'],
            shuffle=True,
            pin_memory=True,
            drop_last=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config.TRAINING_CONFIG['batch_size'],
            shuffle=False,
            pin_memory=True,
            drop_last=True
        )

        # === Initialize model ===
        model = DeezyMatchEncoder(
            vocab_size=tokenizer.get_vocab_size(),
            **config.MODEL_CONFIG
        ).to(device)
        log_config("Trainable Parameters", 
                 f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.TRAINING_CONFIG['learning_rate'],
            weight_decay=config.TRAINING_CONFIG['weight_decay']
        )

        # === Training loop per fold ===
        for epoch in range(1, config.TRAINING_CONFIG['num_epochs'] + 1):
            model.train()
            train_loss = 0.0
            progress_bar = tqdm(train_loader, desc=f"Fold {fold+1} | Epoch {epoch} [Train]")

            for x1, x2 in progress_bar:
                x1, x2 = x1.to(device), x2.to(device)
                optimizer.zero_grad()

                emb1 = model.encode(x1)
                emb2 = model.encode(x2)
                loss = symmetric_infonce_loss(emb1, emb2, config.TRAINING_CONFIG['temperature'])

                loss.backward()
                if config.TRAINING_CONFIG['clip_grad_norm']:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAINING_CONFIG['clip_grad_norm'])
                optimizer.step()

                batch_loss = loss.item()
                train_loss += batch_loss
                progress_bar.set_postfix({'batch_loss': f'{batch_loss:.4f}'})

            avg_train_loss = train_loss / len(train_loader)

            # === Validation ===
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x1, x2 in tqdm(val_loader, desc=f"Fold {fold+1} | Epoch {epoch} [Val]"):
                    x1, x2 = x1.to(device), x2.to(device)
                    emb1 = model.encode(x1)
                    emb2 = model.encode(x2)
                    loss = symmetric_infonce_loss(emb1, emb2, config.TRAINING_CONFIG['temperature'])
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)
            logger.info(f"Fold {fold+1} | Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

            # === Save checkpoint per fold ===
            ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"encoder_fold_{fold+1}_epoch_{epoch}.pth")
            utils.CheckpointManager.save({
                'fold': fold + 1,
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss
            }, ckpt_path)

    logger.info("Cross-validation training completed successfully.")

if __name__ == "__main__":
    main()
