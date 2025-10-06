import os
import logging
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from tokenizers import Tokenizer

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
    # Configuration
    device = torch.device(config.DEVICE)
    logger.info("="*60)
    logger.info(f"{'TRAINING CONFIGURATION':^60}")
    log_config("Device", device)
    log_config("Batch Size", config.TRAINING_CONFIG['batch_size'])
    logger.info("="*60)

    # Load data
    try:
        df = pd.read_csv(config.TRAIN_DATA, low_memory=False)
        logger.info(f"Loaded {len(df):,} training examples")
    except Exception as e:
        logger.error(f"Data loading failed: {e}")
        return

    # Initialize tokenizer
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

    # Initialize model
    try:
        model = DeezyMatchEncoder(
            vocab_size=tokenizer.get_vocab_size(),
            **config.MODEL_CONFIG
        ).to(device)
        log_config("Trainable Parameters", 
                 f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    except Exception as e:
        logger.error(f"Model initialization failed: {e}")
        return

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.TRAINING_CONFIG['learning_rate'],
        weight_decay=config.TRAINING_CONFIG['weight_decay']
    )

    # Resume from checkpoint if available
    start_epoch = 0
    checkpoint_path = config.get_checkpoint_path()
    if checkpoint_path:
        logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        try:
            checkpoint = utils.CheckpointManager.load(checkpoint_path, model, optimizer, device)
            start_epoch = checkpoint['epoch']
            logger.info(f"Resumed training from epoch {start_epoch}")
        except Exception as e:
            logger.warning(f"Checkpoint loading failed: {e}. Starting fresh.")

    # Prepare dataset
    dataset = ContrastiveProductDataset(
        df=df,
        tokenizer=tokenizer,
        max_len=config.MAX_LEN,
        use_shuffled_positives=config.TRAINING_CONFIG['use_shuffled_positives'],
        shuffle_probability=config.TRAINING_CONFIG['shuffle_probability']
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.TRAINING_CONFIG['batch_size'],
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )

    # Training loop
    for epoch in range(start_epoch + 1, config.TRAINING_CONFIG['num_epochs'] + 1):
        model.train()
        epoch_loss = 0.0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
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
            epoch_loss += batch_loss
            progress_bar.set_postfix({'batch_loss': f'{batch_loss:.4f}'})

        avg_loss = epoch_loss / len(dataloader)
        logger.info(f"Epoch {epoch} | Loss: {avg_loss:.4f}")

        # Save checkpoint
        ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"encoder_epoch_{epoch}.pth")
        utils.CheckpointManager.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss
        }, ckpt_path)

    logger.info("Training completed successfully")

if __name__ == "__main__":
    main()