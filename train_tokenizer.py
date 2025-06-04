import os
import pandas as pd
from tokenizers import Tokenizer, normalizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import NFKC
from tqdm import tqdm
from typing import List, Iterator
import logging

# Import config
import config
from DeezyMatch.utils import setup_logging, minimal_clean_text

# Setup logging
setup_logging(log_file=os.path.join(config.LOG_DIR, 'tokenizer_training.log'))
logger = logging.getLogger(__name__)

# Configuration
TEXT_COLUMNS = ['product_name', 'product_name_ar']
VOCAB_SIZE = 512
MIN_FREQUENCY = 2
SPECIAL_TOKENS = ["<pad>", "<unk>"]
TOKENIZER_OUTPUT_PATH = config.TOKENIZER_PATH
CSV_PATH = os.path.join(config.DATA_DIR, 'raw', 'master_file_activated.csv')
CHUNK_SIZE = 5000

def get_training_corpus(filepath: str, text_columns: List[str]) -> Iterator[List[str]]:
    """Yield batches of cleaned text from CSV columns."""
    logger.info(f"Reading training data from: {filepath}")
    
    if not os.path.exists(filepath):
        logger.error(f"Training data file not found: {filepath}")
        raise FileNotFoundError(f"Training data file not found: {filepath}")

    # Validate columns
    header_df = pd.read_csv(filepath, nrows=0)
    available_columns = [col for col in text_columns if col in header_df.columns]
    if not available_columns:
        logger.error(f"No valid text columns found in {filepath}")
        raise ValueError(f"No valid text columns found: {text_columns}")

    reader = pd.read_csv(filepath, chunksize=CHUNK_SIZE, usecols=available_columns)
    
    for chunk in tqdm(reader, desc="Processing chunks"):
        batch_texts = []
        for col in available_columns:
            cleaned_text = chunk[col].fillna("").astype(str).apply(minimal_clean_text)
            batch_texts.extend([text for text in cleaned_text if text])
        
        if batch_texts:
            yield batch_texts

def create_tokenizer() -> Tokenizer:
    """Initialize BPE tokenizer with normalization."""
    logger.info("Creating BPE tokenizer instance")
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.normalizer = normalizers.Sequence([NFKC()])
    tokenizer.pre_tokenizer = Whitespace()
    return tokenizer

def train_and_save_tokenizer():
    """Main function to train and save tokenizer."""
    logger.info("=== Starting Tokenizer Training ===")
    
    try:
        # Initialize components
        tokenizer = create_tokenizer()
        trainer = BpeTrainer(
            vocab_size=VOCAB_SIZE,
            min_frequency=MIN_FREQUENCY,
            special_tokens=SPECIAL_TOKENS
        )
        
        # Train and save
        corpus = get_training_corpus(CSV_PATH, TEXT_COLUMNS)
        tokenizer.train_from_iterator(corpus, trainer=trainer)
        
        os.makedirs(os.path.dirname(TOKENIZER_OUTPUT_PATH), exist_ok=True)
        tokenizer.save(TOKENIZER_OUTPUT_PATH)
        
        logger.info(f"Tokenizer saved to {TOKENIZER_OUTPUT_PATH}")
        logger.info("=== Training Completed Successfully ===")
        
        return True
    
    except Exception as e:
        logger.error(f"Tokenizer training failed: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    train_and_save_tokenizer()