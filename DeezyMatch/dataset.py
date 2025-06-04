import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from tokenizers import Tokenizer
import logging
import random
from typing import Tuple, List

logger = logging.getLogger(__name__)


class ContrastiveProductDataset(Dataset):
    """Dataset for contrastive training using in-batch negatives.
    
    Returns positive pairs (anchor, positive) for encoder training.
    Positive (seller) texts can be optionally augmented by shuffling their words.
    During training, the model learns to map original marketplace names (anchors)
    close to both original and shuffled versions of their matching seller items.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: Tokenizer,
        max_len: int = 64,
        use_shuffled_positives: bool = False,
        shuffle_probability: float = 0.5,
        marketplace_col_ar: str = "marketplace_product_name_ar",
        marketplace_col_en: str = "marketplace_product_name_en",
        seller_col: str = "seller_item_name",
    ):
        """Initialize the dataset.
        
        Args:
            df: DataFrame containing product pairs.
            tokenizer: Pre-trained Tokenizer instance.
            max_len: Maximum sequence length for padding/truncation.
            use_shuffled_positives: If True, may return shuffled versions
                of marketplace names as anchors.
            shuffle_probability: Probability of returning shuffled version
                when use_shuffled_positives is True.
            marketplace_col_ar: Column name for Arabic marketplace item names.
            marketplace_col_en: Column name for English marketplace item names.
            seller_col: Column name for seller item names (positives).
        """
        if df.empty:
            raise ValueError("Input DataFrame cannot be empty.")

        # Validate required columns
        required_cols = {marketplace_col_ar, marketplace_col_en, seller_col}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        self.tokenizer = tokenizer
        self.max_len = max_len
        self.use_shuffled_positives = use_shuffled_positives
        self.shuffle_probability = max(0.0, min(1.0, shuffle_probability))
        
        # Configure tokenizer
        pad_id = tokenizer.token_to_id("<pad>")
        if pad_id is None:
            raise ValueError("'<pad>' token not found in tokenizer vocabulary.")
        self.tokenizer.enable_padding(pad_id=pad_id, pad_token="<pad>", length=max_len)
        self.tokenizer.enable_truncation(max_length=max_len)

        # Load and clean text data
        self.market_ar = df[marketplace_col_ar].fillna("").astype(str).tolist()
        self.market_en = df[marketplace_col_en].fillna("").astype(str).tolist()
        self.seller = df[seller_col].fillna("").astype(str).tolist()

        # Pre-compute language flags
        self.is_arabic = np.array([self._is_arabic(text) for text in self.seller], dtype=bool)
        self.n_items = len(self.seller)

        logger.info(
            f"Dataset initialized with {self.n_items:,} items. "
            f"Shuffle augmentation: {'ON' if use_shuffled_positives else 'OFF'} "
            f"(probability: {self.shuffle_probability:.1%})"
        )

    def __len__(self) -> int:
        return self.n_items

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get an (anchor, positive) pair.
        
        Args:
            idx: Index of the sample to retrieve.
            
        Returns:
            Tuple of (anchor_ids, positive_ids) as LongTensors.
        """
        if not 0 <= idx < self.n_items:
            raise IndexError(f"Index {idx} out of range [0, {self.n_items})")

        # Get marketplace text as anchor (unchanged)
        market_text = (
            self.market_ar[idx] 
            if self.is_arabic[idx] 
            else self.market_en[idx]
        )
        seller_text = self.seller[idx]

        # Apply shuffling to marketplace text with probability
        should_shuffle = (
            self.use_shuffled_positives 
            and random.random() < self.shuffle_probability
        )
        anchor_text = (
            self._shuffle_words(market_text) 
            if should_shuffle
            else market_text
        )

        # Tokenize (padding/truncation handled by tokenizer)
        anchor_ids = self.tokenizer.encode(anchor_text).ids
        positive_ids = self.tokenizer.encode(seller_text).ids

        return (
            torch.tensor(anchor_ids, dtype=torch.long),
            torch.tensor(positive_ids, dtype=torch.long)
        )

    @staticmethod
    def _is_arabic(text: str) -> bool:
        """Check if text contains Arabic characters."""
        if not isinstance(text, str):
            return False
        return any("\u0600" <= char <= "\u06FF" for char in text)

    @staticmethod
    def _shuffle_words(text: str) -> str:
        """Apply random word shuffling augmentation."""
        words = text.split()
        if len(words) <= 2:
            return text

        def basic_shuffle(w: List[str]) -> List[str]:
            w = w.copy()
            random.shuffle(w)
            return w

        def reverse_order(w: List[str]) -> List[str]:
            return w[::-1]
        
        def split_and_swap(w: List[str]) -> List[str]:
            half = len(w) // 2
            return w[half:] + w[:half]

        strategies = [
            basic_shuffle,
            reverse_order,
            split_and_swap
        ]

        return " ".join(random.choice(strategies)(words))