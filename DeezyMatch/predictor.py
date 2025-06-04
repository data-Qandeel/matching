import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Union, List, Tuple, Optional

from DeezyMatch.model import DeezyMatchEncoder
from DeezyMatch.utils import CheckpointManager, setup_logging
import config

logger = logging.getLogger(__name__)

class Predictor:
    """Efficient text embedding and similarity ranking."""
    
    def __init__(self):
        """Initialize predictor."""
        # Set up basic configuration
        self._setup_logging()
        self.device = torch.device(config.DEVICE)
        self.max_len = config.MAX_LEN
        
        # Load model components
        self._load_tokenizer()
        self._load_model()
        
        logger.info(f"Predictor ready on {self.device}")

    def _setup_logging(self):
        """ Configure logging. """
        setup_logging()

    def _load_tokenizer(self):
        """Load and configure the tokenizer."""
        self.tokenizer = Tokenizer.from_file(config.TOKENIZER_PATH)
        self.pad_token_id = self.tokenizer.token_to_id("<pad>") or 0
        
        self.tokenizer.enable_padding(
            pad_id=self.pad_token_id,
            pad_token="<pad>",
            length=self.max_len
        )
        self.tokenizer.enable_truncation(max_length=self.max_len)
        
        logger.info(f"Tokenizer loaded (vocab: {self.tokenizer.get_vocab_size()})")

    def _load_model(self):
        """Initialize and load the encoder model."""
        # Create model
        self.model = DeezyMatchEncoder(
            vocab_size=self.tokenizer.get_vocab_size(),
            **config.MODEL_CONFIG
        )
        
        # Load checkpoint
        checkpoint_path = config.get_checkpoint_path()
        CheckpointManager.load(checkpoint_path, self.model, map_location=self.device)
        
        # Prepare model for inference
        self.model.to(self.device).eval()
        self.embedding_dim = config.MODEL_CONFIG['final_embedding_dim']

    def encode_batch(self, texts: List[str]) -> torch.Tensor:
        """Encode batch of texts."""
        if not texts:
            return torch.empty((0, self.embedding_dim), device=self.device)

        # Tokenize
        encoded_batch = self.tokenizer.encode_batch(texts)
        tokens = torch.tensor(
            [encoding.ids for encoding in encoded_batch],
            device=self.device
        )
        
        # Get embeddings
        with torch.no_grad():
            return self.model.encode(tokens)

    def _prepare_texts(self, texts: Union[str, List[str], pd.Series]) -> List[str]:
        """Normalize input texts to list of strings."""
        if isinstance(texts, str):
            return [texts]
        if isinstance(texts, pd.Series):
            return texts.fillna("").astype(str).tolist()
        return [str(x) if x is not None else "" for x in texts]

    def rank_candidates(
        self,
        queries: Union[str, List[str], pd.Series],
        candidates: Union[List[str], pd.Series],
        top_k: int = 10,
        batch_size: int = 256,
        show_progress: bool = False,
        price_weight: float = 0.0,
        query_prices: Optional[np.ndarray] = None,
        candidate_prices: Optional[np.ndarray] = None,
        price_bandwidth: float = 2.0
    ) -> Union[List[Tuple[str, float, int]], List[List[Tuple[str, float, int]]]]:
        """Rank candidates by similarity to queries, optionally using price."""
        # Prepare inputs
        is_single_query = isinstance(queries, str)
        queries = self._prepare_texts(queries)
        candidates = self._prepare_texts(candidates)
        
        # Handle empty inputs
        if not queries or not candidates:
            return [] if is_single_query else [[] for _ in queries]

        # Check if using price similarity
        using_prices = (price_weight > 0 and 
                       query_prices is not None and 
                       candidate_prices is not None)
        
        # Convert price arrays if needed
        if using_prices:
            query_prices = self._to_tensor(query_prices[:len(queries)])
            candidate_prices = self._to_tensor(candidate_prices[:len(candidates)])

        # First encode and normalize all candidates (with caching)
        candidate_embeddings = self._encode_in_batches(candidates, batch_size, "Encoding candidates", show_progress)
        candidate_embeddings = F.normalize(candidate_embeddings, dim=1)

        k = min(top_k, len(candidates))
        results = []
        
        # Process queries in batches
        for i in tqdm(range(0, len(queries), batch_size), disable=not show_progress, desc="Processing queries"):
            batch_queries = queries[i:i+batch_size]
            
            # Encode query batch
            with torch.no_grad():
                encoded_batch = self.tokenizer.encode_batch(batch_queries)
                query_tokens = torch.tensor(
                    [encoding.ids for encoding in encoded_batch],
                    device=self.device
                )
                query_embeddings = F.normalize(self.model.encode(query_tokens), dim=1)
            
            # Calculate similarities
            similarity = torch.matmul(query_embeddings, candidate_embeddings.T)
            
            # Apply price similarity if needed
            if using_prices:
                batch_query_prices = query_prices[i:i+batch_size]
                price_sim = self._compute_price_similarity(
                    batch_query_prices, candidate_prices, price_bandwidth
                )
                similarity = (1 - price_weight) * similarity + price_weight * price_sim
            
            # Get top-k for each query in batch
            top_scores, top_indices = torch.topk(similarity, k, dim=1)
            
            # Format results for this batch
            for scores, indices in zip(top_scores.cpu(), top_indices.cpu()):
                query_results = [
                    (candidates[idx.item()], score.item(), idx.item())
                    for score, idx in zip(scores, indices)
                    if not torch.isnan(score)
                ]
                results.append(query_results)
        
        # Return results in requested format
        return results[0] if is_single_query else results

    def _to_tensor(self, data) -> torch.Tensor:
        """Convert various data types to torch tensor."""
        if isinstance(data, torch.Tensor):
            return data.to(self.device)
        if isinstance(data, (pd.Series, pd.DataFrame)):
            data = data.to_numpy()
        if isinstance(data, (list, tuple)):
            data = np.array(data)
        return torch.tensor(data, dtype=torch.float, device=self.device)

    def _encode_in_batches(self, texts, batch_size, desc, show_progress):
        """Encode texts in batches with progress tracking."""
        embeddings = torch.zeros((len(texts), self.embedding_dim), device=self.device)
        
        for i in tqdm(range(0, len(texts), batch_size), disable=not show_progress, desc=desc):
            batch = texts[i:i+batch_size]
            embeddings[i:i+len(batch)] = self.encode_batch(batch)
            
        return embeddings

    def _compute_price_similarity(self, query_prices, candidate_prices, bandwidth):
        """Compute price similarity using Gaussian kernel."""
        # Apply log transform for better scaling
        eps = 1e-8
        q_log = torch.log(query_prices + eps).view(-1, 1)
        c_log = torch.log(candidate_prices + eps).view(1, -1)
        
        # Gaussian kernel for price similarity
        diff = torch.abs(q_log - c_log)
        return torch.exp(-(diff ** 2) / (2 * bandwidth ** 2))

    def cleanup(self):
        """Release temporary resources."""
        torch.cuda.empty_cache()