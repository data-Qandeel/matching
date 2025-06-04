"""
DeezyMatch: A simple contrastive learning framework for product matching.
"""
from DeezyMatch.model import DeezyMatchEncoder
from DeezyMatch.dataset import ContrastiveProductDataset
from DeezyMatch import utils
from DeezyMatch.predictor import Predictor

__version__ = "0.4.0"

__all__ = [
    "DeezyMatchEncoder",
    "ContrastiveProductDataset",
    "utils",
    "Predictor",
    "__version__",
]