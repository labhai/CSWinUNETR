"""Dataset loaders and training splits."""

from .ffhq_wrinkle import SPLIT_DIR, FFHQWrinkle, validate_splits

__all__ = ["SPLIT_DIR", "FFHQWrinkle", "validate_splits"]
