"""Data processing module for protein sequences."""
from .dataset import (
    UniRefDataset,
    create_data_splits,
    get_data_collator
)

__all__ = [
    'UniRefDataset',
    'create_data_splits',
    'get_data_collator'
]