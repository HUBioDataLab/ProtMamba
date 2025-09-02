"""
Dataset module for protein sequence data loading and processing.
"""
import torch
from torch.utils.data import random_split
from Bio import SeqIO
from datasets import Dataset
from transformers import DataCollatorForLanguageModeling


class UniRefDataset(Dataset):
    """
    Dataset class for loading and processing UniRef protein sequences.
    """
    
    def __init__(
        self,
        dataset_path: str,
        tokenizer,
        device: torch.device,
        max_len: int,
        num_data: int = int(1e5),
        random_crop: bool = False
    ):
        """
        Initialize UniRef dataset.
        
        Args:
            dataset_path: Path to the FASTA file
            tokenizer: Tokenizer for encoding sequences
            device: Device to load tensors to
            max_len: Maximum sequence length
            num_data: Number of data samples to use
            random_crop: Whether to use random cropping
        """
        self.device = device
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.num_data = num_data
        self.random_crop = random_crop
        self.dataset_path = dataset_path
        self.seq_gen = None

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return int(self.num_data)

    def __getitem__(self, index):
        """
        Get a single sample from the dataset.
        
        Args:
            index: Index of the sample (not used in generator-based approach)
            
        Returns:
            Dictionary containing input_ids, attention_mask, and labels
        """
        if self.seq_gen is None:
            self.seq_gen = self.sequence_generator()
        return next(self.seq_gen)

    def sequence_generator(self):
        """
        Generator function for loading sequences from FASTA file.
        
        Yields:
            Dictionary containing tokenized sequence data
        """
        with open(self.dataset_path) as handle:
            for record in SeqIO.parse(handle, "fasta"):
                seq = str(record.seq)
                
                # Apply random cropping if enabled
                if self.random_crop and len(seq) > self.max_len:
                    start_idx = torch.randint(0, len(seq) - self.max_len + 1, (1,)).item()
                    seq = seq[start_idx:start_idx + self.max_len]
                
                # Tokenize the sequence
                encoded = self.tokenizer(
                    seq,
                    padding='max_length',
                    truncation=True,
                    max_length=self.max_len,
                    return_tensors='pt'
                )
                
                input_ids = encoded['input_ids'].squeeze(0).to(self.device)
                attention_mask = encoded['attention_mask'].squeeze(0).to(self.device)
                
                yield {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'labels': input_ids.clone()
                }


def create_data_splits(dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Split dataset into train, validation, and test sets.
    
    Args:
        dataset: The dataset to split
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        test_ratio: Proportion of data for testing
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Split ratios must sum to 1.0"
    
    dataset_size = len(dataset)
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    test_size = int(test_ratio * dataset_size)
    
    # Adjust for rounding errors
    discard_size = dataset_size - train_size - val_size - test_size
    
    if discard_size > 0:
        train_dataset, val_dataset, test_dataset, _ = random_split(
            dataset, [train_size, val_size, test_size, discard_size]
        )
    else:
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size]
        )
    
    return train_dataset, val_dataset, test_dataset


def get_data_collator(tokenizer):
    """
    Create data collator for language modeling.
    
    Args:
        tokenizer: Tokenizer to use for data collation
        
    Returns:
        DataCollatorForLanguageModeling instance
    """
    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
        return_tensors='pt'
    )