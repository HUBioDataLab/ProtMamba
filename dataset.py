from torch.utils.data import Dataset
import torch
import random
from Bio import SeqIO

class UniProtDataset(Dataset):
    def __init__(self, tokenizer, device, masking_prob=0.15, num_data=408368587, random_crop=False, max_length=500, dataset_path="/home/mennan/uniprot_sprot.fasta"):
        '''
        Args:
            tokenizer: Tokenizer object to encode sequences.
            device: Device to which the tensors will be moved.
            masking_prob: Probability of masking tokens.
            num_data: Number of data samples to use.
            random_crop: Whether to randomly crop sequences.
            max_length: Maximum length of sequences.
            dataset_path: Path to the dataset file.
        '''
        self.dataset_path = dataset_path
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.device = device
        self.masking_prob = masking_prob
        self.random_crop = random_crop
        self.num_data = num_data
        
    
        self.seq_gen = self.sequence_generator(dataset_path)
        
   

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        sequence = " ".join(next(self.seq_gen))
        if self.random_crop and len(sequence) > self.max_length:
            start_idx = random.randint(0, len(sequence) - self.max_length - 1)
            sequence = sequence[start_idx:]
        encoded_sequence = self.tokenizer(sequence, add_special_tokens=True, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
        input_ids = encoded_sequence["input_ids"].squeeze(0)
        
        # Create mask where tokens are not special tokens
        special_tokens = {self.tokenizer.cls_token_id, self.tokenizer.sep_token_id, self.tokenizer.pad_token_id}
        mask = (torch.rand(len(input_ids)) < self.masking_prob) & (~torch.isin(input_ids, torch.tensor(list(special_tokens), dtype=torch.long)))
        
        # Apply mask to input ids
        masked_input_ids = torch.where(mask, torch.tensor(self.tokenizer.mask_token_id, dtype=torch.long), input_ids)
        
        # Labels are the original ids where masked, else pad token id
        labels = torch.where(mask, input_ids, torch.tensor(self.tokenizer.pad_token_id, dtype=torch.long))
        
        return masked_input_ids.to(self.device), labels.to(self.device)

    
    def sequence_generator(self, fasta_path):
        with open(fasta_path) as handle:
            for record in SeqIO.parse(handle, "fasta"):
                yield str(record.seq)
