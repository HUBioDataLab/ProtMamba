from Bio import SeqIO
from datasets import Dataset
class UniRefDataset(Dataset):
    def __init__(self, dataset_path: str, tokenizer, device: torch.device, max_len: int, num_data: int = int(60e6)) -> None:
        self.device = device
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.num_data = num_data
        self.dataset_path = dataset_path
        self.seq_gen = None

    def __len__(self) -> int:
        return int(self.num_data)

    def __getitem__(self, index):
        if self.seq_gen is None:
            self.seq_gen = self.sequence_generator()
        return next(self.seq_gen)

    def sequence_generator(self):
        with open(self.dataset_path) as handle:
            for record in SeqIO.parse(handle, "fasta"):
                seq = str(record.seq)
                encoded = self.tokenizer(seq, padding='max_length', truncation=True, max_length=self.max_len, return_tensors='pt')
                input_ids = encoded['input_ids'].to(self.device)
                attention_mask = encoded['attention_mask'].to(self.device)
                yield {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'labels': input_ids.clone()
                }
                
