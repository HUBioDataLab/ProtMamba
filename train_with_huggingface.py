import torch
from transformers import TrainerCallback, BertTokenizer
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
import wandb
from torch.utils.data import random_split
from transformers import Trainer, TrainingArguments, EvalPrediction
from transformers import MambaForCausalLM, MambaConfig, DataCollatorForLanguageModeling
from Bio import SeqIO
import argparse
import numpy as np
import os
import random
from datasets import Dataset


os.environ["WANDB_DISABLE_CODE"] = "true"

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(42)
#count: int = 0

class UniRefDataset(Dataset):
    def __init__(self, dataset_path: str, tokenizer, device: torch.device, max_len: int, num_data: int = 1e5, random_crop: bool = False) -> None:
        self.device = device
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.num_data = num_data
        self.random_crop = random_crop
        self.seq_gen = self.sequence_generator(fasta_path = dataset_path)
                
    def sequence_generator(self, fasta_path):
        with open(fasta_path) as handle:
            for record in SeqIO.parse(handle, "fasta"):
                seq = record.seq
                seq = ' '.join(seq)
                seq = tokenizer(seq, padding='max_length', truncation=True, max_length = self.max_len)  # Customize as needed
                input_ids = torch.tensor(seq['input_ids'], dtype=torch.long).unsqueeze(0)  # Add batch dimension
                attention_mask = torch.tensor(seq['attention_mask'], dtype=torch.long).unsqueeze(0)  # Add batch dimension
                yield {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': input_ids}

    def __len__(self) -> int:
        return int(self.num_data)
    
    def __getitem__(self, index):
        return next(self.seq_gen)
    
class LogTrainingMetricsCallback(TrainerCallback):
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            global run
            run.log(logs)
               
def compute_metrics(pred: EvalPrediction):
    global tokenizer
    global device
    logits = pred.predictions
    labels = pred.label_ids
    non_special_tokens_mask=        (labels != tokenizer.pad_token_id) & \
                                    (labels != tokenizer.bos_token_id) & \
                                    (labels != tokenizer.eos_token_id) & \
                                    (labels != tokenizer.cls_token_id) & \
                                    (labels != tokenizer.sep_token_id)
                                  
    # Ensure that the predictions (logits) are valid tensors
    if not isinstance(logits, torch.Tensor):
        logits = torch.tensor(logits)  # Convert to PyTorch tensor if necessary
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels)  # Convert to PyTorch tensor if necessary
    if not isinstance(non_special_tokens_mask, torch.Tensor):
        non_special_tokens_mask = torch.tensor(non_special_tokens_mask)
    ###DO PROPER MASKING
    predictions = logits.argmax(dim=-1)
    predictions = torch.masked_select(predictions, non_special_tokens_mask)
    labels = torch.masked_select(labels, non_special_tokens_mask)

    predictions = predictions.cpu().numpy()
    labels = labels.cpu().numpy()
    # Compute metrics
    precision = precision_score(labels, predictions, average='macro')
    recall = recall_score(labels, predictions, average='macro')
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='macro')
    return {
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "f1": f1
    }

# Argument Parsing
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type = int, default = 1)
parser.add_argument("--batch_size" , type = int, default = 128)
parser.add_argument("--lr", type = float, default = 1e-2)
parser.add_argument("--max_length" , type = int, default = 1024)
parser.add_argument("--patience" , type = int, default = 10)
parser.add_argument("--masking_prob", type = float,  default = 0.15)
parser.add_argument("--nhl", help = "number of hidden layers",  default = 32)
parser.add_argument("--weight_decay", type = float, default = 1e-3)
parser.add_argument("--state_size" , type = int, default = 16)
parser.add_argument("--hidden_size" , type = int, default = 768)
parser.add_argument("--random_crop", type=bool, default = True)
parser.add_argument("--device_index", type = int, help = "index of the cuda device", default = 0)
parser.add_argument("--dataset_path", type = str, default = "/media/ubuntu/8TB/mennan/data/uniref50.fasta")
parser.add_argument("--model_load_path", type = str, help = "provide the path to the model state dict if not training from zero", default = None)
parser.add_argument("--model_save_path", type = str, default = "/media/ubuntu/8TB/mennan/model_checkpoints_A/")
parser.add_argument("--wandb_run_id", type = str, help = "provide wandb run id if continuing a previous run", default = None)
parser.add_argument("--wandb_run_name", type = str, help  = "provide wandb run name for new runs", default = None)
parser.add_argument("--log_freq", type = int, default = 100)
parser.add_argument("--num_data", type = int, default = 50e6)
parser.add_argument("--lr_warmup_steps", type = int, default = 500)
args = parser.parse_args()

# Tokenizer setup using BertTokenizerFast
tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
if tokenizer.eos_token is None:
    tokenizer.add_special_tokens({'eos_token': '[EOS]'})
if tokenizer.bos_token is None:
    tokenizer.add_special_tokens({'bos_token': '[BOS]'})
print(tokenizer.vocab)


device = torch.device(f'cuda:{args.device_index}' if torch.cuda.is_available() else 'cpu')

config_mamba = MambaConfig(
    vocab_size=tokenizer.vocab_size,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    bos_token_id=tokenizer.bos_token_id,
    hidden_size=args.hidden_size,
    num_hidden_layers=args.nhl,
    state_size=args.state_size,
    use_cache = False
)

model = MambaForCausalLM(config=config_mamba)

dataset = UniRefDataset(
    dataset_path=args.dataset_path,
    tokenizer=tokenizer,
    num_data=args.num_data,
    max_len=args.max_length,
    device = device
)
len_dataset = len(dataset)
train_size = int(0.8 * len_dataset)
test_size = int(0.1)
val_size = int(0.001 * len_dataset)
discard_size = int(len_dataset - train_size - test_size - val_size)
train_dataset, test_dataset, val_dataset,_ = random_split(dataset, [train_size, test_size,  val_size, discard_size])

config_param = {
    "dataset": "Uniref50",
    "architecture": "MambaForCausalLM",
    "train_method": "autoregressive",
    "epochs": args.epochs,
    "batch_size": args.batch_size,
    "learning_rate": args.lr,
    "max_length": args.max_length,
    "patience": args.patience,
    "num_hidden_layers": args.nhl,
    "weight_decay": args.weight_decay,
    "hidden_size": args.hidden_size,
    "random_crop": True,
    "state_size": args.state_size,
    "num_data": len_dataset
}
wandb.login()
run = wandb.init(
    project="protmamba2",
    config=config_param,
    resume=False,
    name=f"{args.wandb_run_name}",
    mode="online"
)
print(len_dataset)
del len_dataset

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    warmup_steps=args.lr_warmup_steps,
    weight_decay=args.weight_decay,
    logging_dir='./logs',
    logging_steps = args.log_freq,
    evaluation_strategy="steps",
    eval_steps = 10 * args.log_freq,
    save_total_limit=1,
    report_to="wandb",
    learning_rate = args.lr
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[LogTrainingMetricsCallback(tokenizer=tokenizer, model=model)]
)

trainer.train()
metrics = trainer.evaluate()
print(metrics)
trainer.save_model(f"{args.model_save_path}{run.name}")
