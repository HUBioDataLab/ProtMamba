import torch

from transformers import MambaForCausalLM
from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

import argparse
import wandb

from datasets import load_dataset
from eval_metrics import compute_metrics
from utils import set_seed, LogTrainingMetricsCallback

set_seed(42)

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type = int, help = "number of epochs to train the model", default = 25) 
parser.add_argument("--batch_size", type = int, help = "batch size per device for both training and evaluation", default = 128) 
parser.add_argument("--nhl", type = int, help = "number of hidden layers in the model",  default = 32) 
parser.add_argument("--state_size" , type = int, help = "the size of the state space latents used in the model", default = 16) 
parser.add_argument("--hidden_size" , type = int, help = "dimensionality of the embeddings and hidden states", default = 768) 
parser.add_argument("--max_length" , type = int, help = "context size, or length of each input sequence to the model", default = 512)
parser.add_argument("--lr", type = float, help = "maximum value of learning rate to be used during training", default = 1e-3) 
parser.add_argument("--warmup_ratio", type = float, help =  "ratio of the learning rate warm up phase to the entire trining", default = 0.0001)
parser.add_argument("--weight_decay", type = float, help = "l2 regularization constant", default = 0.001)
parser.add_argument("--max_grad_norm", type = float, help = "maximum gradient norm threshold for norm clipping", default = 1.0)
parser.add_argument("--device_index", type = int, help = "index of the cuda device", default = 0)
parser.add_argument("--tokenizer_path", type = str, help = "path to the pretrained tokenizer, either huggingface hub directory or local directory")
parser.add_argument("--model_save_path", type = str, help = "directory to save the model weights and optimizer state")
parser.add_argument("--dataset_path", type = str, help = "path to the dataset in the fasta file format")
parser.add_argument("--model_load_path", type = str, help = "the path to the model state dict if not training from zero" )
parser.add_argument("--wandb_run_id", type = str, help = "provide wandb run id if continuing a previous run")
parser.add_argument("--wandb_run_name", type = str, help  = "provide wandb run name for new runs")
parser.add_argument("--wandb_project_name", type = str, help  = "provide wandb project name for new runs")
parser.add_argument("--log_freq", type = int, help = "frequency of logging training metrics to wandb, provide in terms of training steps", default = 5000) 
parser.add_argument("--eval_freq", type = int, help = "frequency of evaluation step during training, provide in terms of training steps", default = 30000)
parser.add_argument("--num_data", type = int, help = "total number of protein sequences to use in UniRef50 dataset", default = int(60e6))
args = parser.parse_args()

device = torch.device(f'cuda:{args.device_index}' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained("/home/mennan/esm2_t6_8M_UR50D", padding_side = "left")

def train():
    datasets = {'train' : load_dataset("Mennan/dnmt")['train'],
            'test'  : load_dataset("Mennan/dnmt")['test'],
            'valid' : load_dataset("Mennan/dnmt")['valid']}
    run = wandb.init(
        project = args.wandb_project_name,
        mode =  'online',
        name = args.wandb_run_name
    )  
    training_args = TrainingArguments(
        gradient_accumulation_steps=1,
        output_dir = f"{args.model_save_path}{args.wandb_run_name}",
        num_train_epochs = args.epochs,
        per_device_train_batch_size= args.batch_size,
        per_device_eval_batch_size= args.batch_size,
        weight_decay= args.weight_decay,
        logging_dir = f"{args.model_save_path}/logs/",
        logging_steps=args.log_freq,
        evaluation_strategy="steps",
        eval_steps=args.eval_freq,
        eval_on_start = False,
        report_to="wandb",
        learning_rate=args.lr,
        save_total_limit=1,
        greater_is_better=False,
        metric_for_best_model='eval_loss',
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        save_strategy='steps',
        save_steps = args.eval_freq,
        bf16 = True,
        resume_from_checkpoint = args.model_load_path 
    )
    model = MambaForCausalLM.from_pretrained(args.model_load_path)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer = tokenizer,
        mlm = False
    )
    callbacks = [
        LogTrainingMetricsCallback(run)
    ]
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets['train'],
        eval_dataset=datasets['test'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks = callbacks
    )
    trainer.train()
    metrics = trainer.evaluate()
    print(metrics)
    run.finish()
if __name__ == "__main__":
    train()
    
