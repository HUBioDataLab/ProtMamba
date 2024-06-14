from torch.utils.data import Dataset
import torch
from Bio import SeqIO
import numpy as np
from tqdm.auto import tqdm
import wandb
import random

from transformers import BertTokenizer

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



class EarlyStopping:
    def __init__(self, patience = 2 , verbose=False, delta= 1e-4):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience     = patience
        self.verbose      = verbose
        self.counter      = 0
        self.best_score   = None
        self.early_stop   = False
        self.val_loss_min = np.Inf
        self.delta        = delta

    def __call__(self, val_loss):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.val_loss_min = val_loss
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.val_loss_min = val_loss

            self.counter = 0


def train(model, train_loader, val_loader, mask_token_id, criterion, optimizer, metrics, run, tokenizer, device, epochs=5, patience = 5, log_freq = 1000):

    accuracy   = metrics["accuracy"]
    precision  = metrics["precision"]
    recall     = metrics["recall"]
    f1         = metrics["f1"]
    perplexity = metrics["perplexity"]
    wandb.watch(model, criterion, log = "all", log_freq = 10)
    model.train()  # Set model to training mode
    early_stopping = EarlyStopping(patience = patience, verbose=True)
    best_val_loss = np.inf
    best_train_acc = 0
    batch_ct = run.step
    
    for epoch in range(epochs):
        train_progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} Training", leave=False)
        model.train()
        metric_results = {  "training_loss"      : 0,
                            "training_accuracy"  : 0,
                            "training_precision" : 0,
                            "training_f1"        : 0,
                            "training_perplexity": 0,
                            "training_recall"    : 0
                            }
        for _, batch in enumerate(train_progress_bar):
            input_ids = batch[0].to(device)
            labels = batch[1].to(device)
            optimizer.zero_grad()
            
            outputs     = model(input_ids = input_ids.to(torch.int32))
            logits      = outputs.logits
            loss        = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            labels      = labels.to(dtype = torch.int32).view(-1)
            predictions = logits.argmax(dim = -1).view(-1).to(device)
            
            mask = labels == tokenizer.pad_token_id
            predictions[mask] = tokenizer.pad_token_id
       
            loss.backward()
            optimizer.step()
            batch_ct += 1
            metric_results["training_loss"]       += loss.item()
            metric_results["training_accuracy"]   += accuracy(predictions, labels).item()
            metric_results["training_precision"]  += precision(predictions, labels).item()
            metric_results["training_f1"]         += f1(predictions, labels).item()
            metric_results["training_recall"]     += recall(predictions, labels).item()
            metric_results["training_perplexity"] += perplexity(loss).item()
            
            if batch_ct % log_freq == 0:
                b_metric_results = {}
                for key in metric_results.keys():
                    b_metric_results[f"per_{log_freq}_batch_{key}"] = metric_results[key] / batch_ct
                run.log(b_metric_results, step = batch_ct)
                if batch_ct % (10 * log_freq) == 0: 
                    if b_metric_results["per_1000_batch_training_accuracy"] > best_train_acc:
                        print(f"Saving model at batch {batch_ct}")
                        torch.save(model.state_dict(), f"big_run_batch{batch_ct}")
                        best_train_acc = b_metric_results["per_1000_batch_training_accuracy"]
                        
            break
        for metric in metric_results.keys(): 
                metric_results[metric] = metric_results[metric]/ len(train_loader) 
                
        
                
        #print(f"Epoch {epoch}, Training metrics: {str(metric_results)}")
        run.log(metric_results, step = batch_ct)
                
        model.eval()
        # Evaluate on the validation set
        val_progress_bar = tqdm(val_loader, desc="Validating", leave=False)
        with torch.no_grad():
            metric_results = {"val_loss": 0,
                              "val_accuracy": 0,
                              "val_precision": 0,
                              "val_f1": 0,
                              "val_perplexity": 0,
                              "val_recall": 0
                              }
            for _, batch in enumerate(val_progress_bar):
                inputs = batch[0].to(device)
                labels = batch[1].to(device)
                output = model(inputs)
                logits = output.logits
                loss   = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
                labels = labels.to(dtype = torch.int32).view(-1)
                predictions = logits.argmax(dim = -1).view(-1).to(device)

                #predictions = [tokenizer.pad_token_id if labels[i] == tokenizer.pad_token_id else predictions[i] for i in range(len(predictions))]
                mask = labels == tokenizer.pad_token_id
                predictions[mask] = tokenizer.pad_token_id

                metric_results["val_loss"]       += loss.item()
                metric_results["val_accuracy"]   += accuracy(predictions, labels).item()
                metric_results["val_precision"]  += precision(predictions, labels).item()
                metric_results["val_f1"]         += f1(predictions, labels).item()
                metric_results["val_recall"]     += recall(predictions, labels).item()
                metric_results["val_perplexity"] += perplexity(loss).item()
                
            for metric in metric_results.keys(): 
                metric_results[metric] = metric_results[metric]/ len(val_loader) 
                
            
            #print(f"Epoch {epoch}, Validation Metrics: {str(metric_results)}")
            run.log(metric_results, step = batch_ct)

            early_stopping(metric_results["val_loss"])
            if metric_results["val_loss"] < best_val_loss:
                print(f"Validation loss decreased ({best_val_loss:.6f} --> {metric_results['val_loss']:.6f}). Saving model...")
                torch.save(model.state_dict(), f"{run.name}")
                best_val_loss = metric_results["val_loss"]
            if early_stopping.early_stop:
                print("Early stopping")
                break
        break

def test(model, test_loader, criterion, device, tokenizer, run, metrics):
    accuracy   = metrics["accuracy"]
    precision  = metrics["precision"]
    recall     = metrics["recall"]
    f1         = metrics["f1"]
    perplexity = metrics["perplexity"]
    
    model.eval()  # Set model to evaluation mode
    metric_results = {  "test_loss"      : 0,
                        "test_accuracy"  : 0,
                        "test_precision" : 0,
                        "test_f1"        : 0,
                        "test_perplexity": 0,
                        "test_recall"    : 0
                      }
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing", leave=False):
            input_ids = batch[0].to(device)
            labels = batch[1].to(device)
            
            outputs = model(input_ids=input_ids)
            logits = outputs.logits
        
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            labels = labels.to(dtype = torch.int32).view(-1)
            predictions = logits.argmax(dim = -1).view(-1).to(device)
            mask = labels == tokenizer.pad_token_id
            predictions[mask] = tokenizer.pad_token_id
            
            metric_results["test_loss"]       += loss.item()
            metric_results["test_accuracy"]   += accuracy(predictions, labels).item()
            metric_results["test_precision"]  += precision(predictions, labels).item()
            metric_results["test_f1"]         += f1(predictions, labels).item()
            metric_results["test_recall"]     += recall(predictions, labels).item()
            metric_results["test_perplexity"] += perplexity(loss).item()

        for metric in metric_results.keys(): 
            metric_results[metric] /= len(test_loader)
                
    run.log(metric_results)
    
    
