
import torch
import numpy as np
from tqdm.auto import tqdm
import wandb
from early_stop import EarlyStopping

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
