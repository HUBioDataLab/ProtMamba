import torch
from tqdm import tqdm

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
    
    
