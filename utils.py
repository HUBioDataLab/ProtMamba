import torch
from transformers import TrainerCallback
import numpy as np
import os
import random
import math

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
class LogTrainingMetricsCallback(TrainerCallback):
    def __init__(self, run):
        self.run = run

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            # Calculate perplexity if 'loss' is in logs
            if 'loss' in logs:
                perplexity = math.exp(logs['loss'])
                logs['perplexity'] = perplexity
            
            self.run.log(logs)

    def on_train_begin(self, args, state, control, **kwargs):
        print("Training is starting...")

    def on_train_end(self, args, state, control, **kwargs):
        print("Training is ending...")
        
        