"""
Custom callbacks for training monitoring and logging.
"""
import math
import torch
from transformers import TrainerCallback


class GPUMonitorCallback(TrainerCallback):
    """
    Callback for monitoring GPU memory usage during training.
    """
    
    def __init__(self, run=None, log_freq=100):
        """
        Initialize GPU monitor callback.
        
        Args:
            run: WandB run object for logging
            log_freq: Frequency of logging in steps
        """
        self.run = run
        self.log_freq = log_freq
        self.num_gpus = torch.cuda.device_count()

    def get_gpu_stats(self):
        """
        Get GPU memory statistics.
        
        Returns:
            Dictionary containing GPU memory stats
        """
        stats = {}
        for gpu_id in range(self.num_gpus):
            # Memory allocated in bytes, convert to GB
            allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
            # Memory cached (reserved) in bytes, convert to GB
            reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3
            # Maximum memory allocated in bytes, convert to GB
            max_allocated = torch.cuda.max_memory_allocated(gpu_id) / 1024**3
            
            stats.update({
                f"gpu{gpu_id}_allocated_gb": allocated,
                f"gpu{gpu_id}_reserved_gb": reserved,
                f"gpu{gpu_id}_max_allocated_gb": max_allocated,
                f"gpu{gpu_id}_utilization_percent": (
                    (allocated / reserved * 100) if reserved > 0 else 0
                )
            })
        return stats

    def on_step_end(self, args, state, control, **kwargs):
        """Log GPU stats at specified frequency."""
        if state.global_step % self.log_freq == 0:
            gpu_stats = self.get_gpu_stats()
            
            # Log to wandb if available
            if self.run:
                self.run.log(gpu_stats, step=state.global_step)
            
            # Print to console
            print(f"\nStep {state.global_step} GPU Stats:")
            for k, v in gpu_stats.items():
                print(f"  {k}: {v:.2f}")


class LogTrainingMetricsCallback(TrainerCallback):
    """
    Callback for logging training metrics including perplexity.
    """
    
    def __init__(self, run=None):
        """
        Initialize metrics logging callback.
        
        Args:
            run: WandB run object for logging
        """
        self.run = run

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log training metrics including calculated perplexity."""
        if logs is not None:
            # Calculate perplexity if 'loss' is in logs
            if 'loss' in logs:
                try:
                    perplexity = math.exp(logs['loss'])
                    logs['perplexity'] = perplexity
                except OverflowError:
                    logs['perplexity'] = float('inf')
            
            # Log to wandb if available
            if self.run:
                self.run.log(logs)

    def on_train_begin(self, args, state, control, **kwargs):
        """Print message when training starts."""
        print("\n" + "="*50)
        print("Training is starting...")
        print("="*50 + "\n")

    def on_train_end(self, args, state, control, **kwargs):
        """Print message when training ends."""
        print("\n" + "="*50)
        print("Training completed successfully!")
        print("="*50 + "\n")


class BatchSizeCallback(TrainerCallback):
    """
    Callback for monitoring actual batch sizes during training.
    """
    
    def __init__(self, log_freq=100):
        """
        Initialize batch size monitoring callback.
        
        Args:
            log_freq: Frequency of logging in steps
        """
        self.log_freq = log_freq
        self.step_count = 0

    def on_step_begin(self, args, state, control, **kwargs):
        """Log actual batch size at specified frequency."""
        self.step_count += 1
        if self.step_count % self.log_freq == 0:
            if 'inputs' in kwargs and kwargs['inputs'] is not None:
                if hasattr(kwargs['inputs'], 'input_ids'):
                    actual_batch_size = kwargs['inputs'].input_ids.shape[0]
                    print(f"Step {self.step_count}: Actual batch size: {actual_batch_size}")


class EarlyStoppingCallback(TrainerCallback):
    """
    Callback for early stopping based on validation loss.
    """
    
    def __init__(self, patience=10, min_delta=0.0001):
        """
        Initialize early stopping callback.
        
        Args:
            patience: Number of epochs with no improvement before stopping
            min_delta: Minimum change to qualify as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.wait = 0

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Check if training should stop based on validation loss."""
        if metrics is not None and 'eval_loss' in metrics:
            current_loss = metrics['eval_loss']
            
            if current_loss < self.best_loss - self.min_delta:
                self.best_loss = current_loss
                self.wait = 0
                print(f"Validation loss improved to {current_loss:.4f}")
            else:
                self.wait += 1
                print(f"No improvement for {self.wait} evaluations (best: {self.best_loss:.4f})")
                
                if self.wait >= self.patience:
                    print(f"Early stopping triggered after {self.wait} evaluations without improvement")
                    control.should_training_stop = True
        
        return control