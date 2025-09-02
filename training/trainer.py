"""
Training module for setting up and running the training process.
"""
import os
import wandb
from transformers import Trainer, TrainingArguments
from typing import Optional, List

from .callbacks import (
    GPUMonitorCallback,
    LogTrainingMetricsCallback,
    BatchSizeCallback,
    EarlyStoppingCallback
)
from .metrics import create_compute_metrics_fn


def create_training_arguments(
    output_dir: str,
    training_config,
    system_config,
    resume_from_checkpoint: Optional[str] = None
) -> TrainingArguments:
    """
    Create TrainingArguments for the Trainer.
    
    Args:
        output_dir: Directory to save model and logs
        training_config: Training configuration object
        system_config: System configuration object
        resume_from_checkpoint: Path to resume training from
        
    Returns:
        TrainingArguments object
    """
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=training_config.epochs,
        per_device_train_batch_size=training_config.batch_size,
        per_device_eval_batch_size=training_config.batch_size,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        learning_rate=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
        warmup_steps=training_config.warmup_steps,
        max_grad_norm=training_config.max_grad_norm,
        logging_dir=f"{output_dir}/logs/",
        logging_steps=training_config.log_freq,
        evaluation_strategy="steps",
        eval_steps=training_config.eval_freq,
        eval_on_start=training_config.eval_on_start,
        save_strategy="steps",
        save_steps=training_config.eval_freq,
        save_total_limit=training_config.save_total_limit,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        load_best_model_at_end=True,
        report_to="wandb" if system_config.wandb_mode != "disabled" else "none",
        bf16=training_config.bf16,
        resume_from_checkpoint=resume_from_checkpoint
    )


def initialize_wandb(
    system_config,
    training_config,
    model_config,
    data_config
):
    """
    Initialize Weights & Biases logging.
    
    Args:
        system_config: System configuration
        training_config: Training configuration
        model_config: Model configuration
        data_config: Data configuration
        
    Returns:
        WandB run object
    """
    run_config = {
        'batch_size': training_config.batch_size,
        'learning_rate': training_config.learning_rate,
        'epochs': training_config.epochs,
        'weight_decay': training_config.weight_decay,
        'warmup_steps': training_config.warmup_steps,
        'max_grad_norm': training_config.max_grad_norm,
        'hidden_size': model_config.hidden_size,
        'num_hidden_layers': model_config.num_hidden_layers,
        'state_size': model_config.state_size,
        'max_length': data_config.max_length,
        'num_data': data_config.num_data,
    }
    
    # Initialize wandb
    if system_config.wandb_run_id:
        # Resume existing run
        run = wandb.init(
            project=system_config.wandb_project,
            resume=True,
            id=system_config.wandb_run_id,
            mode=system_config.wandb_mode,
            config=run_config
        )
        print(f"Resumed WandB run: {system_config.wandb_run_id}")
    else:
        # Start new run
        run = wandb.init(
            project=system_config.wandb_project,
            resume=False,
            mode=system_config.wandb_mode,
            name=system_config.wandb_run_name,
            config=run_config
        )
        print(f"Started new WandB run: {wandb.run.id}")
    
    return run


def create_callbacks(
    run=None,
    patience: int = 10,
    gpu_log_freq: int = 100,
    batch_log_freq: int = 100
) -> List:
    """
    Create list of training callbacks.
    
    Args:
        run: WandB run object
        patience: Early stopping patience
        gpu_log_freq: GPU monitoring frequency
        batch_log_freq: Batch size monitoring frequency
        
    Returns:
        List of callback objects
    """
    callbacks = []
    
    # Add logging callback
    callbacks.append(LogTrainingMetricsCallback(run))
    
    # Add GPU monitoring if CUDA available
    if torch.cuda.is_available():
        callbacks.append(GPUMonitorCallback(run, gpu_log_freq))
    
    # Add batch size monitoring
    callbacks.append(BatchSizeCallback(batch_log_freq))
    
    # Add early stopping
    callbacks.append(EarlyStoppingCallback(patience))
    
    return callbacks


def setup_trainer(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    data_collator,
    training_args,
    callbacks
):
    """
    Set up the Hugging Face Trainer.
    
    Args:
        model: Model to train
        tokenizer: Tokenizer
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        data_collator: Data collator
        training_args: Training arguments
        callbacks: List of callbacks
        
    Returns:
        Configured Trainer object
    """
    # Create compute metrics function with tokenizer
    compute_metrics = create_compute_metrics_fn(tokenizer)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks
    )
    
    return trainer


def run_training(trainer, run=None):
    """
    Execute the training process.
    
    Args:
        trainer: Configured Trainer object
        run: Optional WandB run to finish after training
        
    Returns:
        Final evaluation metrics
    """
    try:
        # Start training
        trainer.train()
        
        # Final evaluation
        print("\nRunning final evaluation...")
        metrics = trainer.evaluate()
        
        print("\nFinal Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
        
        # Save final model
        print("\nSaving final model...")
        trainer.save_model()
        
        return metrics
        
    finally:
        # Ensure wandb run is finished
        if run:
            run.finish()
            print("WandB run finished")


import torch  # Import needed for GPU check in create_callbacks