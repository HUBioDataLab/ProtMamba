"""
Main training script for Protein Language Model.
"""
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from config.training_config import (
    get_args_parser,
    parse_configs
)
from utils.seed import (
    set_seed,
    setup_environment,
    get_device,
    print_model_stats
)
from data.dataset import (
    UniRefDataset,
    create_data_splits,
    get_data_collator
)
from models.mamba_model import (
    setup_model_and_tokenizer
)
from training.trainer import (
    create_training_arguments,
    initialize_wandb,
    create_callbacks,
    setup_trainer,
    run_training
)


def main():
    """Main training function."""
    
    # Parse arguments
    parser = get_args_parser()
    args = parser.parse_args()
    
    # Parse configs
    model_config, training_config, data_config, system_config = parse_configs(args)
    
    # Print configuration
    print("="*60)
    print("TRAINING CONFIGURATION")
    print("="*60)
    print(f"Model Configuration:")
    print(f"  Hidden size: {model_config.hidden_size}")
    print(f"  Number of layers: {model_config.num_hidden_layers}")
    print(f"  State size: {model_config.state_size}")
    print(f"\nTraining Configuration:")
    print(f"  Batch size: {training_config.batch_size}")
    print(f"  Learning rate: {training_config.learning_rate}")
    print(f"  Epochs: {training_config.epochs}")
    print(f"  Weight decay: {training_config.weight_decay}")
    print(f"\nData Configuration:")
    print(f"  Dataset path: {data_config.dataset_path}")
    print(f"  Max length: {data_config.max_length}")
    print(f"  Number of samples: {data_config.num_data}")
    print("="*60 + "\n")
    
    # Setup environment
    setup_environment()
    set_seed(system_config.seed)
    
    # Get device
    device = get_device(system_config.device_index)
    
    # Initialize WandB
    run = initialize_wandb(
        system_config,
        training_config,
        model_config,
        data_config
    )
    
    # Setup model and tokenizer
    print("\nSetting up model and tokenizer...")
    model_config_dict = {
        'hidden_size': model_config.hidden_size,
        'num_hidden_layers': model_config.num_hidden_layers,
        'state_size': model_config.state_size,
        'use_cache': model_config.use_cache
    }
    
    model, tokenizer, mamba_config = setup_model_and_tokenizer(
        model_config_dict,
        data_config.tokenizer_path,
        system_config.model_load_path
    )
    
    # Print model statistics
    print_model_stats(model)
    
    # Move model to device
    model = model.to(device)
    
    # Create dataset
    print("\nCreating dataset...")
    dataset = UniRefDataset(
        dataset_path=data_config.dataset_path,
        tokenizer=tokenizer,
        device=device,
        max_len=data_config.max_length,
        num_data=data_config.num_data,
        random_crop=data_config.random_crop
    )
    
    # Create data splits
    print("Creating train/val/test splits...")
    train_dataset, val_dataset, test_dataset = create_data_splits(
        dataset,
        train_ratio=data_config.train_split,
        val_ratio=data_config.val_split,
        test_ratio=data_config.test_split
    )
    
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    
    # Get data collator
    data_collator = get_data_collator(tokenizer)
    
    # Create training arguments
    output_dir = os.path.join(
        system_config.model_save_path,
        system_config.wandb_run_name
    )
    
    training_args = create_training_arguments(
        output_dir=output_dir,
        training_config=training_config,
        system_config=system_config,
        resume_from_checkpoint=system_config.model_load_path
    )
    
    # Create callbacks
    callbacks = create_callbacks(
        run=run,
        patience=system_config.patience,
        gpu_log_freq=100,
        batch_log_freq=100
    )
    
    # Setup trainer
    print("\nSetting up trainer...")
    trainer = setup_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        training_args=training_args,
        callbacks=callbacks
    )
    
    # Run training
    print("\nStarting training...")
    final_metrics = run_training(trainer, run)
    
    # Optional: Test set evaluation
    if test_dataset is not None:
        print("\nEvaluating on test set...")
        test_metrics = trainer.evaluate(eval_dataset=test_dataset)
        print("\nTest Set Metrics:")
        for key, value in test_metrics.items():
            print(f"  {key}: {value:.4f}")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()