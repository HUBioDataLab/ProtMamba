"""
Training configuration module for Protein Language Model.
"""
import argparse
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for the Mamba model architecture."""
    hidden_size: int = 512
    num_hidden_layers: int = 24
    state_size: int = 16
    vocab_size: Optional[int] = None
    eos_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None
    bos_token_id: Optional[int] = None
    use_cache: bool = False


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    epochs: int = 10000
    learning_rate: float = 1e-4
    batch_size: int = 4096 * 8
    gradient_accumulation_steps: int = 1
    weight_decay: float = 0.001
    warmup_steps: int = 20000
    max_grad_norm: float = 5.0
    log_freq: int = 500
    eval_freq: int = 3000
    eval_on_start: bool = True
    save_total_limit: int = 1
    bf16: bool = True


@dataclass
class DataConfig:
    """Configuration for dataset and data processing."""
    dataset_path: str = "/gpfs/projects/etur06/mennan/uniref50.fasta"
    tokenizer_path: str = "/home/ibgc/ibgc814261/esm2_t6_8M_UR50D"
    max_length: int = 512
    num_data: int = int(60e6)
    masking_prob: float = 0.15
    random_crop: bool = True
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1


@dataclass
class SystemConfig:
    """Configuration for system and environment settings."""
    device_index: int = 0
    model_save_path: str = "/gpfs/projects/etur06/output/"
    model_load_path: Optional[str] = None
    wandb_run_id: Optional[str] = None
    wandb_run_name: str = "40M_512length_1000epoch_training"
    wandb_project: str = "protmamba2"
    wandb_mode: str = "offline"
    patience: int = 10
    seed: int = 42


def get_args_parser():
    """Create and return argument parser with all configuration options."""
    parser = argparse.ArgumentParser(description='Protein Language Model Training')
    
    # Model arguments
    parser.add_argument("--hidden_size", type=int, default=512,
                       help="Hidden size of the model")
    parser.add_argument("--nhl", type=int, default=24,
                       help="Number of hidden layers")
    parser.add_argument("--state_size", type=int, default=16,
                       help="State size for Mamba model")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=10000,
                       help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=4096 * 8,
                       help="Batch size for training")
    parser.add_argument("--weight_decay", type=float, default=0.001,
                       help="Weight decay for optimizer")
    parser.add_argument("--warmup_steps", type=int, default=20000,
                       help="Number of warmup steps")
    parser.add_argument("--max_grad_norm", type=float, default=5.0,
                       help="Maximum gradient norm for clipping")
    parser.add_argument("--log_freq", type=int, default=500,
                       help="Logging frequency in steps")
    parser.add_argument("--eval_freq", type=int, default=3000,
                       help="Evaluation frequency in steps")
    
    # Data arguments
    parser.add_argument("--dataset_path", type=str,
                       default="/gpfs/projects/etur06/mennan/uniref50.fasta",
                       help="Path to the dataset")
    parser.add_argument("--tokenizer_path", type=str,
                       default="/home/ibgc/ibgc814261/esm2_t6_8M_UR50D",
                       help="Path to the tokenizer")
    parser.add_argument("--max_length", type=int, default=512,
                       help="Maximum sequence length")
    parser.add_argument("--num_data", type=int, default=int(60e6),
                       help="Number of data samples to use")
    parser.add_argument("--masking_prob", type=float, default=0.15,
                       help="Masking probability for MLM")
    parser.add_argument("--random_crop", type=bool, default=True,
                       help="Whether to use random cropping")
    
    # System arguments
    parser.add_argument("--device_index", type=int, default=0,
                       help="Index of the CUDA device")
    parser.add_argument("--model_save_path", type=str,
                       default="/gpfs/projects/etur06/output/",
                       help="Path to save the model")
    parser.add_argument("--model_load_path", type=str, default=None,
                       help="Path to load a pretrained model")
    parser.add_argument("--wandb_run_id", type=str, default=None,
                       help="WandB run ID for resuming runs")
    parser.add_argument("--wandb_run_name", type=str,
                       default="40M_512length_1000epoch_training",
                       help="WandB run name")
    parser.add_argument("--patience", type=int, default=10,
                       help="Patience for early stopping")
    
    return parser


def parse_configs(args):
    """Parse command line arguments into configuration dataclasses."""
    model_config = ModelConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.nhl,
        state_size=args.state_size
    )
    
    training_config = TrainingConfig(
        epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
        log_freq=args.log_freq,
        eval_freq=args.eval_freq
    )
    
    data_config = DataConfig(
        dataset_path=args.dataset_path,
        tokenizer_path=args.tokenizer_path,
        max_length=args.max_length,
        num_data=args.num_data,
        masking_prob=args.masking_prob,
        random_crop=args.random_crop
    )
    
    system_config = SystemConfig(
        device_index=args.device_index,
        model_save_path=args.model_save_path,
        model_load_path=args.model_load_path,
        wandb_run_id=args.wandb_run_id,
        wandb_run_name=args.wandb_run_name,
        patience=args.patience
    )
    
    return model_config, training_config, data_config, system_config