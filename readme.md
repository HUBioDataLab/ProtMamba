# Protein Language Model Training

An implementation of a Mamba-based protein language model using Hugging Face Transformers.

## Project Structure

```
protmamba/
├── config/
│   ├── __init__.py
│   └── training_config.py
├── data/
│   ├── __init__.py
│   └── dataset.py
├── models/
│   ├── __init__.py
│   └── mamba_model.py
├── training/
│   ├── __init__.py
│   ├── callbacks.py
│   ├── metrics.py
│   └── trainer.py
├── utils/
│   ├── __init__.py
│   └── seed.py
├── main.py
├── dependencies.yml
└── README.md
```
## Installation

1. Clone the repository:
```bash
git clone <protmamba>
cd protmamba
```

2. Install dependencies:
```bash
conda env create -f environment.yml
```

## Usage

### Basic Training

Run training with default parameters:
```bash
python main.py
```

### Custom Configuration

Specify custom parameters:
```bash
python main.py \
    --batch_size 512 \
    --lr 5e-5 \
    --epochs 100 \
    --hidden_size 768 \
    --nhl 32 \
    --dataset_path /path/to/your/data.fasta \
    --model_save_path /path/to/save/models \
    --wandb_run_name my_experiment
```

### Resume Training

Resume from a checkpoint:
```bash
python main.py \
    --model_load_path /path/to/checkpoint \
    --wandb_run_id previous_run_id
```

## Key Parameters

### Model Architecture
- `--hidden_size`: Hidden dimension of the model (default: 512)
- `--nhl`: Number of hidden layers (default: 24)
- `--state_size`: State size for Mamba architecture (default: 16)

### Training
- `--epochs`: Number of training epochs (default: 10000)
- `--lr`: Learning rate (default: 1e-4)
- `--batch_size`: Batch size per device (default: 32768)
- `--weight_decay`: Weight decay for regularization (default: 0.001)
- `--warmup_steps`: Number of warmup steps (default: 20000)

### Data
- `--dataset_path`: Path to FASTA file with protein sequences
- `--tokenizer_path`: Path to pretrained tokenizer
- `--max_length`: Maximum sequence length (default: 512)
- `--num_data`: Number of sequences to use (default: 60M)

### System
- `--device_index`: CUDA device index (default: 0)
- `--model_save_path`: Directory to save models
- `--wandb_run_name`: Name for WandB run
- `--wandb_run_id`: ID for resuming WandB run

## Monitoring

The training process includes several monitoring features:

1. **WandB Integration**: Automatic logging of all metrics to Weights & Biases
2. **GPU Monitoring**: Real-time GPU memory usage tracking
3. **Training Metrics**: Loss, perplexity, accuracy, precision, recall, F1 scores
4. **Early Stopping**: Automatic stopping when validation loss stops improving

## Callbacks

The project includes several custom callbacks:

- `GPUMonitorCallback`: Tracks GPU memory usage
- `LogTrainingMetricsCallback`: Logs metrics including perplexity
- `BatchSizeCallback`: Monitors actual batch sizes
- `EarlyStoppingCallback`: Implements early stopping

## Data Processing

The `UniRefDataset` class handles:
- Streaming FASTA file reading
- Dynamic sequence tokenization
- Optional random cropping
- Automatic train/validation/test splitting

## Model Details

The project uses the Mamba architecture for efficient sequence modeling:
- Causal language modeling objective
- Support for long protein sequences
- Efficient state-space modeling
- Compatible with Hugging Face ecosystem