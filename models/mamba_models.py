"""
Model module for Mamba architecture setup and initialization.
"""
from transformers import MambaForCausalLM, MambaConfig, AutoTokenizer
from typing import Optional


def create_mamba_config(
    vocab_size: int,
    hidden_size: int = 512,
    num_hidden_layers: int = 24,
    state_size: int = 16,
    eos_token_id: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    bos_token_id: Optional[int] = None,
    use_cache: bool = False
) -> MambaConfig:
    """
    Create Mamba model configuration.
    
    Args:
        vocab_size: Size of the vocabulary
        hidden_size: Hidden dimension size
        num_hidden_layers: Number of transformer layers
        state_size: State size for Mamba
        eos_token_id: End of sequence token ID
        pad_token_id: Padding token ID
        bos_token_id: Beginning of sequence token ID
        use_cache: Whether to use KV cache
        
    Returns:
        MambaConfig object
    """
    return MambaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        state_size=state_size,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        bos_token_id=bos_token_id,
        use_cache=use_cache
    )


def initialize_model(
    model_config: MambaConfig,
    model_load_path: Optional[str] = None
) -> MambaForCausalLM:
    """
    Initialize Mamba model from config or checkpoint.
    
    Args:
        model_config: Model configuration
        model_load_path: Optional path to load pretrained model
        
    Returns:
        Initialized MambaForCausalLM model
    """
    if model_load_path is not None:
        print(f"Loading pretrained model from: {model_load_path}")
        model = MambaForCausalLM.from_pretrained(model_load_path)
    else:
        print("Initializing new model from config")
        model = MambaForCausalLM(model_config)
    
    return model


def load_tokenizer(tokenizer_path: str, padding_side: str = 'left'):
    """
    Load tokenizer from path.
    
    Args:
        tokenizer_path: Path to tokenizer files
        padding_side: Side to pad sequences ('left' or 'right')
        
    Returns:
        Loaded tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        padding_side=padding_side
    )
    print(f"Loaded tokenizer from: {tokenizer_path}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    return tokenizer


def setup_model_and_tokenizer(
    model_config_dict: dict,
    tokenizer_path: str,
    model_load_path: Optional[str] = None
):
    """
    Complete setup for model and tokenizer.
    
    Args:
        model_config_dict: Dictionary of model configuration parameters
        tokenizer_path: Path to tokenizer
        model_load_path: Optional path to pretrained model
        
    Returns:
        Tuple of (model, tokenizer, config)
    """
    # Load tokenizer first
    tokenizer = load_tokenizer(tokenizer_path)
    
    # Update config with tokenizer info
    model_config_dict.update({
        'vocab_size': tokenizer.vocab_size,
        'eos_token_id': tokenizer.eos_token_id,
        'pad_token_id': tokenizer.pad_token_id,
        'bos_token_id': tokenizer.bos_token_id
    })
    
    # Create config and initialize model
    config = create_mamba_config(**model_config_dict)
    model = initialize_model(config, model_load_path)
    
    return model, tokenizer, config