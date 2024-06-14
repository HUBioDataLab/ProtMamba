Repository for ProtMamba model

_main.py_ contains the main script to train and test the model. Can be run with the arguments below:

    --epochs EPOCHS
  
    --batch_size BATCH_SIZE
  
    --lr LR
  
    --max_length MAX_LENGTH
  
    --patience PATIENCE
  
    --masking_prob MASKING_PROB
  
    --nhl NHL             number of hidden layers
  
    --weight_decay WEIGHT_DECAY
  
    --state_size STATE_SIZE
  
    --hidden_size HIDDEN_SIZE
  
    --random_crop RANDOM_CROP
  
    --device_index DEVICE_INDEX
                        index of the cuda device
                        
    --dataset_path DATASET_PATH
  
    --tokenizer_link TOKENIZER_LINK
                        link to huggingface tokenizer
                        
    --model_path MODEL_PATH
                        provide the path to the model state dict if not training from zero
                        
    --wandb_run_id WANDB_RUN_ID
                        provide wandb run id if continuing a previous run
                        
    --wandb_run_name WANDB_RUN_NAME
                        provide wandb run name for new runs
                        
    --log_freq LOG_FREQ

_utils.py_ contains the dataset and early stopping classes, and train/test functions

