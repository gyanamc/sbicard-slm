import os
import yaml
import torch
import inspect
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
try:
    from trl import SFTConfig
    HAS_SFT_CONFIG = True
except ImportError:
    HAS_SFT_CONFIG = False
from utils.dataset_loader import load_and_prepare_dataset

def train():
    # Load configuration
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    model_name = config['model_name']
    new_model = config['new_model_name']

    # Check for GPU
    use_gpu = torch.cuda.is_available()
    device_map = "auto" if use_gpu else "cpu"
    
    if not use_gpu:
        print("WARNING: CUDA not available. Running in CPU mode. Quantization disabled.")
        config['load_in_4bit'] = False
        config['fp16'] = False
        config['bf16'] = False
        # Use test dataset for local trial
        config['dataset_path'] = "data_processing/processed_data/test_augmented.jsonl"
        config['num_train_epochs'] = 1
        config['save_steps'] = 5
        config['eval_steps'] = 5
        config['logging_steps'] = 1

    # QLoRA configuration (only if GPU)
    bnb_config = None
    if config['load_in_4bit'] and use_gpu:
        compute_dtype = getattr(torch, "float16")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=config['load_in_4bit'],
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=False,
        )

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load dataset
    print(f"Loading dataset from {config['dataset_path']}...")
    if not os.path.exists(config['dataset_path']):
        print(f"Dataset {config['dataset_path']} not found. Switching to test data.")
        config['dataset_path'] = "data_processing/processed_data/test_augmented.jsonl"
        
    with open("config_local.yaml", "w") as f:
        yaml.dump(config, f)
        
    dataset = load_and_prepare_dataset("config_local.yaml")
    train_dataset = dataset['train']
    eval_dataset = dataset['test']
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(eval_dataset)}")

    # LoRA config
    peft_config = LoraConfig(
        lora_alpha=config['lora_alpha'],
        lora_dropout=config['lora_dropout'],
        r=config['lora_r'],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=config['target_modules']
    )

    # Training arguments
    if HAS_SFT_CONFIG:
        # Determine the correct key for max sequence length
        sft_params = inspect.signature(SFTConfig.__init__).parameters
        max_seq_len_key = 'max_length' if 'max_length' in sft_params else 'max_seq_length'
        
        # Build dictionary of kwargs for SFTConfig
        sft_kwargs = {
            'output_dir': config['output_dir'],
            'num_train_epochs': config['num_train_epochs'],
            'per_device_train_batch_size': config['per_device_train_batch_size'] if use_gpu else 1,
            'gradient_accumulation_steps': config['gradient_accumulation_steps'],
            'optim': "paged_adamw_32bit" if use_gpu else "adamw_torch",
            'save_steps': config['save_steps'],
            'logging_steps': config['logging_steps'],
            'learning_rate': config['learning_rate'],
            'weight_decay': config['weight_decay'],
            'fp16': config['fp16'],
            'bf16': config['bf16'],
            'max_grad_norm': config['max_grad_norm'],
            'max_steps': 5 if not use_gpu else -1,
            'warmup_ratio': config['warmup_ratio'],
            'group_by_length': config['group_by_length'],
            'lr_scheduler_type': config['lr_scheduler_type'],
            'report_to': "none",
            'use_cpu': not use_gpu,
            'packing': False,
            'dataset_text_field': "text"
        }
        sft_kwargs[max_seq_len_key] = config['max_seq_length']
        
        training_args = SFTConfig(**sft_kwargs)
    else:
        training_args = TrainingArguments(
            output_dir=config['output_dir'],
            num_train_epochs=config['num_train_epochs'],
            per_device_train_batch_size=config['per_device_train_batch_size'] if use_gpu else 1,
            gradient_accumulation_steps=config['gradient_accumulation_steps'],
            optim="paged_adamw_32bit" if use_gpu else "adamw_torch",
            save_steps=config['save_steps'],
            logging_steps=config['logging_steps'],
            learning_rate=config['learning_rate'],
            weight_decay=config['weight_decay'],
            fp16=config['fp16'],
            bf16=config['bf16'],
            max_grad_norm=config['max_grad_norm'],
            max_steps=5 if not use_gpu else -1, # Limit steps for local trial
            warmup_ratio=config['warmup_ratio'],
            group_by_length=config['group_by_length'],
            lr_scheduler_type=config['lr_scheduler_type'],
            report_to="none", # Disable tensorboard for local trial
            use_cpu=not use_gpu,
        )

    # Trainer parameters setup
    trainer_kwargs = {
        'model': model,
        'train_dataset': train_dataset,
        'eval_dataset': eval_dataset,
        'peft_config': peft_config,
        'args': training_args,
    }
    
    # Handle the tokenizer vs processing_class change in TRL
    trainer_params = inspect.signature(SFTTrainer.__init__).parameters
    if 'processing_class' in trainer_params:
        trainer_kwargs['processing_class'] = tokenizer
    else:
        trainer_kwargs['tokenizer'] = tokenizer
        
    # Add older params if we are not using SFTConfig
    if not HAS_SFT_CONFIG:
        trainer_kwargs['max_seq_length'] = config['max_seq_length']
        trainer_kwargs['packing'] = False

    trainer = SFTTrainer(**trainer_kwargs)

    # Train model
    trainer.train()

    # Save trained model
    trainer.model.save_pretrained(new_model)
    tokenizer.save_pretrained(new_model)
    
    print(f"Model saved to {new_model}")

if __name__ == "__main__":
    train()
