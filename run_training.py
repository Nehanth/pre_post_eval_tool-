"""
Training script using InstructLab training library
"""

import yaml
import os
import sys
from pathlib import Path
from datasets import load_dataset
import json

def load_config(config_path="config.yaml"):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_directories(config):
    """Create necessary directories"""
    paths = [
        config['training']['output_dir'],
        config['training']['checkpoints_dir'], 
        config['training']['data_output_dir'],
        Path(config['logging']['log_file']).parent
    ]
    
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)
        print(f"[SUCCESS] Created directory: {path}")

def prepare_dataset(config):
    """Download and prepare the training dataset"""
    dataset_config = config['training']['dataset']
    
    print(f"[RUNNING] Loading dataset: {dataset_config['name']}")
    
    try:
        # Load dataset from HuggingFace
        dataset = load_dataset(dataset_config['name'], split=dataset_config['split'])
        
        # Take subset for testing
        if dataset_config['subset_size']:
            dataset = dataset.select(range(min(dataset_config['subset_size'], len(dataset))))
            print(f"[SUCCESS] Using {len(dataset)} examples from dataset")
        
        # Convert to JSONL format for InstructLab
        output_path = Path(config['training']['data_output_dir']) / "train_data.jsonl"
        
        with open(output_path, 'w') as f:
            for example in dataset:
                # Convert to messages format expected by InstructLab
                if 'messages' in example:
                    # Already in correct format
                    json.dump(example, f)
                    f.write('\n')
                elif 'conversations' in example:
                    # Convert conversations to messages format
                    messages = []
                    for turn in example['conversations']:
                        messages.append({
                            "role": turn.get('from', 'user'),
                            "content": turn.get('value', '')
                        })
                    json.dump({"messages": messages}, f)
                    f.write('\n')
                else:
                    # Try to construct from text fields
                    if 'instruction' in example and 'output' in example:
                        messages = [
                            {"role": "user", "content": example['instruction']},
                            {"role": "assistant", "content": example['output']}
                        ]
                        json.dump({"messages": messages}, f)
                        f.write('\n')
        
        print(f"[SUCCESS] Dataset prepared: {output_path}")
        return str(output_path)
        
    except Exception as e:
        print(f"[ERROR] Failed to prepare dataset: {e}")
        return None

def run_instructlab_training(config, data_path):
    """Run training using InstructLab training library"""
    try:
        from instructlab.training import (
            run_training,
            TorchrunArgs,
            TrainingArgs,
            DataProcessArgs,
            data_process as dp
        )
        
        print("[SUCCESS] InstructLab training library loaded")
        
        # Setup training arguments
        training_params = config['training']['parameters']
        distributed = config['training']['distributed']
        
        training_args = TrainingArgs(
            model_path=config['training']['model_path'],
            data_path=data_path,
            ckpt_output_dir=config['training']['checkpoints_dir'],
            data_output_dir=config['training']['data_output_dir'],
            max_seq_len=training_params['max_seq_len'],
            max_batch_len=training_params['max_batch_len'],
            num_epochs=training_params['num_epochs'],
            effective_batch_size=training_params['effective_batch_size'],
            save_samples=training_params['save_samples'],
            learning_rate=training_params['learning_rate'],
            warmup_steps=training_params['warmup_steps'],
            random_seed=training_params['random_seed'],
            disable_flash_attn=True,  
            process_data=True,  
            distributed_backend="fsdp"  
        )
        
        # Multi-GPU configuration using all 8 GPUs
        torchrun_args = TorchrunArgs(
            nnodes=distributed['nnodes'],  
            nproc_per_node=distributed['nproc_per_node'],  
            node_rank=distributed['node_rank'],  
            rdzv_id=distributed['rdzv_id'],
            rdzv_endpoint=distributed['rdzv_endpoint']
        )
        
        print(f"[RUNNING] Starting training with {distributed['nproc_per_node']} GPUs...")
        print(f"Model: {config['training']['model_path']}")
        print(f"Dataset: {data_path}")
        print(f"Epochs: {training_params['num_epochs']}")
        print(f"Learning rate: {training_params['learning_rate']}")
        print(f"Distributed setup: {distributed['nnodes']} nodes, {distributed['nproc_per_node']} GPUs per node")
        
        # Preprocess data separately (as shown in documentation)
        print("[RUNNING] Preprocessing data separately...")
        data_process_args = DataProcessArgs(
            data_output_path=training_args.data_output_dir,
            model_path=training_args.model_path,
            data_path=training_args.data_path,
            max_seq_len=training_args.max_seq_len,
            chat_tmpl_path=None  # Use model's built-in chat template
        )
        
        dp.main(data_process_args)
        print("[SUCCESS] Data preprocessing completed")
        
        # Run training
        run_training(
            torch_args=torchrun_args,
            train_args=training_args
        )
        
        print("[SUCCESS] Training completed!")
        return True
        
    except ImportError as e:
        print(f"[ERROR] InstructLab training library import failed: {e}")
        print("Please install with: pip install instructlab-training")
        return False
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main training function"""
    print("Starting model training with InstructLab...")
    
    # Load configuration
    try:
        config = load_config()
        print("[SUCCESS] Loaded configuration from config.yaml")
    except FileNotFoundError:
        print("[ERROR] config.yaml not found")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"[ERROR] Error parsing config.yaml: {e}")
        sys.exit(1)
    
    # Setup directories
    setup_directories(config)
    
    # Prepare dataset
    data_path = prepare_dataset(config)
    if not data_path:
        print("[ERROR] Failed to prepare dataset")
        sys.exit(1)
    
    # Run training
    success = run_instructlab_training(config, data_path)
    
    if success:
        print("\n[SUCCESS] Training pipeline completed!")
        print(f"Trained model saved to: {config['training']['output_dir']}")
        print("\nNext steps:")
        print("1. Run post-training evaluation")
        print("2. Compare with baseline results")
    else:
        print("[ERROR] Training failed")
        sys.exit(1)

if __name__ == "__main__":
    main() 