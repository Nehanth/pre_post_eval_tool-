"""
Script to run post-training evaluation using OLMES on the trained model
"""

import yaml
import os
import subprocess
import sys
import threading
import time
import json
from pathlib import Path
from tqdm import tqdm

def load_config(config_path="config.yaml"):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def check_gpu_available():
    """Check if NVIDIA GPU is available"""
    try:
        result = subprocess.run("nvidia-smi", shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            # Count GPUs by looking for GPU device lines
            lines = result.stdout.split('\n')
            gpu_count = 0
            
            for line in lines:
                if '|' in line and 'NVIDIA' in line and ('L4' in line or 'RTX' in line or 'GTX' in line or 'Tesla' in line or 'A100' in line or 'V100' in line):
                    gpu_count += 1
            
            # If the above doesn't work, try a simpler approach
            if gpu_count == 0:
                for line in lines:
                    if line.strip().startswith('|') and 'NVIDIA' in line:
                        gpu_count += 1
            
            return gpu_count
        return 0
    except:
        return 0

def find_trained_model_path(config):
    """Dynamically find the path to the trained model"""
    # Check for manual override first
    post_training_config = config['evaluation'].get('post_training', {})
    override_path = post_training_config.get('model_path_override')
    
    if override_path:
        override_path = Path(override_path)
        if override_path.exists():
            print(f"[SUCCESS] Using override path: {override_path}")
            return str(override_path.absolute())
        else:
            print(f"[WARNING] Override path doesn't exist: {override_path}")
    
    # Get base directories from config
    checkpoints_dir = Path(config['training']['checkpoints_dir'])
    output_dir = Path(config['training']['output_dir'])
    
    print(f"[INFO] Searching for trained model...")
    print(f"Checkpoints dir: {checkpoints_dir}")
    print(f"Output dir: {output_dir}")
    
    # Get preferences from config
    model_selection = post_training_config.get('model_selection', 'auto')
    preferred_format = post_training_config.get('preferred_format', 'hf_format')
    
    print(f"[INFO] Model selection strategy: {model_selection}")
    print(f"[INFO] Preferred format: {preferred_format}")
    
    if preferred_format == "hf_format":
        search_locations = [
            
            checkpoints_dir / "hf_format",
            Path("./checkpoints/hf_format"),
            
            checkpoints_dir / "full_state", 
            Path("./checkpoints/full_state"),
            
            output_dir,
            Path("./models/trained"),
            Path("./checkpoints")
        ]
    else:  # full_state preferred
        search_locations = [
            checkpoints_dir / "full_state",
            Path("./checkpoints/full_state"),
            checkpoints_dir / "hf_format",
            Path("./checkpoints/hf_format"),
            output_dir,
            Path("./models/trained"),
            Path("./checkpoints")
        ]
    
    best_model = None
    best_samples = 0
    
    for base_dir in search_locations:
        if not base_dir.exists():
            continue
            
        print(f"[INFO] Searching in: {base_dir}")
        
        if base_dir.name == "hf_format":
            # Find the checkpoint with the highest sample count
            sample_dirs = list(base_dir.glob("samples_*"))
            for sample_dir in sample_dirs:
                if sample_dir.is_dir():
                    # Extract sample number from directory name
                    try:
                        sample_num = int(sample_dir.name.split("_")[1])
                        # Check if it contains model files
                        if (any(sample_dir.glob("*.safetensors*")) or 
                            any(sample_dir.glob("pytorch_model*.bin")) or
                            any(sample_dir.glob("model.safetensors*"))):
                            
                            if sample_num > best_samples:
                                best_samples = sample_num
                                best_model = sample_dir
                                print(f"[INFO] Found candidate: {sample_dir} (samples: {sample_num})")
                    except (ValueError, IndexError):
                        continue
        
        elif base_dir.name == "full_state":
            # Find the latest epoch
            epoch_dirs = list(base_dir.glob("epoch_*"))
            latest_epoch = -1
            for epoch_dir in epoch_dirs:
                if epoch_dir.is_dir():
                    try:
                        epoch_num = int(epoch_dir.name.split("_")[1])
                        if (epoch_num > latest_epoch and 
                            any(epoch_dir.glob("pytorch_model*.bin"))):
                            latest_epoch = epoch_num
                            if not best_model:  # Only use if no HF format found
                                best_model = epoch_dir
                                print(f"[INFO] Found candidate: {epoch_dir} (epoch: {epoch_num})")
                    except (ValueError, IndexError):
                        continue
        
        else:
            # Check if directory directly contains model files
            if (any(base_dir.glob("*.safetensors*")) or 
                any(base_dir.glob("pytorch_model*.bin")) or
                any(base_dir.glob("model.safetensors*"))):
                if not best_model:  # Only use as fallback
                    best_model = base_dir
                    print(f"[INFO] Found candidate: {base_dir}")
    
    if best_model:
        print(f"[SUCCESS] Selected trained model: {best_model}")
        if best_samples > 0:
            print(f"[INFO] Model trained on {best_samples} samples")
        return str(best_model.absolute())
    
    return None

def setup_directories(config):
    """Create necessary directories for post-training results"""
    post_training_dir = Path(config['evaluation']['post_training_results_dir'])
    post_training_dir.mkdir(parents=True, exist_ok=True)
    
    logs_dir = Path(config['logging']['log_file']).parent
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    return post_training_dir

def run_command_with_progress(cmd, description):
    """Run command with a progress bar"""
    print(f"[RUNNING] {description}")
    print(f"Command: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    
    # Create a progress bar
    pbar = tqdm(desc="Evaluating", unit="step", dynamic_ncols=True)
    
    # Flag to control progress bar
    process_running = True
    
    def update_progress():
        """Update progress bar while process runs"""
        step = 0
        while process_running:
            pbar.update(1)
            step += 1
            time.sleep(2)  # Update every 2 seconds
    
    # Start progress thread
    progress_thread = threading.Thread(target=update_progress)
    progress_thread.daemon = True
    progress_thread.start()
    
    try:
        # Run the actual command
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        process_running = False
        pbar.close()
        
        print("[SUCCESS] Post-training evaluation completed!")
        print(f"Output preview: {result.stdout[:200]}..." if len(result.stdout) > 200 else result.stdout)
        return True, result.stdout
        
    except subprocess.CalledProcessError as e:
        process_running = False
        pbar.close()
        print(f"[ERROR] Command failed: {e}")
        print(f"Error output: {e.stderr}")
        return False, e.stderr
    except FileNotFoundError:
        process_running = False
        pbar.close()
        print("[ERROR] 'olmes' command not found. Please run setup_environment.py first.")
        return False, "Command not found"

def run_olmes_evaluation(config, model_path, output_dir):
    """Run OLMES evaluation on the trained model"""
    tasks = config['evaluation']['tasks']
    
    print(f"Trained Model: {model_path}")
    print(f"Tasks: {', '.join(tasks)}")
    print(f"Output directory: {output_dir}")
    
    # Check for evaluation limit
    limit = config['evaluation'].get('limit')
    if limit:
        print(f"Limit: {limit} examples per task")
    else:
        print("Limit: Full evaluation (no limit)")
    
    # Check for GPU availability
    gpu_count = check_gpu_available()
    max_gpus = config['evaluation']['gpu'].get('max_gpus', 8)
    
    if gpu_count > 0:
        gpus_to_use = min(gpu_count, max_gpus)
        print(f"[SUCCESS] Detected {gpu_count} NVIDIA GPU(s) - using {gpus_to_use} GPUs for parallel evaluation")
        use_gpu = True
    else:
        print("[INFO] No GPU detected - using CPU")
        use_gpu = False
        gpus_to_use = 0
    
    # Build OLMES command for trained model
    cmd = [
        "olmes",
        "--model", model_path,
        "--task"
    ] + tasks + [
        "--output-dir", str(output_dir)
    ]
    
    # Add limit if specified
    if limit:
        cmd.extend(["--limit", str(limit)])
    
    # Add GPU configuration for parallel evaluation
    if use_gpu:
        cmd.extend([
            "--gpus", str(gpus_to_use)
        ])
    
    success, output = run_command_with_progress(cmd, f"OLMES post-training evaluation on {len(tasks)} tasks")
    return success

def compare_results(config):
    """Compare baseline and post-training results if both exist"""
    baseline_dir = Path(config['evaluation']['baseline_results_dir'])
    post_training_dir = Path(config['evaluation']['post_training_results_dir'])
    
    baseline_metrics = baseline_dir / "metrics.json"
    post_training_metrics = post_training_dir / "metrics.json"
    
    if baseline_metrics.exists() and post_training_metrics.exists():
        print("\n" + "="*60)
        print("COMPARISON: BASELINE vs POST-TRAINING")
        print("="*60)
        
        try:
            with open(baseline_metrics, 'r') as f:
                baseline_data = json.load(f)
            with open(post_training_metrics, 'r') as f:
                post_training_data = json.load(f)
            
            print(f"{'Task':<15} {'Baseline':<12} {'Post-Train':<12} {'Improvement':<12}")
            print("-" * 60)
            
            for task in config['evaluation']['tasks']:
                if task in baseline_data and task in post_training_data:
                    baseline_score = baseline_data[task].get('accuracy', 0) * 100
                    post_train_score = post_training_data[task].get('accuracy', 0) * 100
                    improvement = post_train_score - baseline_score
                    
                    print(f"{task:<15} {baseline_score:<12.2f} {post_train_score:<12.2f} {improvement:+.2f}")
            
            print("-" * 60)
            
        except Exception as e:
            print(f"[WARNING] Could not compare results: {e}")
    else:
        if not baseline_metrics.exists():
            print(f"[INFO] Baseline results not found at {baseline_metrics}")
        print("[INFO] Run comparison manually after both evaluations complete")

def main():
    """Main execution function"""
    print("Starting post-training evaluation...")
    
    # Load configuration
    try:
        config = load_config()
        print(f"[SUCCESS] Loaded configuration from config.yaml")
    except FileNotFoundError:
        print("[ERROR] config.yaml not found. Please create it first.")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"[ERROR] Error parsing config.yaml: {e}")
        sys.exit(1)
    
    # Find trained model
    model_path = find_trained_model_path(config)
    if not model_path:
        print("[ERROR] Could not find trained model. Please ensure training completed successfully.")
        print("Expected locations:")
        print("  - ./checkpoints/hf_format/samples_*")
        print("  - ./checkpoints/full_state/epoch_*")
        print("  - ./models/trained")
        sys.exit(1)
    
    # Setup directories
    output_dir = setup_directories(config)
    print(f"[SUCCESS] Created output directory: {output_dir}")
    
    # Run post-training evaluation
    success = run_olmes_evaluation(config, model_path, output_dir)
    
    if success:
        print("\n[SUCCESS] Post-training evaluation completed!")
        print(f"Results saved to: {output_dir}")
        
        # List result files
        result_files = list(output_dir.glob("*"))
        if result_files:
            print(f"\nResult files created:")
            for f in result_files:
                print(f"  - {f.name}")
        
        # Try to compare with baseline results
        compare_results(config)
        
        print("\n" + "="*60)
        print("🎉 PRE/POST TRAINING EVALUATION PIPELINE COMPLETE! 🎉")
        print("="*60)
        print("Next steps:")
        print("1. Review the comparison results above")
        print("2. Analyze individual task improvements")
        print("3. Consider additional training if needed")
        
    else:
        print("[ERROR] Post-training evaluation failed. Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main() 