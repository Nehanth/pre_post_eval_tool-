"""
Simple script to run baseline evaluation using OLMES
"""

import yaml
import os
import subprocess
import sys
import threading
import time
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

def setup_directories(config):
    """Create necessary directories"""
    baseline_dir = Path(config['evaluation']['baseline_results_dir'])
    baseline_dir.mkdir(parents=True, exist_ok=True)
    
    logs_dir = Path(config['logging']['log_file']).parent
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    return baseline_dir

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
        
        print("[SUCCESS] Evaluation completed!")
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

def run_olmes_evaluation(config, output_dir):
    """Run OLMES evaluation on the baseline model"""
    model_name = config['model']['name']
    tasks = config['evaluation']['tasks']
    
    print(f"Model: {model_name}")
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
    max_gpus = config['evaluation']['gpu'].get('max_gpus', 8)  # Use all available GPUs
    
    if gpu_count > 0:
        gpus_to_use = min(gpu_count, max_gpus)
        print(f"[SUCCESS] Detected {gpu_count} NVIDIA GPU(s) - using {gpus_to_use} GPUs for parallel evaluation")
        use_gpu = True
    else:
        print("[INFO] No GPU detected - using CPU")
        use_gpu = False
        gpus_to_use = 0
    
    # Build OLMES command
    cmd = [
        "olmes",
        "--model", model_name,
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
    
    success, output = run_command_with_progress(cmd, f"OLMES evaluation on {len(tasks)} tasks")
    return success

def main():
    """Main execution function"""
    print("Starting baseline evaluation...")
    
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
    
    # Setup directories
    output_dir = setup_directories(config)
    print(f"[SUCCESS] Created output directory: {output_dir}")
    
    # Run evaluation
    success = run_olmes_evaluation(config, output_dir)
    
    if success:
        print("\n[SUCCESS] Baseline evaluation completed!")
        print(f"Results saved to: {output_dir}")
        print("\nNext steps:")
        print("1. Check the results in the output directory")
        print("2. If satisfied, we can move on to training setup")
        
        # List result files
        result_files = list(output_dir.glob("*"))
        if result_files:
            print(f"\nResult files created:")
            for f in result_files:
                print(f"  - {f.name}")
    else:
        print("[ERROR] Evaluation failed. Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main() 