"""
Simple comparison of baseline vs post-training metrics-all.jsonl files
"""

import json
from pathlib import Path

def load_metrics(jsonl_file):
    """Load metrics from a JSONL file"""
    metrics = {}
    with open(jsonl_file, 'r') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                task_name = data['task_name']
                score = data['metrics']['primary_score']
                metrics[task_name] = score
    return metrics

def main():
    """Compare the two metrics files"""
    baseline_file = "results/baseline/metrics-all.jsonl"
    post_training_file = "results/post_training/metrics-all.jsonl"
    
    print("🔬 TRAINING COMPARISON")
    print("=" * 40)
    
    # Load both files
    baseline = load_metrics(baseline_file)
    post_training = load_metrics(post_training_file)
    
    # Compare only the main OLMES tasks (::olmes format)
    olmes_tasks = [task for task in baseline.keys() if "::olmes" in task]
    
    print(f"{'Task':<20} {'Before':<10} {'After':<10} {'Change':<10} {'Result'}")
    print("-" * 60)
    
    for task in sorted(olmes_tasks):
        if task in post_training:
            before = baseline[task]
            after = post_training[task]
            change = after - before
            
            # Clean up task name
            clean_name = task.replace("::olmes", "").replace("_", " ").title()
            
            # Simple status
            if change > 0.01:
                status = "Better"
            elif change < -0.01:
                status = "Worse"
            else:
                status = "Same"
            
            print(f"{clean_name:<20} {before*100:.1f}%     {after*100:.1f}%     {change:+.3f}     {status}")
    
    print("-" * 60)
    
    # Simple summary
    changes = [post_training[task] - baseline[task] for task in olmes_tasks if task in post_training]
    avg_change = sum(changes) / len(changes) if changes else 0
    
    print(f"\nAverage change: {avg_change:+.3f} points ({avg_change*100:+.1f}%)")
    
    if avg_change > 0.01:
        print("Training helped!")
    elif avg_change < -0.01:
        print("Training hurt performance")
    else:
        print("Training had minimal impact")

if __name__ == "__main__":
    main() 