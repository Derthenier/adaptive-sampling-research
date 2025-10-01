"""
Research Experiment Tracking Framework
For tracking diffusion model experiments systematically

This provides structure for rigorous research and reproducibility
"""

import json
import time
from datetime import datetime
from pathlib import Path
import torch


class ExperimentTracker:
    """
    Track experiments with proper logging, metrics, and reproducibility
    
    Usage:
        tracker = ExperimentTracker("adaptive_sampling_v1")
        tracker.log_config(config_dict)
        
        for step in training:
            metrics = train_step()
            tracker.log_metrics(metrics, step)
        
        tracker.save_checkpoint(model, "best")
        tracker.finalize()
    """
    
    def __init__(self, experiment_name, base_dir="experiments"):
        self.experiment_name = experiment_name
        self.start_time = datetime.now()
        
        # Create experiment directory
        self.exp_dir = Path(base_dir) / f"{experiment_name}_{self.start_time.strftime('%Y%m%d_%H%M%S')}"
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Subdirectories
        self.checkpoints_dir = self.exp_dir / "checkpoints"
        self.logs_dir = self.exp_dir / "logs"
        self.samples_dir = self.exp_dir / "samples"
        self.metrics_dir = self.exp_dir / "metrics"
        
        for d in [self.checkpoints_dir, self.logs_dir, self.samples_dir, self.metrics_dir]:
            d.mkdir(exist_ok=True)
        
        # Tracking
        self.config = {}
        self.metrics_history = []
        self.notes = []
        
        print(f"📊 Experiment: {experiment_name}")
        print(f"📁 Directory: {self.exp_dir}")
    
    def log_config(self, config):
        """Log experiment configuration"""
        self.config = config
        self.config['experiment_name'] = self.experiment_name
        self.config['start_time'] = str(self.start_time)
        self.config['gpu'] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        
        # Save config
        with open(self.exp_dir / "config.json", "w") as f:
            json.dump(self.config, f, indent=2)
        
        print("✓ Config logged")
    
    def log_metrics(self, metrics, step):
        """Log metrics at a given step"""
        metrics['step'] = step
        metrics['timestamp'] = time.time()
        self.metrics_history.append(metrics)
        
        # Append to CSV for easy analysis
        import csv
        csv_path = self.metrics_dir / "metrics.csv"
        
        # Write header if new file
        write_header = not csv_path.exists()
        
        with open(csv_path, "a", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=metrics.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(metrics)
    
    def log_note(self, note):
        """Add a timestamped note"""
        timestamped_note = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {note}"
        self.notes.append(timestamped_note)
        
        # Append to notes file
        with open(self.exp_dir / "notes.txt", "a") as f:
            f.write(timestamped_note + "\n")
        
        print(f"📝 {note}")
    
    def save_checkpoint(self, model, name="latest"):
        """Save model checkpoint"""
        checkpoint_path = self.checkpoints_dir / f"{name}.pt"
        
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'config': self.config,
            'timestamp': datetime.now().isoformat(),
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"💾 Checkpoint saved: {name}")
    
    def save_samples(self, images, step, prefix="sample"):
        """Save generated image samples"""
        from PIL import Image
        
        step_dir = self.samples_dir / f"step_{step:06d}"
        step_dir.mkdir(exist_ok=True)
        
        for i, img in enumerate(images):
            if isinstance(img, torch.Tensor):
                # Convert tensor to PIL
                img = img.cpu().permute(1, 2, 0).numpy()
                img = (img * 255).astype('uint8')
                img = Image.fromarray(img)
            
            img.save(step_dir / f"{prefix}_{i:03d}.png")
        
        print(f"🖼️  Saved {len(images)} samples at step {step}")
    
    def finalize(self):
        """Finalize experiment and generate summary"""
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        summary = {
            'experiment_name': self.experiment_name,
            'start_time': str(self.start_time),
            'end_time': str(end_time),
            'duration_seconds': duration.total_seconds(),
            'total_steps': len(self.metrics_history),
            'config': self.config,
            'notes': self.notes,
        }
        
        # Save summary
        with open(self.exp_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        print("\n" + "="*60)
        print(f"✅ Experiment Complete: {self.experiment_name}")
        print(f"⏱️  Duration: {duration}")
        print(f"📁 Results: {self.exp_dir}")
        print("="*60)
        
        return summary


class ResearchProtocol:
    """
    Template for rigorous research experiments
    
    Ensures you follow scientific method:
    1. Hypothesis
    2. Baseline
    3. Treatment
    4. Evaluation
    5. Analysis
    """
    
    @staticmethod
    def create_protocol(research_question):
        """Generate experiment protocol"""
        
        protocol = {
            "research_question": research_question,
            "hypothesis": "",
            "baseline": {
                "model": "",
                "metrics": [],
                "expected_performance": {},
            },
            "treatment": {
                "modifications": [],
                "expected_improvement": "",
                "risks": [],
            },
            "evaluation": {
                "metrics": [],
                "datasets": [],
                "statistical_tests": [],
            },
            "success_criteria": {
                "minimum_improvement": "",
                "secondary_goals": [],
            },
            "timeline": {
                "implementation": "",
                "training": "",
                "evaluation": "",
                "total": "",
            },
            "resources": {
                "compute": "",
                "data": "",
                "prior_work": [],
            }
        }
        
        return protocol


# Example usage
if __name__ == "__main__":
    # Create experiment
    tracker = ExperimentTracker("test_experiment")
    
    # Log configuration
    config = {
        "model": "stable-diffusion-1.5",
        "learning_rate": 1e-5,
        "batch_size": 4,
        "num_steps": 1000,
    }
    tracker.log_config(config)
    
    # Simulate training
    tracker.log_note("Started training with adaptive learning rate")
    
    for step in range(10):
        metrics = {
            "loss": 0.5 - step * 0.01,  # Fake decreasing loss
            "fid_score": 25 + step * 0.5,  # Fake FID
        }
        tracker.log_metrics(metrics, step)
    
    tracker.log_note("Training completed successfully")
    
    # Finalize
    tracker.finalize()
    
    print("\n📋 Protocol Template:")
    protocol = ResearchProtocol.create_protocol(
        "Can adaptive sampling reduce inference time by 30% with <5% quality loss?"
    )
    print(json.dumps(protocol, indent=2))
