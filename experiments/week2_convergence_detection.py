#!/usr/bin/env python3
"""
Week 2: Dynamic Convergence Detection

Instead of predicting steps from text, we DETECT convergence during generation.

Key idea: Monitor how much the latent changes each step. When change is small,
the image has converged - stop early!
"""

import torch
from diffusers import StableDiffusionPipeline
import time
from pathlib import Path
import json

class ConvergenceDetector:
    """
    Detects when diffusion process has converged by monitoring latent changes
    """
    
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5"):
        print("🔬 Initializing Convergence Detector...")
        
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
        self.pipe = self.pipe.to("cuda")
        self.pipe.enable_attention_slicing()
        
        print("✓ Ready!")
    
    def measure_latent_change(self, latent_current, latent_previous):
        """
        Measure how much the latent changed between steps
        
        Returns:
            - mean_change: Average pixel change
            - max_change: Maximum pixel change
            - relative_change: Change relative to magnitude
        """
        if latent_previous is None:
            return None
        
        # Calculate absolute difference
        diff = torch.abs(latent_current - latent_previous)
        
        mean_change = diff.mean().item()
        max_change = diff.max().item()
        
        # Relative change (normalized by current magnitude)
        magnitude = torch.abs(latent_current).mean().item()
        relative_change = mean_change / (magnitude + 1e-8)
        
        return {
            'mean_change': mean_change,
            'max_change': max_change,
            'relative_change': relative_change,
        }
    
    def has_converged(self, change_history, window=3, threshold=0.01):
        """
        Determine if generation has converged
        
        Args:
            change_history: List of recent change measurements
            window: Number of recent steps to consider
            threshold: Maximum allowed change to consider converged
        
        Returns:
            bool: True if converged
        """
        if len(change_history) < window:
            return False
        
        # Look at recent changes
        recent_changes = change_history[-window:]
        
        # Check if all recent changes are below threshold
        avg_recent_change = sum(recent_changes) / len(recent_changes)
        
        return avg_recent_change < threshold
    
    def generate_with_early_stopping(
        self, 
        prompt, 
        max_steps=50,
        min_steps=10,
        convergence_threshold=0.01,
        seed=42
    ):
        """
        Generate image with dynamic early stopping
        
        Args:
            prompt: Text prompt
            max_steps: Maximum steps to run
            min_steps: Minimum steps before checking convergence
            convergence_threshold: Threshold for convergence detection
            seed: Random seed for reproducibility
        
        Returns:
            image, actual_steps, convergence_data
        """
        generator = torch.Generator("cuda").manual_seed(seed)
        
        # Track convergence
        previous_latent = None
        change_history = []
        convergence_data = []
        stopped_early = False
        actual_steps = max_steps
        
        def callback(step, timestep, latents):
            nonlocal previous_latent, stopped_early, actual_steps
            
            # Measure change from previous step
            change = self.measure_latent_change(latents, previous_latent)
            
            if change:
                change_history.append(change['relative_change'])
                convergence_data.append({
                    'step': step,
                    'mean_change': change['mean_change'],
                    'relative_change': change['relative_change'],
                })
                
                # Check for convergence (after minimum steps)
                if step >= min_steps:
                    if self.has_converged(change_history, threshold=convergence_threshold):
                        print(f"   ⚡ Converged at step {step}! Stopping early.")
                        stopped_early = True
                        actual_steps = step
                        return True  # Stop generation
            
            previous_latent = latents.clone()
            return False
        
        print(f"\n🎨 Generating: '{prompt}'")
        print(f"   Max steps: {max_steps}, Will check convergence after step {min_steps}")
        
        start_time = time.time()
        
        # Generate with callback
        with torch.inference_mode():
            result = self.pipe(
                prompt=prompt,
                num_inference_steps=max_steps,
                guidance_scale=7.5,
                generator=generator,
                callback=callback,
                callback_steps=1,
            )
        
        elapsed = time.time() - start_time
        
        # Results
        savings = ((max_steps - actual_steps) / max_steps) * 100 if stopped_early else 0
        
        print(f"   ✓ Completed in {elapsed:.2f}s")
        print(f"   📊 Steps: {actual_steps}/{max_steps}")
        if stopped_early:
            print(f"   💰 Saved {savings:.1f}% computation")
        else:
            print(f"   ⚠️  Ran full {max_steps} steps (didn't converge early)")
        
        return result.images[0], actual_steps, {
            'stopped_early': stopped_early,
            'actual_steps': actual_steps,
            'max_steps': max_steps,
            'generation_time': elapsed,
            'savings_percent': savings,
            'convergence_history': convergence_data,
        }
    
    def run_experiments(self, test_prompts, thresholds=[0.005, 0.01, 0.02]):
        """
        Test different convergence thresholds
        """
        results_dir = Path("week2_convergence_results")
        results_dir.mkdir(exist_ok=True)
        
        print("="*60)
        print("🔬 WEEK 2: CONVERGENCE DETECTION EXPERIMENTS")
        print("="*60)
        print()
        print("Testing dynamic early stopping with different thresholds:")
        print(f"  Thresholds: {thresholds}")
        print(f"  Prompts: {len(test_prompts)}")
        print()
        
        all_results = []
        
        for threshold in thresholds:
            print(f"\n{'='*60}")
            print(f"🎯 Testing threshold: {threshold}")
            print(f"{'='*60}")
            
            threshold_results = []
            
            for prompt in test_prompts:
                image, steps, data = self.generate_with_early_stopping(
                    prompt,
                    max_steps=30,
                    min_steps=10,
                    convergence_threshold=threshold,
                )
                
                # Save image
                prompt_slug = prompt[:30].replace(" ", "_").replace(",", "")
                img_path = results_dir / f"{prompt_slug}_threshold_{threshold}_steps_{steps}.png"
                image.save(img_path)
                
                result = {
                    'prompt': prompt,
                    'threshold': threshold,
                    'data': data,
                }
                threshold_results.append(result)
            
            all_results.append({
                'threshold': threshold,
                'results': threshold_results,
            })
        
        # Save results
        with open(results_dir / "convergence_results.json", "w") as f:
            json.dump(all_results, f, indent=2)
        
        # Print summary
        print("\n" + "="*60)
        print("📊 SUMMARY")
        print("="*60)
        
        for threshold_data in all_results:
            threshold = threshold_data['threshold']
            results = threshold_data['results']
            
            avg_steps = sum(r['data']['actual_steps'] for r in results) / len(results)
            avg_savings = sum(r['data']['savings_percent'] for r in results) / len(results)
            stopped_early_count = sum(1 for r in results if r['data']['stopped_early'])
            
            print(f"\n🎯 Threshold {threshold}:")
            print(f"   Average steps: {avg_steps:.1f}/30")
            print(f"   Average savings: {avg_savings:.1f}%")
            print(f"   Stopped early: {stopped_early_count}/{len(results)} prompts")
        
        print("\n" + "="*60)
        print("✅ EXPERIMENTS COMPLETE!")
        print(f"📁 Results saved to: {results_dir}")
        print("="*60)
        print()
        print("NEXT STEPS:")
        print("1. Look at the generated images")
        print("2. Compare early-stopped vs full 30-step images")
        print("3. Find optimal threshold (balance quality vs speed)")
        print("4. Document findings for Week 3!")


# =============================================================================
# MAIN EXPERIMENT SCRIPT
# =============================================================================

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║  WEEK 2: DYNAMIC CONVERGENCE DETECTION                  ║
    ║                                                          ║
    ║  Instead of predicting steps from text, we DETECT       ║
    ║  when the diffusion process has converged by monitoring ║
    ║  latent changes. Stop when converged = adaptive speed!  ║
    ║                                                          ║
    ║  Expected time: 10-15 minutes                           ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    input("Press ENTER to start experiments...")
    
    # Test prompts (diverse set)
    test_prompts = [
        "a red apple",
        "a serene mountain landscape at sunset",
        "a cyberpunk city street at night with neon lights",
        "a cute robot reading a book",
        "an intricate steampunk mechanical clock",
    ]
    
    detector = ConvergenceDetector()
    
    # Test different thresholds
    detector.run_experiments(
        test_prompts=test_prompts,
        thresholds=[0.005, 0.01, 0.02],  # Strict, medium, lenient
    )
    
    print("\n🎉 Week 2 experiments complete!")
    print("\nWhat you learned:")
    print("  • How to detect convergence dynamically")
    print("  • Whether early stopping maintains quality")
    print("  • Optimal threshold for stopping")
    print("\nNext: Analyze results and refine approach!")
