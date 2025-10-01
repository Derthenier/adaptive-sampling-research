#!/usr/bin/env python3
"""
Week 2 v2: Perceptual Convergence Detection

LESSON LEARNED: Latent changes don't correlate with visual changes!

NEW APPROACH: Measure PERCEPTUAL changes (what you actually see)
instead of latent space changes.
"""

import torch
from diffusers import StableDiffusionPipeline
import time
from pathlib import Path
import json
import lpips
from PIL import Image
import numpy as np


class PerceptualConvergenceDetector:
    """
    Detects convergence by measuring PERCEPTUAL changes (visual)
    instead of latent space changes
    """
    
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5"):
        print("🔬 Initializing Perceptual Convergence Detector v2...")
        print("   (Learning from v1: latent changes were wrong metric!)")
        
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
        self.pipe = self.pipe.to("cuda")
        self.pipe.enable_attention_slicing()
        
        # Load LPIPS for perceptual similarity
        print("   Loading LPIPS (perceptual metric)...")
        self.lpips_model = lpips.LPIPS(net='alex').cuda()
        
        print("✓ Ready!")
    
    def decode_latents(self, latents):
        """
        Decode latents to pixel images
        """
        # Scale latents
        latents = 1 / 0.18215 * latents
        
        # Decode with VAE
        with torch.no_grad():
            image = self.pipe.vae.decode(latents).sample
        
        # Convert to [-1, 1] range for LPIPS
        return image
    
    def measure_perceptual_change(self, latent_current, latent_previous):
        """
        Measure PERCEPTUAL change between consecutive steps
        
        Uses LPIPS (Learned Perceptual Image Patch Similarity)
        Returns perceptual distance (lower = more similar)
        """
        if latent_previous is None:
            return None
        
        # Decode both latents to images
        img_current = self.decode_latents(latent_current)
        img_previous = self.decode_latents(latent_previous)
        
        # Measure perceptual difference
        with torch.no_grad():
            perceptual_dist = self.lpips_model(img_current, img_previous)
        
        return perceptual_dist.item()
    
    def has_converged(self, change_history, window=3, threshold=0.02):
        """
        Determine if generation has converged based on perceptual changes
        
        Args:
            change_history: List of recent perceptual changes
            window: Number of recent steps to consider
            threshold: Maximum perceptual change to consider converged
        
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
        max_steps=30,
        min_steps=15,  # Higher min_steps since decoding is slower
        convergence_threshold=0.02,
        check_every=2,  # Check every N steps (decoding is expensive)
        seed=42
    ):
        """
        Generate image with perceptual early stopping
        
        Args:
            prompt: Text prompt
            max_steps: Maximum steps to run
            min_steps: Minimum steps before checking convergence
            convergence_threshold: Threshold for perceptual convergence
            check_every: Check convergence every N steps (save computation)
            seed: Random seed
        
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
            
            # Only check every N steps (decoding is expensive)
            if step % check_every != 0:
                previous_latent = latents.clone()
                return False
            
            # Measure perceptual change
            if previous_latent is not None:
                perceptual_change = self.measure_perceptual_change(
                    latents, previous_latent
                )
                
                change_history.append(perceptual_change)
                convergence_data.append({
                    'step': step,
                    'perceptual_change': perceptual_change,
                })
                
                print(f"   Step {step}: perceptual change = {perceptual_change:.4f}")
                
                # Check for convergence (after minimum steps)
                if step >= min_steps:
                    if self.has_converged(
                        change_history, 
                        window=2,  # Smaller window since we check less frequently
                        threshold=convergence_threshold
                    ):
                        print(f"   ⚡ CONVERGED at step {step}! Stopping early.")
                        stopped_early = True
                        actual_steps = step
                        return True  # Stop generation
            
            previous_latent = latents.clone()
            return False
        
        print(f"\n🎨 Generating: '{prompt}'")
        print(f"   Max steps: {max_steps}, checking convergence after step {min_steps}")
        print(f"   Checking every {check_every} steps (decoding is expensive)")
        
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
            print(f"   ⚠️  Ran full {max_steps} steps")
        
        return result.images[0], actual_steps, {
            'stopped_early': stopped_early,
            'actual_steps': actual_steps,
            'max_steps': max_steps,
            'generation_time': elapsed,
            'savings_percent': savings,
            'convergence_history': convergence_data,
        }
    
    def run_experiments(self, test_prompts, thresholds=[0.01, 0.02, 0.03]):
        """
        Test different perceptual convergence thresholds
        """
        results_dir = Path("week2_perceptual_results")
        results_dir.mkdir(exist_ok=True)
        
        print("="*60)
        print("🔬 WEEK 2 v2: PERCEPTUAL CONVERGENCE DETECTION")
        print("="*60)
        print()
        print("NEW APPROACH: Measuring visual/perceptual changes")
        print("(Not latent changes - we learned those don't work!)")
        print()
        print(f"Testing thresholds: {thresholds}")
        print(f"Prompts: {len(test_prompts)}")
        print()
        print("Note: This is slower than v1 (decoding at each check)")
        print("But measures what actually matters - visual quality!")
        print()
        
        all_results = []
        
        for threshold in thresholds:
            print(f"\n{'='*60}")
            print(f"🎯 Testing perceptual threshold: {threshold}")
            print(f"{'='*60}")
            
            threshold_results = []
            
            for prompt in test_prompts:
                image, steps, data = self.generate_with_early_stopping(
                    prompt,
                    max_steps=30,
                    min_steps=15,
                    convergence_threshold=threshold,
                    check_every=2,  # Check every 2 steps
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
        with open(results_dir / "perceptual_results.json", "w") as f:
            json.dump(all_results, f, indent=2)
        
        # Print summary
        print("\n" + "="*60)
        print("📊 SUMMARY - PERCEPTUAL CONVERGENCE")
        print("="*60)
        
        for threshold_data in all_results:
            threshold = threshold_data['threshold']
            results = threshold_data['results']
            
            if not results:
                continue
            
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
        print("COMPARE:")
        print("  Week 2 v1 (latent changes): 0% savings")
        print("  Week 2 v2 (perceptual): [see results above]")
        print()
        print("Did perceptual detection work better? Check the images!")


# =============================================================================
# MAIN EXPERIMENT SCRIPT
# =============================================================================

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║  WEEK 2 v2: PERCEPTUAL CONVERGENCE DETECTION            ║
    ║                                                          ║
    ║  LESSON FROM v1: Latent changes don't work!             ║
    ║                                                          ║
    ║  NEW APPROACH: Measure VISUAL/PERCEPTUAL changes        ║
    ║  using LPIPS (what humans actually perceive).           ║
    ║                                                          ║
    ║  This measures what matters: visual quality!            ║
    ║                                                          ║
    ║  Expected time: 15-20 minutes (slower but better!)      ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # Check if lpips is installed
    try:
        import lpips
    except ImportError:
        print("❌ LPIPS not installed!")
        print("   Install with: pip install lpips")
        print()
        input("Install lpips and press ENTER to continue...")
        import lpips
    
    input("Press ENTER to start perceptual convergence experiments...")
    
    # Test prompts (same as v1 for comparison)
    test_prompts = [
        "a red apple",
        "a serene mountain landscape at sunset",
        "a cyberpunk city street at night with neon lights",
        "a cute robot reading a book",
        "an intricate steampunk mechanical clock",
    ]
    
    detector = PerceptualConvergenceDetector()
    
    # Test different thresholds
    # Note: Perceptual thresholds are different scale than latent thresholds
    detector.run_experiments(
        test_prompts=test_prompts,
        thresholds=[0.01, 0.02, 0.03],  # LPIPS scale
    )
    
    print("\n🎉 Week 2 v2 complete!")
    print("\nWhat you learned:")
    print("  • Latent changes ≠ Visual changes (v1 lesson)")
    print("  • Perceptual metrics matter more (v2 approach)")
    print("  • How to pivot when hypothesis fails")
    print("\nNext: Analyze if perceptual detection works better!")