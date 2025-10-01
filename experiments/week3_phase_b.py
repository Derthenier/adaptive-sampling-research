#!/usr/bin/env python3
"""
Week 3 Phase B: Threshold Optimization
Fine-tune threshold based on Phase A results
"""

import torch
from diffusers import StableDiffusionPipeline
import lpips
import time
import json
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

class ThresholdOptimizer:
    """Optimize LPIPS threshold for best speed/quality tradeoff"""
    
    def __init__(self, min_steps=15, check_every=2):
        print("🔬 Initializing Threshold Optimizer...")
        
        # Load models
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            safety_checker=None,
        )
        self.pipe = self.pipe.to("cuda")
        self.pipe.enable_attention_slicing()
        
        self.lpips_model = lpips.LPIPS(net='alex').cuda()
        
        # Fixed parameters
        self.min_steps = min_steps
        self.check_every = check_every
        self.max_steps = 30
        
        # Results
        self.results_dir = Path("week3_phase_b_results")
        self.results_dir.mkdir(exist_ok=True)
        
        print("✓ Ready to optimize!")
    
    def get_representative_prompts(self, phase_a_results=None):
        """
        Select representative prompts based on Phase A results
        If Phase A results available, pick:
        - 2 that worked perfectly
        - 2 that were borderline
        - 1 that hit max steps
        
        Otherwise, use diverse defaults
        """
        # Default representative prompts if Phase A not analyzed
        return [
            "a portrait of a woman with red hair",  # Simple, likely stops early
            "a busy marketplace with people and stalls",  # Complex, might need more steps
            "a mountain landscape at sunset",  # Medium complexity
            "a vintage camera on a wooden table",  # Object, controlled
            "swirling abstract colors, digital art",  # Abstract, unpredictable
        ]
    
    def compute_perceptual_change(self, img1_tensor, img2_tensor):
        """Compute LPIPS between two images"""
        with torch.no_grad():
            img1_norm = img1_tensor * 2 - 1
            img2_norm = img2_tensor * 2 - 1
            distance = self.lpips_model(img1_norm, img2_norm)
            return distance.item()
    
    def generate_with_threshold(self, prompt: str, threshold: float, seed: int = 42):
        """Generate with specific threshold"""
        generator = torch.Generator("cuda").manual_seed(seed)
        
        previous_image = None
        detection_log = []
        stopped_step = None
        
        def callback(step, timestep, latents):
            nonlocal previous_image, stopped_step
            
            if step < self.min_steps or step % self.check_every != 0:
                return
            
            # Decode to image
            latents_scaled = latents / self.pipe.vae.config.scaling_factor
            current_image = self.pipe.vae.decode(latents_scaled, return_dict=False)[0]
            
            if previous_image is not None:
                change = self.compute_perceptual_change(previous_image, current_image)
                detection_log.append({
                    'step': step,
                    'lpips_change': change,
                })
                
                # Check convergence
                if change < threshold:
                    stopped_step = step
                    return False  # Stop generation
            
            previous_image = current_image.clone()
        
        # Generate
        start_time = time.time()
        result = self.pipe(
            prompt=prompt,
            num_inference_steps=self.max_steps,
            guidance_scale=7.5,
            generator=generator,
            callback=callback,
            callback_steps=1,
        )
        generation_time = time.time() - start_time
        
        actual_steps = stopped_step if stopped_step else self.max_steps
        
        return {
            'image': result.images[0],
            'actual_steps': actual_steps,
            'stopped_early': stopped_step is not None,
            'generation_time': generation_time,
            'detection_log': detection_log,
        }
    
    def test_prompt_across_thresholds(self, prompt: str, thresholds: list, seed: int = 42):
        """Test single prompt across multiple thresholds"""
        print(f"\n📝 Testing: {prompt}")
        
        results = []
        
        for threshold in thresholds:
            print(f"  Testing threshold {threshold:.3f}...", end=' ')
            
            result = self.generate_with_threshold(prompt, threshold, seed)
            
            result_data = {
                'threshold': threshold,
                'actual_steps': result['actual_steps'],
                'stopped_early': result['stopped_early'],
                'generation_time': result['generation_time'],
                'speedup_percent': ((self.max_steps - result['actual_steps']) / self.max_steps) * 100,
            }
            results.append(result_data)
            
            print(f"{result['actual_steps']} steps ({result_data['speedup_percent']:.1f}% speedup)")
        
        return results
    
    def optimize(self, thresholds=None):
        """Run threshold optimization"""
        if thresholds is None:
            # Default: test around 0.02 baseline
            thresholds = [0.015, 0.018, 0.019, 0.020, 0.021, 0.022, 0.025]
        
        print("🚀 Starting Threshold Optimization")
        print("="*60)
        print(f"Testing thresholds: {thresholds}")
        print(f"Representative prompts: 5")
        print("="*60)
        
        prompts = self.get_representative_prompts()
        
        all_results = {}
        
        # Test each prompt
        for i, prompt in enumerate(prompts, 1):
            print(f"\n[{i}/{len(prompts)}] {prompt}")
            
            prompt_results = self.test_prompt_across_thresholds(prompt, thresholds)
            
            # Save images for comparison
            prompt_slug = f"prompt_{i:02d}"
            prompt_dir = self.results_dir / prompt_slug
            prompt_dir.mkdir(exist_ok=True)
            
            # Save result data
            with open(prompt_dir / "threshold_results.json", "w") as f:
                json.dump({
                    'prompt': prompt,
                    'results': prompt_results
                }, f, indent=2)
            
            all_results[prompt] = prompt_results
        
        # Analyze and recommend
        optimal_threshold = self._analyze_results(all_results, thresholds)
        
        # Save overall results
        with open(self.results_dir / "optimization_results.json", "w") as f:
            json.dump({
                'thresholds_tested': thresholds,
                'optimal_threshold': optimal_threshold,
                'all_results': all_results,
            }, f, indent=2)
        
        return optimal_threshold
    
    def _analyze_results(self, all_results: dict, thresholds: list):
        """Analyze results and recommend optimal threshold"""
        print("\n" + "="*60)
        print("📊 THRESHOLD OPTIMIZATION ANALYSIS")
        print("="*60)
        
        # Aggregate statistics per threshold
        threshold_stats = {t: [] for t in thresholds}
        
        for prompt, results in all_results.items():
            for result in results:
                threshold = result['threshold']
                threshold_stats[threshold].append({
                    'steps': result['actual_steps'],
                    'speedup': result['speedup_percent'],
                    'stopped': result['stopped_early'],
                })
        
        # Calculate metrics per threshold
        print("\n📈 Results by Threshold:")
        print(f"{'Threshold':<12} {'Avg Steps':<12} {'Avg Speedup':<15} {'Stop Rate':<12}")
        print("-" * 60)
        
        best_threshold = None
        best_score = 0
        
        for threshold in thresholds:
            stats = threshold_stats[threshold]
            avg_steps = np.mean([s['steps'] for s in stats])
            avg_speedup = np.mean([s['speedup'] for s in stats])
            stop_rate = sum(s['stopped'] for s in stats) / len(stats)
            
            # Score: balance speedup and stop rate
            # Want high speedup AND high stop rate
            score = avg_speedup * stop_rate
            
            print(f"{threshold:<12.3f} {avg_steps:<12.1f} {avg_speedup:<15.1f}% {stop_rate:<12.1%}")
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        print("\n" + "="*60)
        print(f"🎯 RECOMMENDED THRESHOLD: {best_threshold:.3f}")
        print("="*60)
        
        stats = threshold_stats[best_threshold]
        print(f"\nExpected performance:")
        print(f"  Average steps: {np.mean([s['steps'] for s in stats]):.1f}/30")
        print(f"  Average speedup: {np.mean([s['speedup'] for s in stats]):.1f}%")
        print(f"  Early stop rate: {sum(s['stopped'] for s in stats) / len(stats):.1%}")
        
        # Create visualization
        self._plot_threshold_analysis(threshold_stats, thresholds, best_threshold)
        
        return best_threshold
    
    def _plot_threshold_analysis(self, threshold_stats: dict, thresholds: list, best_threshold: float):
        """Create visualization of threshold optimization"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Steps vs Threshold
        avg_steps = [np.mean([s['steps'] for s in threshold_stats[t]]) for t in thresholds]
        ax1.plot(thresholds, avg_steps, 'bo-', linewidth=2, markersize=8)
        ax1.axvline(best_threshold, color='red', linestyle='--', label=f'Optimal: {best_threshold:.3f}')
        ax1.axhline(30, color='gray', linestyle=':', label='Max steps (30)')
        ax1.set_xlabel('LPIPS Threshold', fontsize=12)
        ax1.set_ylabel('Average Steps', fontsize=12)
        ax1.set_title('Steps vs Threshold', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Speedup vs Threshold
        avg_speedup = [np.mean([s['speedup'] for s in threshold_stats[t]]) for t in thresholds]
        ax2.plot(thresholds, avg_speedup, 'go-', linewidth=2, markersize=8)
        ax2.axvline(best_threshold, color='red', linestyle='--', label=f'Optimal: {best_threshold:.3f}')
        ax2.set_xlabel('LPIPS Threshold', fontsize=12)
        ax2.set_ylabel('Average Speedup (%)', fontsize=12)
        ax2.set_title('Speedup vs Threshold', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'threshold_optimization.png', dpi=150)
        print(f"\n📊 Visualization saved: {self.results_dir / 'threshold_optimization.png'}")


if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║  WEEK 3 PHASE B: THRESHOLD OPTIMIZATION                 ║
    ║                                                          ║
    ║  Testing thresholds: 0.015, 0.018, 0.019, 0.020,       ║
    ║                      0.021, 0.022, 0.025                ║
    ║                                                          ║
    ║  Will find optimal threshold for:                       ║
    ║  • Maximum speedup                                      ║
    ║  • Consistent early stopping                            ║
    ║  • Quality maintenance                                  ║
    ║                                                          ║
    ║  Expected time: 30-40 minutes                           ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    input("Press ENTER to start Phase B...")
    
    optimizer = ThresholdOptimizer(min_steps=15, check_every=2)
    optimal_threshold = optimizer.optimize()
    
    print("\n" + "="*60)
    print("🎉 PHASE B COMPLETE!")
    print("="*60)
    print(f"\nOptimal threshold found: {optimal_threshold:.3f}")
    print("\nNext steps:")
    print("1. Review threshold_optimization.png visualization")
    print("2. Use optimal threshold for Phase C (large-scale validation)")
    print("3. Document methodology for paper/blog post")
    print("\n🚀 Ready for Phase C: 100+ prompt validation!")
