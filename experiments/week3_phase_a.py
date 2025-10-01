#!/usr/bin/env python3
"""
Week 3 Phase A: Medium-Scale Generalization Test
Test 25-30 diverse prompts to validate perceptual detection generalizes
"""

import torch
from diffusers import StableDiffusionPipeline
import lpips
import time
import json
from pathlib import Path
from PIL import Image
from typing import List, Dict
import numpy as np

class GeneralizationTester:
    """Test perceptual detection across diverse prompt categories"""
    
    def __init__(self, threshold=0.02, min_steps=15, check_every=2):
        print("🔬 Initializing Generalization Tester...")
        
        # Load SD pipeline
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            safety_checker=None,
        )
        self.pipe = self.pipe.to("cuda")
        self.pipe.enable_attention_slicing()
        
        # Load LPIPS for perceptual detection
        self.lpips_model = lpips.LPIPS(net='alex').cuda()
        
        # Detection parameters
        self.threshold = threshold
        self.min_steps = min_steps
        self.check_every = check_every
        self.max_steps = 30
        
        # Results
        self.results_dir = Path("week3_phase_a_results")
        self.results_dir.mkdir(exist_ok=True)
        
        print(f"✓ Threshold: {threshold}")
        print(f"✓ Min steps: {min_steps}")
        print(f"✓ Check every: {check_every}")
    
    def get_test_prompts(self) -> Dict[str, List[str]]:
        """Diverse prompts covering multiple categories"""
        return {
            'portraits': [
                "a portrait of a woman with red hair and green eyes",
                "an old man with a white beard, wise expression",
                "a young child laughing, natural lighting",
                "a profile view of a person in silhouette",
                "headshot photo of a professional businesswoman",
            ],
            'landscapes': [
                "a serene mountain landscape at sunset",
                "a tropical beach with palm trees and blue water",
                "a misty forest with tall trees and morning light",
                "a desert landscape with sand dunes and clear sky",
                "a snowy mountain peak under starry night sky",
            ],
            'objects': [
                "a vintage camera on a wooden desk",
                "a red apple on a white plate",
                "a stack of colorful books on a shelf",
                "a steaming cup of coffee on a table",
                "a modern smartphone on a marble surface",
            ],
            'complex_scenes': [
                "a busy marketplace with many people and colorful stalls",
                "a cozy living room with furniture, plants, and decorations",
                "a futuristic city street with cars and neon lights",
                "a medieval castle on a hill surrounded by villages",
                "a crowded train station with people and luggage",
            ],
            'abstract_artistic': [
                "swirling colors and abstract shapes, digital art",
                "a geometric pattern with triangles and circles",
                "watercolor painting of flowing water and light",
            ],
            'edge_cases': [
                "a simple solid red background",
                "a single white sphere on black background",
            ]
        }
    
    def compute_perceptual_change(self, img1_tensor, img2_tensor):
        """Compute LPIPS between two images"""
        with torch.no_grad():
            # LPIPS expects [-1, 1] range
            img1_norm = img1_tensor * 2 - 1
            img2_norm = img2_tensor * 2 - 1
            distance = self.lpips_model(img1_norm, img2_norm)
            return distance.item()
    
    def tensor_to_pil(self, tensor):
        """Convert tensor to PIL Image"""
        tensor = tensor.squeeze(0).cpu()
        tensor = torch.clamp(tensor, 0, 1)
        array = (tensor.permute(1, 2, 0).numpy() * 255).astype('uint8')
        return Image.fromarray(array)
    
    def generate_with_detection(self, prompt: str, seed: int = 42):
        """Generate image with dynamic convergence detection"""
        generator = torch.Generator("cuda").manual_seed(seed)
        
        # Storage for detection
        previous_image = None
        detection_log = []
        stopped_step = None
        
        def callback(step, timestep, latents):
            nonlocal previous_image, stopped_step
            
            # Only check after min_steps and at check intervals
            if step < self.min_steps or step % self.check_every != 0:
                return
            
            # Decode current latents to image
            latents_scaled = latents / self.pipe.vae.config.scaling_factor
            current_image = self.pipe.vae.decode(latents_scaled, return_dict=False)[0]
            
            if previous_image is not None:
                # Compute perceptual change
                change = self.compute_perceptual_change(previous_image, current_image)
                detection_log.append({
                    'step': step,
                    'lpips_change': change,
                    'threshold': self.threshold,
                    'stopped': change < self.threshold
                })
                
                # Check if converged
                if change < self.threshold:
                    stopped_step = step
                    print(f"  🛑 Convergence detected at step {step} (change: {change:.4f})")
                    # Force stop by returning False
                    return False
            
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
        
        # Determine actual steps
        actual_steps = stopped_step if stopped_step else self.max_steps
        
        return {
            'image': result.images[0],
            'actual_steps': actual_steps,
            'max_steps': self.max_steps,
            'stopped_early': stopped_step is not None,
            'generation_time': generation_time,
            'detection_log': detection_log,
        }
    
    def test_category(self, category: str, prompts: List[str]):
        """Test all prompts in a category"""
        print(f"\n{'='*60}")
        print(f"📂 Testing Category: {category.upper()}")
        print(f"{'='*60}")
        
        category_dir = self.results_dir / category
        category_dir.mkdir(exist_ok=True)
        
        results = []
        
        for i, prompt in enumerate(prompts, 1):
            print(f"\n[{i}/{len(prompts)}] {prompt}")
            
            # Generate with detection
            result = self.generate_with_detection(prompt)
            
            # Save image
            img_path = category_dir / f"prompt_{i:02d}_steps_{result['actual_steps']}.png"
            result['image'].save(img_path)
            
            # Store results
            result_data = {
                'prompt': prompt,
                'actual_steps': result['actual_steps'],
                'max_steps': result['max_steps'],
                'stopped_early': result['stopped_early'],
                'generation_time': result['generation_time'],
                'speedup_percent': ((result['max_steps'] - result['actual_steps']) / result['max_steps']) * 100,
                'detection_log': result['detection_log'],
            }
            results.append(result_data)
            
            # Print summary
            print(f"  ✓ Steps: {result['actual_steps']}/{result['max_steps']}")
            print(f"  ⏱️  Time: {result['generation_time']:.2f}s")
            if result['stopped_early']:
                print(f"  ⚡ Speedup: {result_data['speedup_percent']:.1f}%")
        
        # Save category results
        with open(category_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        # Category summary
        self._print_category_summary(category, results)
        
        return results
    
    def _print_category_summary(self, category: str, results: List[Dict]):
        """Print summary statistics for category"""
        print(f"\n{'='*60}")
        print(f"📊 {category.upper()} SUMMARY")
        print(f"{'='*60}")
        
        total = len(results)
        stopped_early = sum(1 for r in results if r['stopped_early'])
        avg_steps = np.mean([r['actual_steps'] for r in results])
        avg_speedup = np.mean([r['speedup_percent'] for r in results if r['stopped_early']])
        
        print(f"Total prompts: {total}")
        print(f"Stopped early: {stopped_early}/{total} ({stopped_early/total*100:.1f}%)")
        print(f"Average steps: {avg_steps:.1f}/{results[0]['max_steps']}")
        if stopped_early > 0:
            print(f"Average speedup (when stopped): {avg_speedup:.1f}%")
        print()
    
    def run_full_test(self):
        """Run complete generalization test"""
        print("🚀 Starting Week 3 Phase A: Generalization Test")
        print("="*60)
        
        all_prompts = self.get_test_prompts()
        all_results = {}
        
        for category, prompts in all_prompts.items():
            results = self.test_category(category, prompts)
            all_results[category] = results
        
        # Save aggregate results
        with open(self.results_dir / "all_results.json", "w") as f:
            json.dump(all_results, f, indent=2)
        
        # Print overall analysis
        self._print_overall_analysis(all_results)
        
        return all_results
    
    def _print_overall_analysis(self, all_results: Dict):
        """Print overall analysis across all categories"""
        print("\n" + "="*60)
        print("🎯 OVERALL GENERALIZATION ANALYSIS")
        print("="*60)
        
        # Aggregate statistics
        total_prompts = 0
        total_stopped = 0
        all_steps = []
        all_speedups = []
        
        category_stats = {}
        
        for category, results in all_results.items():
            total_prompts += len(results)
            stopped = sum(1 for r in results if r['stopped_early'])
            total_stopped += stopped
            
            steps = [r['actual_steps'] for r in results]
            speedups = [r['speedup_percent'] for r in results if r['stopped_early']]
            
            all_steps.extend(steps)
            all_speedups.extend(speedups)
            
            category_stats[category] = {
                'count': len(results),
                'stopped': stopped,
                'avg_steps': np.mean(steps),
                'avg_speedup': np.mean(speedups) if speedups else 0,
            }
        
        # Print statistics
        print(f"\n📊 Overall Statistics:")
        print(f"  Total prompts tested: {total_prompts}")
        print(f"  Stopped early: {total_stopped}/{total_prompts} ({total_stopped/total_prompts*100:.1f}%)")
        print(f"  Overall avg steps: {np.mean(all_steps):.1f}/30")
        print(f"  Overall avg speedup: {np.mean(all_speedups):.1f}%")
        
        print(f"\n📈 By Category:")
        for category, stats in category_stats.items():
            print(f"  {category:20s}: {stats['avg_steps']:.1f} steps, {stats['stopped']}/{stats['count']} stopped ({stats['avg_speedup']:.1f}% speedup)")
        
        # Analysis
        print(f"\n🔍 Key Insights:")
        
        if total_stopped / total_prompts > 0.8:
            print("  ✅ Excellent generalization - works across categories!")
        elif total_stopped / total_prompts > 0.5:
            print("  ⚠️  Moderate generalization - some categories problematic")
        else:
            print("  ❌ Poor generalization - threshold or method needs adjustment")
        
        # Identify problematic categories
        problem_categories = [cat for cat, stats in category_stats.items() 
                            if stats['stopped'] / stats['count'] < 0.5]
        if problem_categories:
            print(f"  ⚠️  Problematic categories: {', '.join(problem_categories)}")
        
        # Step distribution
        step_std = np.std(all_steps)
        print(f"\n  Step consistency: {step_std:.2f} std dev")
        if step_std < 3:
            print("  ✅ Very consistent stopping points")
        elif step_std < 5:
            print("  ✅ Reasonably consistent")
        else:
            print("  ⚠️  High variance - content-dependent behavior")
        
        print(f"\n💡 Recommendation:")
        if total_stopped / total_prompts > 0.8 and np.mean(all_steps) < 25:
            print("  ✅ Proceed to Phase B: Fine-tune threshold")
            print("  → Current threshold (0.02) works well!")
            print("  → Test nearby values to optimize further")
        elif total_stopped / total_prompts > 0.5:
            print("  ⚠️  Consider threshold adjustment before Phase B")
            if np.mean(all_steps) > 25:
                print("  → Try LOWER threshold (more aggressive stopping)")
            else:
                print("  → Investigate specific category failures")
        else:
            print("  ❌ Threshold needs significant adjustment")
            print("  → Try much lower threshold OR re-examine detection method")


if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║  WEEK 3 PHASE A: GENERALIZATION TEST                    ║
    ║                                                          ║
    ║  Testing 25-30 diverse prompts to validate:            ║
    ║  • Method generalizes across categories                 ║
    ║  • Threshold 0.02 is appropriate                        ║
    ║  • Identify any systematic failures                     ║
    ║                                                          ║
    ║  Expected time: 45-60 minutes                           ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    input("Press ENTER to start Phase A...")
    
    tester = GeneralizationTester(threshold=0.02, min_steps=15, check_every=2)
    results = tester.run_full_test()
    
    print("\n" + "="*60)
    print("🎉 PHASE A COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Review images in week3_phase_a_results/")
    print("2. Check category summaries")
    print("3. Based on results, decide:")
    print("   - If >80% stopped early → Proceed to Phase B (threshold optimization)")
    print("   - If 50-80% stopped → Adjust threshold, re-test")
    print("   - If <50% stopped → Re-examine detection method")
    print("\n🚀 See you for Phase B!")
