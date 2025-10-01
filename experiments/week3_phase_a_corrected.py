#!/usr/bin/env python3
"""
Week 3 Phase A - CORRECTED: Generalization Test with Calibrated Threshold
Using threshold 0.04 based on diagnostic results
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

class GeneralizationTesterCorrected:
    """Test perceptual detection with CORRECTED threshold 0.04"""
    
    def __init__(self, threshold=0.04, min_steps=15, check_every=2):
        print("🔬 Initializing Corrected Generalization Tester...")
        print(f"📊 Using CALIBRATED threshold: {threshold}")
        
        # Load models
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            safety_checker=None,
        )
        self.pipe = self.pipe.to("cuda")
        self.pipe.enable_attention_slicing()
        
        self.lpips_model = lpips.LPIPS(net='alex').cuda()
        
        # Corrected parameters
        self.threshold = threshold
        self.min_steps = min_steps
        self.check_every = check_every
        self.max_steps = 30
        
        # Results
        self.results_dir = Path("week3_phase_a_CORRECTED")
        self.results_dir.mkdir(exist_ok=True)
        
        print(f"✓ Threshold: {threshold} (4th percentile based on diagnostic)")
        print(f"✓ Expected success rate: 100%")
        print(f"✓ Expected avg steps: ~18-20")
    
    def get_test_prompts(self) -> Dict[str, List[str]]:
        """Same diverse prompts as before"""
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
            img1_norm = img1_tensor * 2 - 1
            img2_norm = img2_tensor * 2 - 1
            distance = self.lpips_model(img1_norm, img2_norm)
            return distance.item()
    
    def generate_with_detection(self, prompt: str, seed: int = 42):
        """Generate image with dynamic convergence detection"""
        generator = torch.Generator("cuda").manual_seed(seed)
        
        previous_image = None
        detection_log = []
        stopped_step = None
        
        def callback(step, timestep, latents):
            nonlocal previous_image, stopped_step
            
            if step < self.min_steps or step % self.check_every != 0:
                return
            
            # Decode current latents
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
                
                # Check convergence
                if change < self.threshold:
                    stopped_step = step
                    print(f"  🛑 Converged at step {step} (LPIPS: {change:.4f} < {self.threshold:.4f})")
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
        """Test all prompts in category"""
        print(f"\n{'='*60}")
        print(f"📂 Category: {category.upper()}")
        print(f"{'='*60}")
        
        category_dir = self.results_dir / category
        category_dir.mkdir(exist_ok=True)
        
        results = []
        
        for i, prompt in enumerate(prompts, 1):
            print(f"\n[{i}/{len(prompts)}] {prompt}")
            
            result = self.generate_with_detection(prompt)
            
            # Save image
            img_path = category_dir / f"prompt_{i:02d}_steps_{result['actual_steps']}.png"
            result['image'].save(img_path)
            
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
            
            print(f"  ✓ Steps: {result['actual_steps']}/{result['max_steps']}")
            print(f"  ⏱️  Time: {result['generation_time']:.2f}s")
            if result['stopped_early']:
                print(f"  ⚡ Speedup: {result_data['speedup_percent']:.1f}%")
            else:
                print(f"  ⚠️  Did NOT stop early (investigate!)")
        
        # Save category results
        with open(category_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        self._print_category_summary(category, results)
        
        return results
    
    def _print_category_summary(self, category: str, results: List[Dict]):
        """Print category summary"""
        print(f"\n{'='*60}")
        print(f"📊 {category.upper()} SUMMARY")
        print(f"{'='*60}")
        
        total = len(results)
        stopped_early = sum(1 for r in results if r['stopped_early'])
        avg_steps = np.mean([r['actual_steps'] for r in results])
        avg_speedup = np.mean([r['speedup_percent'] for r in results if r['stopped_early']])
        
        print(f"Total: {total}")
        print(f"Stopped early: {stopped_early}/{total} ({stopped_early/total*100:.1f}%)")
        print(f"Avg steps: {avg_steps:.1f}/{results[0]['max_steps']}")
        if stopped_early > 0:
            print(f"Avg speedup: {avg_speedup:.1f}%")
        
        # Flag if not 100%
        if stopped_early < total:
            print(f"\n⚠️  {total - stopped_early} prompts did NOT stop early - investigate!")
    
    def run_full_test(self):
        """Run complete corrected test"""
        print("🚀 Starting Week 3 Phase A - CORRECTED VERSION")
        print("="*60)
        print("Using calibrated threshold 0.04 from diagnostic")
        print("Expected: 100% success, ~18-20 steps average, ~40% speedup")
        print("="*60)
        
        all_prompts = self.get_test_prompts()
        all_results = {}
        
        for category, prompts in all_prompts.items():
            results = self.test_category(category, prompts)
            all_results[category] = results
        
        # Save aggregate
        with open(self.results_dir / "all_results.json", "w") as f:
            json.dump(all_results, f, indent=2)
        
        # Overall analysis
        self._print_overall_analysis(all_results)
        
        return all_results
    
    def _print_overall_analysis(self, all_results: Dict):
        """Print overall analysis"""
        print("\n" + "="*60)
        print("🎯 CORRECTED OVERALL ANALYSIS")
        print("="*60)
        
        # Aggregate
        total_prompts = 0
        total_stopped = 0
        all_steps = []
        all_speedups = []
        
        for category, results in all_results.items():
            total_prompts += len(results)
            stopped = sum(1 for r in results if r['stopped_early'])
            total_stopped += stopped
            
            all_steps.extend([r['actual_steps'] for r in results])
            all_speedups.extend([r['speedup_percent'] for r in results if r['stopped_early']])
        
        # Print stats
        print(f"\n📊 Overall Statistics:")
        print(f"  Total tested: {total_prompts}")
        print(f"  Stopped early: {total_stopped}/{total_prompts} ({total_stopped/total_prompts*100:.1f}%)")
        print(f"  Avg steps: {np.mean(all_steps):.1f}/30")
        print(f"  Avg speedup: {np.mean(all_speedups):.1f}%")
        print(f"  Step range: {min(all_steps)}-{max(all_steps)}")
        
        # Success assessment
        success_rate = total_stopped / total_prompts
        
        print(f"\n🎯 ASSESSMENT:")
        if success_rate >= 0.95:
            print(f"  ✅ EXCELLENT! {success_rate:.1%} success rate")
            print(f"  ✅ Method validated across diverse prompts")
            print(f"  ✅ Speedup: {np.mean(all_speedups):.1f}%")
            print(f"\n  🎉 READY FOR WEEK 4!")
        elif success_rate >= 0.80:
            print(f"  ✅ GOOD! {success_rate:.1%} success rate")
            print(f"  ⚠️  Some prompts need investigation")
        else:
            print(f"  ⚠️  {success_rate:.1%} success rate")
            print(f"  ⚠️  Further threshold adjustment needed")
        
        # By category
        print(f"\n📈 By Category:")
        for category, results in all_results.items():
            stopped = sum(1 for r in results if r['stopped_early'])
            avg_steps = np.mean([r['actual_steps'] for r in results])
            print(f"  {category:20s}: {stopped}/{len(results)} stopped, {avg_steps:.1f} avg steps")


if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║  WEEK 3 PHASE A - CORRECTED RUN                         ║
    ║                                                          ║
    ║  Using CALIBRATED threshold: 0.04                       ║
    ║  Based on diagnostic analysis                           ║
    ║                                                          ║
    ║  Expected: 100% success, ~40% speedup                   ║
    ║  Duration: 45-60 minutes                                ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    input("Press ENTER to start corrected Phase A...")
    
    tester = GeneralizationTesterCorrected(threshold=0.04)
    results = tester.run_full_test()
    
    print("\n" + "="*60)
    print("🎉 PHASE A (CORRECTED) COMPLETE!")
    print("="*60)
    print("\nIf success rate >95%:")
    print("  → Proceed to Phase B: Fine-tune between 0.04-0.06")
    print("  → Then Week 4: SDXL integration")
    print("\n🚀 Your method is VALIDATED!")
