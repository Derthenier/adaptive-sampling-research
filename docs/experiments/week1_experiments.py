#!/usr/bin/env python3
"""
Week 1: Baseline Analysis - Understanding Step Count Impact

This script systematically tests different step counts to understand:
1. Quality vs speed tradeoff
2. When images converge
3. Which prompts need more/fewer steps

Run this to collect your first research data!
"""

import torch
from diffusers import StableDiffusionPipeline
import time
import json
from pathlib import Path
from PIL import Image
import numpy as np

# For quality metrics
from transformers import CLIPProcessor, CLIPModel


class StepAnalyzer:
    """Analyze impact of different step counts on generation"""
    
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5"):
        print("🔬 Initializing Step Analyzer...")
        
        # Load SD pipeline
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
        self.pipe = self.pipe.to("cuda")
        self.pipe.enable_attention_slicing()
        
        # Load CLIP for quality metrics
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Results storage
        self.results_dir = Path("week1_results")
        self.results_dir.mkdir(exist_ok=True)
        
        print("✓ Ready to analyze!")
    
    def compute_clip_score(self, image, prompt):
        """
        Compute CLIP score (text-image alignment)
        Higher = better alignment with prompt
        """
        inputs = self.clip_processor(
            text=[prompt],
            images=image,
            return_tensors="pt",
            padding=True
        ).to("cuda")
        
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            # Compute cosine similarity
            logits_per_image = outputs.logits_per_image
            score = logits_per_image.item()
        
        return score
    
    def generate_with_trajectory(self, prompt, num_steps, seed=42):
        """
        Generate image and save intermediate latents
        Returns: final_image, generation_time, trajectory_data
        """
        generator = torch.Generator("cuda").manual_seed(seed)
        
        # Store intermediate latents
        latent_trajectory = []
        
        def callback(step, timestep, latents):
            # Save latent at this step
            latent_trajectory.append({
                'step': step,
                'timestep': timestep.item(),
                'latent_mean': latents.mean().item(),
                'latent_std': latents.std().item(),
            })
        
        start_time = time.time()
        
        with torch.inference_mode():
            result = self.pipe(
                prompt=prompt,
                num_inference_steps=num_steps,
                guidance_scale=7.5,
                generator=generator,
                callback=callback,
                callback_steps=1,  # Call every step
            )
        
        elapsed = time.time() - start_time
        
        return result.images[0], elapsed, latent_trajectory
    
    def analyze_prompt(self, prompt, step_counts=[10, 15, 20, 25, 30, 40, 50]):
        """
        Analyze a single prompt across different step counts
        """
        print(f"\n{'='*60}")
        print(f"📝 Analyzing: {prompt}")
        print(f"{'='*60}")
        
        # Create prompt directory
        prompt_slug = prompt[:30].replace(" ", "_").replace(",", "")
        prompt_dir = self.results_dir / prompt_slug
        prompt_dir.mkdir(exist_ok=True)
        
        results = {
            'prompt': prompt,
            'experiments': []
        }
        
        # Test each step count
        for steps in step_counts:
            print(f"\n🎨 Generating with {steps} steps...")
            
            # Generate
            image, gen_time, trajectory = self.generate_with_trajectory(
                prompt, steps
            )
            
            # Compute quality metric
            clip_score = self.compute_clip_score(image, prompt)
            
            # Save image
            img_path = prompt_dir / f"steps_{steps:02d}.png"
            image.save(img_path)
            
            # Store results
            experiment = {
                'steps': steps,
                'generation_time': gen_time,
                'clip_score': clip_score,
                'trajectory': trajectory,
            }
            results['experiments'].append(experiment)
            
            print(f"  ⏱️  Time: {gen_time:.2f}s")
            print(f"  📊 CLIP Score: {clip_score:.2f}")
            print(f"  💾 Saved: {img_path.name}")
        
        # Save results JSON
        with open(prompt_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        # Print analysis
        self._print_analysis(results)
        
        return results
    
    def _print_analysis(self, results):
        """Print summary analysis"""
        print(f"\n{'='*60}")
        print("📊 ANALYSIS SUMMARY")
        print(f"{'='*60}")
        
        experiments = results['experiments']
        
        # Find best quality
        best_quality = max(experiments, key=lambda x: x['clip_score'])
        print(f"\n🏆 Best Quality:")
        print(f"   Steps: {best_quality['steps']}")
        print(f"   CLIP: {best_quality['clip_score']:.2f}")
        print(f"   Time: {best_quality['generation_time']:.2f}s")
        
        # Find fastest
        fastest = min(experiments, key=lambda x: x['generation_time'])
        print(f"\n⚡ Fastest:")
        print(f"   Steps: {fastest['steps']}")
        print(f"   Time: {fastest['generation_time']:.2f}s")
        print(f"   CLIP: {fastest['clip_score']:.2f}")
        
        # Quality at 30 steps (baseline)
        baseline = next(e for e in experiments if e['steps'] == 30)
        baseline_clip = baseline['clip_score']
        
        # Find minimal steps to achieve 95% of baseline quality
        threshold = baseline_clip * 0.95
        acceptable = [e for e in experiments if e['clip_score'] >= threshold]
        if acceptable:
            minimal = min(acceptable, key=lambda x: x['steps'])
            print(f"\n✅ Minimal Acceptable (≥95% quality):")
            print(f"   Steps: {minimal['steps']} (vs 30 baseline)")
            print(f"   Quality: {minimal['clip_score']:.2f} (vs {baseline_clip:.2f})")
            print(f"   Time: {minimal['generation_time']:.2f}s (vs {baseline['generation_time']:.2f}s)")
            speedup = (baseline['generation_time'] / minimal['generation_time'] - 1) * 100
            print(f"   Speedup: {speedup:.1f}%")
        
        print()
    
    def run_benchmark_suite(self):
        """
        Run full benchmark across diverse prompts
        """
        print("🚀 Starting Benchmark Suite")
        print("="*60)
        
        # Diverse test prompts categorized by complexity
        prompts = {
            'simple': [
                "a red apple",
                "blue sky",
                "a cat sitting",
            ],
            'medium': [
                "a serene mountain landscape at sunset",
                "a robot reading a book in a library",
                "a coffee cup on a wooden table",
            ],
            'complex': [
                "a cyberpunk city street at night with neon lights, people walking, and rain falling",
                "three cats and two dogs playing in a garden with flowers and butterflies",
                "an intricate steampunk mechanical clock with gears, made of brass and copper",
            ],
        }
        
        all_results = {}
        
        for category, prompt_list in prompts.items():
            print(f"\n{'='*60}")
            print(f"📂 Category: {category.upper()}")
            print(f"{'='*60}")
            
            all_results[category] = []
            
            for prompt in prompt_list:
                result = self.analyze_prompt(prompt)
                all_results[category].append(result)
        
        # Save aggregate results
        with open(self.results_dir / "benchmark_results.json", "w") as f:
            json.dump(all_results, f, indent=2)
        
        print("\n" + "="*60)
        print("✅ BENCHMARK COMPLETE!")
        print(f"📁 Results saved to: {self.results_dir}")
        print("="*60)
        
        self._print_category_analysis(all_results)
    
    def _print_category_analysis(self, all_results):
        """Analyze patterns across categories"""
        print("\n" + "="*60)
        print("🔍 CATEGORY ANALYSIS")
        print("="*60)
        
        for category, results_list in all_results.items():
            print(f"\n📊 {category.upper()}:")
            
            # Average steps needed for 95% quality
            minimal_steps = []
            for result in results_list:
                experiments = result['experiments']
                baseline = next(e for e in experiments if e['steps'] == 30)
                threshold = baseline['clip_score'] * 0.95
                acceptable = [e for e in experiments if e['clip_score'] >= threshold]
                if acceptable:
                    minimal = min(acceptable, key=lambda x: x['steps'])
                    minimal_steps.append(minimal['steps'])
            
            if minimal_steps:
                avg_minimal = np.mean(minimal_steps)
                print(f"   Average minimal steps: {avg_minimal:.1f}")
                print(f"   Range: {min(minimal_steps)}-{max(minimal_steps)}")
                
                potential_speedup = ((30 / avg_minimal) - 1) * 100
                print(f"   Potential speedup: {potential_speedup:.1f}%")


# =============================================================================
# MAIN EXPERIMENT SCRIPT
# =============================================================================

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║  WEEK 1: ADAPTIVE SAMPLING BASELINE ANALYSIS            ║
    ║                                                          ║
    ║  This will test different step counts to understand:    ║
    ║  • Quality vs speed tradeoffs                           ║
    ║  • Which prompts converge faster                        ║
    ║  • Potential for adaptive sampling                      ║
    ║                                                          ║
    ║  Expected time: 30-45 minutes                           ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    input("Press ENTER to start experiments...")
    
    analyzer = StepAnalyzer()
    analyzer.run_benchmark_suite()
    
    print("\n" + "="*60)
    print("🎉 WEEK 1 EXPERIMENTS COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Review the images in week1_results/")
    print("2. Look at the JSON data")
    print("3. Identify patterns:")
    print("   - Do simple prompts converge faster?")
    print("   - What's the sweet spot for steps?")
    print("   - How much quality loss at fewer steps?")
    print("\n4. Document your findings")
    print("5. Form hypothesis for your predictor!")
    print("\nSee you for Week 2 experiments! 🚀")
