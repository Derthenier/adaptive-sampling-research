#!/usr/bin/env python3
"""
Week 4 Track 1: Quality Validation
Generate A/B comparisons and compute quality metrics
"""

import torch
from diffusers import StableDiffusionPipeline
import lpips
from PIL import Image
import numpy as np
from pathlib import Path
import json
from transformers import CLIPProcessor, CLIPModel

class QualityValidator:
    """Validate that 18-step generation maintains quality vs 30-step baseline"""
    
    def __init__(self, threshold=0.04):
        print("🔬 Quality Validation Mode")
        
        # Load SD pipeline
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            safety_checker=None,
        ).to("cuda")
        self.pipe.enable_attention_slicing()
        
        # Load metrics
        self.lpips_model = lpips.LPIPS(net='alex').cuda()
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        self.threshold = threshold
        
        self.results_dir = Path('week4_quality_validation')
        self.results_dir.mkdir(exist_ok=True)
    
    def pil_to_tensor(self, image):
        """Convert PIL to tensor for LPIPS"""
        tensor = torch.from_numpy(np.array(image)).float() / 255.0
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).cuda()
        return tensor
    
    def compute_lpips(self, img1, img2):
        """Compute perceptual distance between two images"""
        tensor1 = self.pil_to_tensor(img1) * 2 - 1
        tensor2 = self.pil_to_tensor(img2) * 2 - 1
        
        with torch.no_grad():
            distance = self.lpips_model(tensor1, tensor2)
        
        return distance.item()
    
    def compute_clip_score(self, image, prompt):
        """Compute CLIP text-image alignment"""
        inputs = self.clip_processor(
            text=[prompt],
            images=image,
            return_tensors="pt",
            padding=True
        ).to("cuda")
        
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            score = outputs.logits_per_image.item()
        
        return score
    
    def generate_adaptive(self, prompt, seed):
        """Generate with adaptive stopping"""
        
        previous_image = None
        stopped_step = None
        
        def callback(step, timestep, latents):
            nonlocal previous_image, stopped_step
            
            if step < 15 or step % 2 != 0:
                return
            
            latents_scaled = latents / self.pipe.vae.config.scaling_factor
            current_image = self.pipe.vae.decode(latents_scaled, return_dict=False)[0]
            
            if previous_image is not None:
                with torch.no_grad():
                    img1_norm = previous_image * 2 - 1
                    img2_norm = current_image * 2 - 1
                    change = self.lpips_model(img1_norm, img2_norm).item()
                
                if change < self.threshold:
                    stopped_step = step
                    return False
            
            previous_image = current_image.clone()
        
        generator = torch.Generator("cuda").manual_seed(seed)
        result = self.pipe(
            prompt=prompt,
            num_inference_steps=30,
            guidance_scale=7.5,
            generator=generator,
            callback=callback,
            callback_steps=1,
        )
        
        return result.images[0], stopped_step if stopped_step else 30
    
    def generate_baseline(self, prompt, seed, steps=30):
        """Generate baseline with fixed steps"""
        
        generator = torch.Generator("cuda").manual_seed(seed)
        result = self.pipe(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=7.5,
            generator=generator,
        )
        
        return result.images[0]
    
    def validate_prompt(self, prompt, seed=42):
        """Generate both versions and compare quality"""
        
        print(f"\n📝 {prompt}")
        
        # Generate adaptive version
        print("  🔄 Generating adaptive...")
        adaptive_img, adaptive_steps = self.generate_adaptive(prompt, seed)
        
        # Generate baseline
        print("  🔄 Generating baseline (30 steps)...")
        baseline_img = self.generate_baseline(prompt, seed, 30)
        
        # Compute metrics
        print("  📊 Computing metrics...")
        
        # Perceptual distance
        lpips_dist = self.compute_lpips(adaptive_img, baseline_img)
        
        # CLIP scores
        adaptive_clip = self.compute_clip_score(adaptive_img, prompt)
        baseline_clip = self.compute_clip_score(baseline_img, prompt)
        clip_diff = adaptive_clip - baseline_clip
        
        # Create comparison image
        comparison = self.create_comparison(adaptive_img, baseline_img, prompt, 
                                           adaptive_steps, lpips_dist, clip_diff)
        
        # Save
        prompt_slug = prompt[:40].replace(" ", "_").replace(",", "")
        comparison.save(self.results_dir / f"{prompt_slug}_comparison.png")
        adaptive_img.save(self.results_dir / f"{prompt_slug}_adaptive.png")
        baseline_img.save(self.results_dir / f"{prompt_slug}_baseline.png")
        
        # Results
        results = {
            'prompt': prompt,
            'seed': seed,
            'adaptive_steps': adaptive_steps,
            'baseline_steps': 30,
            'speedup_percent': ((30 - adaptive_steps) / 30) * 100,
            'lpips_distance': lpips_dist,
            'adaptive_clip_score': adaptive_clip,
            'baseline_clip_score': baseline_clip,
            'clip_difference': clip_diff,
        }
        
        # Print
        print(f"  ✓ Adaptive: {adaptive_steps} steps")
        print(f"  ✓ LPIPS distance: {lpips_dist:.4f} {'✅ LOW' if lpips_dist < 0.1 else '⚠️ HIGH'}")
        print(f"  ✓ CLIP scores: {adaptive_clip:.2f} vs {baseline_clip:.2f} (diff: {clip_diff:+.2f})")
        
        return results
    
    def create_comparison(self, img1, img2, prompt, steps, lpips_dist, clip_diff):
        """Create side-by-side comparison image"""
        
        width, height = img1.size
        comparison = Image.new('RGB', (width * 2 + 20, height + 100), 'white')
        
        # Paste images
        comparison.paste(img1, (0, 80))
        comparison.paste(img2, (width + 20, 80))
        
        # Add text (using PIL - simple version)
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(comparison)
        
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        except:
            font = ImageFont.load_default()
            font_small = font
        
        # Title
        draw.text((10, 10), f"Prompt: {prompt[:60]}", fill='black', font=font_small)
        
        # Labels
        draw.text((10, 50), f"Adaptive ({steps} steps)", fill='blue', font=font)
        draw.text((width + 30, 50), f"Baseline (30 steps)", fill='red', font=font)
        
        # Metrics
        draw.text((10, height + 85), 
                 f"LPIPS: {lpips_dist:.4f} | CLIP diff: {clip_diff:+.2f}", 
                 fill='black', font=font_small)
        
        return comparison
    
    def run_validation(self, num_prompts=15):
        """Run full quality validation"""
        
        print("="*60)
        print("WEEK 4: QUALITY VALIDATION")
        print("="*60)
        
        # Diverse prompts for validation
        prompts = [
            "a portrait of a woman with red hair",
            "a serene mountain landscape at sunset",
            "a vintage camera on a wooden desk",
            "a cozy coffee shop interior with people",
            "a futuristic city with flying cars",
            "a watercolor painting of a flower garden",
            "a professional headshot photograph",
            "a dramatic ocean wave crashing on rocks",
            "a steaming bowl of ramen noodles",
            "a geometric abstract art composition",
            "a cute puppy playing in grass",
            "a medieval castle on a misty hill",
            "a modern minimalist living room",
            "a busy Tokyo street at night with neon",
            "a detailed pencil sketch of a tree",
        ][:num_prompts]
        
        all_results = []
        
        for i, prompt in enumerate(prompts, 1):
            print(f"\n[{i}/{len(prompts)}]")
            results = self.validate_prompt(prompt, seed=42+i)
            all_results.append(results)
        
        # Save results
        with open(self.results_dir / 'validation_results.json', 'w') as f:
            json.dump(all_results, f, indent=2)
        
        # Analysis
        self.analyze_results(all_results)
        
        return all_results
    
    def analyze_results(self, results):
        """Analyze quality validation results"""
        
        print("\n" + "="*60)
        print("📊 QUALITY VALIDATION ANALYSIS")
        print("="*60)
        
        lpips_values = [r['lpips_distance'] for r in results]
        clip_diffs = [r['clip_difference'] for r in results]
        speedups = [r['speedup_percent'] for r in results]
        
        print(f"\n🔍 Perceptual Distance (LPIPS):")
        print(f"  Mean: {np.mean(lpips_values):.4f}")
        print(f"  Median: {np.median(lpips_values):.4f}")
        print(f"  Max: {max(lpips_values):.4f}")
        print(f"  % below 0.05: {sum(1 for x in lpips_values if x < 0.05) / len(lpips_values) * 100:.1f}%")
        print(f"  % below 0.10: {sum(1 for x in lpips_values if x < 0.10) / len(lpips_values) * 100:.1f}%")
        
        print(f"\n📊 CLIP Score Difference:")
        print(f"  Mean: {np.mean(clip_diffs):+.4f}")
        print(f"  Median: {np.median(clip_diffs):+.4f}")
        print(f"  Range: {min(clip_diffs):+.4f} to {max(clip_diffs):+.4f}")
        
        print(f"\n⚡ Speedup:")
        print(f"  Mean: {np.mean(speedups):.1f}%")
        
        # Assessment
        print(f"\n🎯 QUALITY ASSESSMENT:")
        
        avg_lpips = np.mean(lpips_values)
        avg_clip_diff = abs(np.mean(clip_diffs))
        
        if avg_lpips < 0.05 and avg_clip_diff < 1.0:
            print("  ✅ EXCELLENT - Minimal quality difference!")
            print("  ✅ Images are perceptually nearly identical")
        elif avg_lpips < 0.10 and avg_clip_diff < 2.0:
            print("  ✅ GOOD - Quality well maintained")
            print("  ✅ Differences are subtle")
        else:
            print("  ⚠️  Quality differences may be noticeable")
            print("  ⚠️  Visual inspection recommended")
        
        print(f"\n💡 Paper Claim:")
        print(f'  "Our method achieves {np.mean(speedups):.1f}% speedup while')
        print(f'   maintaining perceptual quality (LPIPS: {np.mean(lpips_values):.4f},')
        print(f'   CLIP score difference: {np.mean(clip_diffs):+.2f})"')


if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║  WEEK 4 TRACK 1: QUALITY VALIDATION                     ║
    ║                                                          ║
    ║  Generating A/B comparisons: adaptive vs baseline       ║
    ║  Computing quality metrics: LPIPS, CLIP                 ║
    ║                                                          ║
    ║  Duration: 30-40 minutes                                ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    input("Press ENTER to start quality validation...")
    
    validator = QualityValidator(threshold=0.04)
    results = validator.run_validation(num_prompts=15)
    
    print("\n🎉 Quality validation complete!")
    print(f"Check {validator.results_dir}/ for comparison images")
