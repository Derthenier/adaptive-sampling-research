#!/usr/bin/env python3
"""
Week 4 Track 1: Quality Validation - FIXED VERSION
The callback return False doesn't actually stop the pipeline in diffusers!
We need to manually run the denoising loop.
"""

import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
import lpips
from PIL import Image
import numpy as np
from pathlib import Path
import json
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

class QualityValidatorFixed:
    """Validate that adaptive stopping maintains quality - ACTUALLY stops early"""
    
    def __init__(self, threshold=0.04):
        print("🔬 Quality Validation Mode (FIXED)")
        
        # Load SD pipeline
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            safety_checker=None,
        ).to("cuda")
        self.pipe.enable_attention_slicing()
        
        # Use DDIM scheduler for consistency
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        
        # Load metrics
        self.lpips_model = lpips.LPIPS(net='alex').cuda()
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        self.threshold = threshold
        
        self.results_dir = Path('week4_quality_validation_FIXED')
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
    
    def latents_to_pil(self, latents):
        """Convert latents to PIL image"""
        latents = latents / self.pipe.vae.config.scaling_factor
        with torch.no_grad():
            image = self.pipe.vae.decode(latents, return_dict=False)[0]
        
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        image = (image * 255).round().astype("uint8")
        return Image.fromarray(image[0])
    
    def generate_adaptive_manual(self, prompt, seed, num_inference_steps=30):
        """
        MANUALLY run the denoising loop with early stopping.
        This is the CORRECT way to actually stop early!
        """
        # Prepare inputs
        generator = torch.Generator(device="cuda").manual_seed(seed)
        
        # Encode prompt
        text_inputs = self.pipe.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.pipe.text_encoder(text_inputs.input_ids.to("cuda"))[0]
        
        # Uncond embeddings for classifier-free guidance
        uncond_input = self.pipe.tokenizer(
            "",
            padding="max_length",
            max_length=self.pipe.tokenizer.model_max_length,
            return_tensors="pt",
        )
        uncond_embeddings = self.pipe.text_encoder(uncond_input.input_ids.to("cuda"))[0]
        
        # Concatenate for classifier-free guidance
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        
        # Prepare latents
        latents = torch.randn(
            (1, self.pipe.unet.config.in_channels, 64, 64),
            generator=generator,
            device="cuda",
            dtype=torch.float16,
        )
        
        # Set timesteps
        self.pipe.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.pipe.scheduler.timesteps
        
        # Scale initial noise
        latents = latents * self.pipe.scheduler.init_noise_sigma
        
        # Denoising loop with early stopping
        previous_image_tensor = None
        stopped_step = None
        
        for i, t in enumerate(tqdm(timesteps, desc="Adaptive generation")):
            # Expand latents for classifier-free guidance
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.pipe.scheduler.scale_model_input(latent_model_input, t)
            
            # Predict noise
            with torch.no_grad():
                noise_pred = self.pipe.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=text_embeddings,
                ).sample
            
            # Classifier-free guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond)
            
            # Compute previous latents
            latents = self.pipe.scheduler.step(noise_pred, t, latents).prev_sample
            
            # Check for convergence (after step 15, every 2 steps)
            if i >= 15 and i % 2 == 0:
                # Decode current latents
                current_image_tensor = self.latents_to_pil(latents)
                current_tensor = self.pil_to_tensor(current_image_tensor) * 2 - 1
                
                if previous_image_tensor is not None:
                    # Compute LPIPS
                    with torch.no_grad():
                        change = self.lpips_model(previous_image_tensor, current_tensor).item()
                    
                    print(f"  Step {i}: LPIPS change = {change:.4f}")
                    
                    if change < self.threshold:
                        stopped_step = i
                        print(f"  ✓ Converged at step {i}!")
                        break
                
                previous_image_tensor = current_tensor
        
        # Decode final image
        final_image = self.latents_to_pil(latents)
        actual_steps = stopped_step if stopped_step else num_inference_steps
        
        return final_image, actual_steps
    
    def generate_baseline(self, prompt, seed, steps=30):
        """Generate baseline with fixed steps (using standard pipeline)"""
        generator = torch.Generator(device="cuda").manual_seed(seed)
        result = self.pipe(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=7.5,
            generator=generator,
        )
        return result.images[0]
    
    def validate_prompt(self, prompt, seed=42):
        """Generate both versions and compare quality"""
        
        print(f"\n{'='*60}")
        print(f"📝 {prompt}")
        print(f"{'='*60}")
        
        # Generate adaptive version (ACTUALLY stops early now!)
        print("  🔄 Generating adaptive (with REAL early stopping)...")
        adaptive_img, adaptive_steps = self.generate_adaptive_manual(prompt, seed)
        
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
        
        # Save images
        prompt_slug = prompt[:40].replace(" ", "_").replace(",", "")
        adaptive_img.save(self.results_dir / f"{prompt_slug}_adaptive.png")
        baseline_img.save(self.results_dir / f"{prompt_slug}_baseline.png")
        
        # Create comparison
        comparison = self.create_comparison(adaptive_img, baseline_img, prompt, 
                                           adaptive_steps, lpips_dist, clip_diff)
        comparison.save(self.results_dir / f"{prompt_slug}_comparison.png")
        
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
        print(f"\n  ✓ Adaptive: {adaptive_steps} steps")
        print(f"  ✓ Baseline: 30 steps")
        print(f"  ✓ Speedup: {results['speedup_percent']:.1f}%")
        print(f"  ✓ LPIPS distance: {lpips_dist:.4f} {'✅ LOW' if lpips_dist < 0.1 else '⚠️ HIGH'}")
        print(f"  ✓ CLIP scores: {adaptive_clip:.2f} vs {baseline_clip:.2f} (diff: {clip_diff:+.2f})")
        
        return results
    
    def create_comparison(self, img1, img2, prompt, steps, lpips_dist, clip_diff):
        """Create side-by-side comparison image"""
        
        width, height = img1.size
        comparison = Image.new('RGB', (width * 2 + 20, height + 100), 'white')
        
        comparison.paste(img1, (0, 80))
        comparison.paste(img2, (width + 20, 80))
        
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(comparison)
        
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        except:
            font = ImageFont.load_default()
            font_small = font
        
        draw.text((10, 10), f"Prompt: {prompt[:60]}", fill='black', font=font_small)
        draw.text((10, 50), f"Adaptive ({steps} steps)", fill='blue', font=font)
        draw.text((width + 30, 50), f"Baseline (30 steps)", fill='red', font=font)
        draw.text((10, height + 85), 
                 f"LPIPS: {lpips_dist:.4f} | CLIP diff: {clip_diff:+.2f}", 
                 fill='black', font=font_small)
        
        return comparison
    
    def run_validation(self, num_prompts=15):
        """Run full quality validation"""
        
        print("="*60)
        print("WEEK 4: QUALITY VALIDATION (FIXED VERSION)")
        print("="*60)
        
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
            print(f"\n{'='*60}")
            print(f"[{i}/{len(prompts)}]")
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
    ║  WEEK 4 TRACK 1: QUALITY VALIDATION (FIXED)            ║
    ║                                                          ║
    ║  This version ACTUALLY stops early!                     ║
    ║  Manual denoising loop with real early stopping         ║
    ║                                                          ║
    ║  Duration: 40-50 minutes (manual loop is slower)        ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    input("Press ENTER to start FIXED quality validation...")
    
    validator = QualityValidatorFixed(threshold=0.04)
    results = validator.run_validation(num_prompts=15)
    
    print("\n🎉 Quality validation complete!")
    print(f"Check {validator.results_dir}/ for comparison images")