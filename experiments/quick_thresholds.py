#!/usr/bin/env python3
"""
Quick test: Try much lower thresholds on 3 prompts to see if that's the issue
"""

import torch
from diffusers import StableDiffusionPipeline
import lpips
from pathlib import Path

def quick_threshold_test():
    """Test if lower thresholds work"""
    
    print("🔬 Quick Threshold Sanity Check")
    print("="*60)
    
    # Load models
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to("cuda")
    pipe.enable_attention_slicing()
    
    lpips_model = lpips.LPIPS(net='alex').cuda()
    
    # Test prompts
    test_prompts = [
        "a portrait of a woman with red hair",
        "a mountain landscape at sunset",
        "a coffee cup on a table",
    ]
    
    # Test these thresholds
    test_thresholds = [0.05, 0.08, 0.10, 0.15]
    
    results_dir = Path('quick_threshold_test')
    results_dir.mkdir(exist_ok=True)
    
    print(f"\nTesting thresholds: {test_thresholds}")
    print(f"On {len(test_prompts)} prompts\n")
    
    for prompt in test_prompts:
        print(f"\n📝 {prompt}")
        print("-" * 60)
        
        for threshold in test_thresholds:
            # Quick generation
            previous_image = None
            stopped_step = None
            lpips_log = []
            
            def callback(step, timestep, latents):
                nonlocal previous_image, stopped_step
                
                if step < 15 or step % 2 != 0:
                    return
                
                latents_scaled = latents / pipe.vae.config.scaling_factor
                current_image = pipe.vae.decode(latents_scaled, return_dict=False)[0]
                
                if previous_image is not None:
                    with torch.no_grad():
                        img1_norm = previous_image * 2 - 1
                        img2_norm = current_image * 2 - 1
                        change = lpips_model(img1_norm, img2_norm).item()
                    
                    lpips_log.append({'step': step, 'change': change})
                    
                    if change < threshold:
                        stopped_step = step
                        return False
                
                previous_image = current_image.clone()
            
            generator = torch.Generator("cuda").manual_seed(42)
            result = pipe(
                prompt=prompt,
                num_inference_steps=30,
                guidance_scale=7.5,
                generator=generator,
                callback=callback,
                callback_steps=1,
            )
            
            actual_steps = stopped_step if stopped_step else 30
            
            # Show LPIPS values
            if lpips_log:
                min_lpips = min(entry['change'] for entry in lpips_log)
                print(f"  Threshold {threshold:.2f}: {actual_steps:2d} steps (min LPIPS: {min_lpips:.4f})")
            else:
                print(f"  Threshold {threshold:.2f}: {actual_steps:2d} steps (no measurements)")
    
    print("\n" + "="*60)
    print("💡 If none of these stopped early, the LPIPS values are very high")
    print("   Check the debug_lpips_values.py output for the actual range")

if __name__ == "__main__":
    quick_threshold_test()
