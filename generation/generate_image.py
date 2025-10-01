#!/usr/bin/env python3
"""
Your first Stable Diffusion image generator
Optimized for RTX 5070Ti 16GB

This script will:
1. Download a Stable Diffusion model (first run only, ~5GB)
2. Generate an image from your text prompt
3. Save it as output.png

Learning notes embedded in comments!
"""

import torch
from diffusers import StableDiffusionPipeline
import time

import os

def generate_unique_name(filename):
    base, ext = os.path.splitext(filename)
    counter = 1
    new_filename = filename
    while os.path.exists(new_filename):
        new_filename = f"{base}_{counter}{ext}"
        counter += 1
    return new_filename

def main():
    print("=" * 60)
    print("STABLE DIFFUSION IMAGE GENERATOR")
    print("=" * 60)
    print()
    
    # Check GPU
    if not torch.cuda.is_available():
        print("❌ CUDA not available! This will be VERY slow on CPU.")
        print("   Check your PyTorch installation.")
        return
    
    print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"✓ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print()
    
    # MODEL SELECTION
    # We'll use Stable Diffusion 1.5 - it's smaller and perfect for learning
    # Later you can try: "stabilityai/stable-diffusion-xl-base-1.0" (SDXL)
    model_id = "runwayml/stable-diffusion-v1-5"
    
    print(f"📥 Loading model: {model_id}")
    print("   (First run: downloads ~5GB, takes 2-5 minutes)")
    print("   (Subsequent runs: loads from cache, ~10 seconds)")
    print()
    
    # Load the pipeline
    # This creates the full SD pipeline: VAE + UNet + Text Encoder
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
#        torch_dtype=torch.float16,  # Use FP16 for speed + memory efficiency
        safety_checker=None,         # Disable for local use (saves VRAM)
        use_safetensors=True,        # Use safer tensor format
    )
    
    # Move to GPU
    pipe = pipe.to("cuda")
    
    # MEMORY OPTIMIZATIONS for 16GB GPU
    # These are techniques you'll learn about later:
    
    # 1. Enable attention slicing (reduces VRAM at slight speed cost)
    pipe.enable_attention_slicing()
    
    # 2. Optional: Enable xformers (faster attention, needs xformers installed)
    # Disabled for RTX 5070 Ti compatibility - use native PyTorch attention instead
    # try:
    #     pipe.enable_xformers_memory_efficient_attention()
    #     print("✓ xformers enabled (faster generation)")
    # except:
    #     print("ℹ xformers not available (that's okay)")
    
    # Use PyTorch's scaled_dot_product_attention (SDPA) instead - works great!
    print("✓ Using PyTorch native attention (optimized for RTX 5070 Ti)")
    
    print()
    print("=" * 60)
    print("MODEL LOADED - Ready to generate!")
    print("=" * 60)
    print()
    
    # YOUR PROMPT
    prompt = "a serene mountain landscape at sunset, digital art, highly detailed"
    
    # You can change this to anything:
    # prompt = "a cute robot learning to paint, studio ghibli style"
    prompt = "cyberpunk city street at night in New Delhi, neon lights, rain"
    # prompt = "a cozy coffee shop interior, warm lighting, photorealistic"
    
    print(f"📝 Prompt: {prompt}")
    print()
    
    # GENERATION PARAMETERS
    # These control the output quality and randomness
    num_inference_steps = 20  # More steps = better quality, slower (20-50 typical)
    guidance_scale = 5.5      # How closely to follow prompt (7-10 typical)
    height = 512              # Image dimensions (512 or 768 for SD 1.5)
    width = 768               # SDXL can do 1024x1024
    
    print("⚙️  Generation settings:")
    print(f"   Steps: {num_inference_steps}")
    print(f"   Guidance: {guidance_scale}")
    print(f"   Size: {width}x{height}")
    print()
    
    print("🎨 Generating image...")
    start_time = time.time()
    
    # THE MAGIC HAPPENS HERE
    # This runs the diffusion process: starting from noise,
    # iteratively denoising to create your image
    with torch.inference_mode():  # Disable gradient tracking (inference only)
        image = pipe(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
        ).images[0]
    
    elapsed = time.time() - start_time
    
    # Save the result
    output_path = "output.png"
    output_path = generate_unique_name(output_path)
    image.save(output_path)
    
    print()
    print("=" * 60)
    print("✨ SUCCESS!")
    print("=" * 60)
    print(f"⏱️  Generation time: {elapsed:.2f} seconds")
    print(f"💾 Saved to: {output_path}")
    print()
    print("🎉 You just ran a diffusion model!")
    print()
    print("WHAT JUST HAPPENED:")
    print("1. Text → CLIP encoded your prompt into embeddings")
    print("2. Random noise → UNet iteratively denoised it (30 steps)")
    print("3. Latent space → VAE decoded it into a pixel image")
    print()
    print("EXPERIMENT:")
    print("- Change the 'prompt' variable above")
    print("- Try num_inference_steps=50 (slower but better)")
    print("- Try guidance_scale=10 (more literal to prompt)")
    print()
    print("Next: We'll learn what each of these components does!")

if __name__ == "__main__":
    main()