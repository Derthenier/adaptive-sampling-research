#!/usr/bin/env python3
"""
Week 5 Data Collection Script - SD 1.5 Predictor Training Data
===============================================================

Generates images at different step counts to create labeled training data
for the step predictor model.

Hardware: Optimized for RTX 5070 Ti (16GB VRAM)
Model: Stable Diffusion 1.5
Output: CSV with prompt, optimal_steps, and quality metrics

Note: Will port to SDXL in Week 7-8 after validating predictor approach
"""

import torch
import lpips
import pandas as pd
import numpy as np
from diffusers import StableDiffusionPipeline
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime
import gc

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Model settings
    MODEL_ID = "runwayml/stable-diffusion-v1-5"
    DEVICE = "cuda"
    DTYPE = torch.float16
    
    # Step counts to test (SD 1.5 default is 50, but 30 is common)
    STEP_COUNTS = [15, 18, 20, 22, 25, 30]
    BASELINE_STEPS = 30  # Reference for quality comparison
    
    # Generation settings
    GUIDANCE_SCALE = 7.5
    WIDTH = 512
    HEIGHT = 512
    SEED = 42  # Fixed seed for reproducibility
    
    # Quality thresholds
    LPIPS_THRESHOLD = 0.10  # Perceptual similarity threshold
    
    # Paths
    PROMPT_FILE = "data/training_prompts_sd15.txt"
    OUTPUT_DIR = "week5_training_data"
    CSV_OUTPUT = "data/training_data.csv"
    CHECKPOINT_FILE = "data/collection_checkpoint.json"
    
    # Performance
    ENABLE_ATTENTION_SLICING = True
    CLEAR_CACHE_FREQUENCY = 5  # Clear CUDA cache every N prompts

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def setup_directories():
    """Create necessary directories"""
    Path(Config.OUTPUT_DIR).mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)
    for steps in Config.STEP_COUNTS:
        Path(f"{Config.OUTPUT_DIR}/steps_{steps}").mkdir(exist_ok=True)
    print(f"✅ Directories created in {Config.OUTPUT_DIR}/")

def load_prompts():
    """Load prompts from file"""
    prompt_file = Path(Config.PROMPT_FILE)
    if not prompt_file.exists():
        raise FileNotFoundError(
            f"Prompt file not found: {Config.PROMPT_FILE}\n"
            "Please create this file with one prompt per line."
        )
    
    with open(prompt_file, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]
    
    print(f"✅ Loaded {len(prompts)} prompts from {Config.PROMPT_FILE}")
    return prompts

def load_checkpoint():
    """Load checkpoint to resume interrupted collection"""
    checkpoint_path = Path(Config.CHECKPOINT_FILE)
    if checkpoint_path.exists():
        with open(checkpoint_path, 'r') as f:
            return json.load(f)
    return {"completed_prompts": [], "results": []}

def save_checkpoint(checkpoint_data):
    """Save checkpoint for resuming"""
    with open(Config.CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)

def initialize_pipeline():
    """Load SD 1.5 pipeline with optimizations"""
    print("🔄 Loading SD 1.5 pipeline...")
    print(f"   Model: {Config.MODEL_ID}")
    print(f"   Device: {Config.DEVICE}")
    print(f"   Precision: {Config.DTYPE}")
    
    pipe = StableDiffusionPipeline.from_pretrained(
        Config.MODEL_ID,
        torch_dtype=Config.DTYPE,
        safety_checker=None  # Disable for faster loading
    )
    pipe = pipe.to(Config.DEVICE)
    
    # Memory optimizations for RTX 5070 Ti
    if Config.ENABLE_ATTENTION_SLICING:
        pipe.enable_attention_slicing()
        print("   ✅ Attention slicing enabled")
    
    # Optional: Enable VAE slicing for even lower memory
    pipe.enable_vae_slicing()
    print("   ✅ VAE slicing enabled")
    
    print("✅ Pipeline loaded successfully\n")
    return pipe

def initialize_lpips():
    """Initialize LPIPS metric"""
    print("🔄 Loading LPIPS model...")
    lpips_model = lpips.LPIPS(net='alex').to(Config.DEVICE)
    print("✅ LPIPS model loaded\n")
    return lpips_model

def generate_image(pipe, prompt, num_steps, seed):
    """Generate single image with specified parameters"""
    generator = torch.Generator(device=Config.DEVICE).manual_seed(seed)
    
    image = pipe(
        prompt=prompt,
        num_inference_steps=num_steps,
        guidance_scale=Config.GUIDANCE_SCALE,
        width=Config.WIDTH,
        height=Config.HEIGHT,
        generator=generator
    ).images[0]
    
    return image

def compute_lpips(lpips_model, img1, img2):
    """Compute LPIPS distance between two images"""
    # Convert PIL images to tensors
    def pil_to_tensor(img):
        img_array = np.array(img).astype(np.float32) / 255.0
        img_array = img_array * 2.0 - 1.0  # Normalize to [-1, 1]
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
        return img_tensor.to(Config.DEVICE)
    
    tensor1 = pil_to_tensor(img1)
    tensor2 = pil_to_tensor(img2)
    
    with torch.no_grad():
        distance = lpips_model(tensor1, tensor2)
    
    return distance.item()

def find_optimal_steps(lpips_scores):
    """
    Find minimum steps that maintain quality below threshold
    
    Args:
        lpips_scores: dict mapping step_count -> lpips_distance
    
    Returns:
        optimal_steps: int or None if all fail threshold
    """
    # Sort by step count
    sorted_steps = sorted(lpips_scores.keys())
    
    for steps in sorted_steps:
        if steps == Config.BASELINE_STEPS:
            continue  # Skip baseline (distance = 0)
        
        if lpips_scores[steps] < Config.LPIPS_THRESHOLD:
            return steps
    
    # If none meet threshold, return baseline
    return Config.BASELINE_STEPS

# ============================================================================
# MAIN COLLECTION LOOP
# ============================================================================

def collect_data_for_prompt(pipe, lpips_model, prompt, prompt_idx, total_prompts):
    """
    Generate images at all step counts and compute metrics
    
    Returns:
        dict with results for this prompt
    """
    print(f"\n{'='*80}")
    print(f"Prompt {prompt_idx + 1}/{total_prompts}")
    print(f"{'='*80}")
    print(f"📝 {prompt[:100]}..." if len(prompt) > 100 else f"📝 {prompt}")
    print()
    
    results = {
        'prompt': prompt,
        'lpips_scores': {},
        'images': {}
    }
    
    # Generate baseline (50 steps) first
    print(f"🎨 Generating baseline ({Config.BASELINE_STEPS} steps)...")
    baseline_img = generate_image(pipe, prompt, Config.BASELINE_STEPS, Config.SEED)
    results['images'][Config.BASELINE_STEPS] = baseline_img
    results['lpips_scores'][Config.BASELINE_STEPS] = 0.0  # Perfect match to itself
    
    # Save baseline image
    baseline_path = f"{Config.OUTPUT_DIR}/steps_{Config.BASELINE_STEPS}/prompt_{prompt_idx:03d}.png"
    baseline_img.save(baseline_path)
    print(f"   ✅ Saved to {baseline_path}")
    
    # Generate and compare at other step counts
    for steps in Config.STEP_COUNTS:
        if steps == Config.BASELINE_STEPS:
            continue
        
        print(f"\n🎨 Generating with {steps} steps...")
        img = generate_image(pipe, prompt, steps, Config.SEED)
        results['images'][steps] = img
        
        # Compute LPIPS vs baseline
        lpips_dist = compute_lpips(lpips_model, img, baseline_img)
        results['lpips_scores'][steps] = lpips_dist
        
        # Save image
        img_path = f"{Config.OUTPUT_DIR}/steps_{steps}/prompt_{prompt_idx:03d}.png"
        img.save(img_path)
        
        # Status indicator
        status = "✅ PASS" if lpips_dist < Config.LPIPS_THRESHOLD else "⚠️  FAIL"
        print(f"   LPIPS: {lpips_dist:.4f} {status}")
        print(f"   Saved to {img_path}")
    
    # Find optimal steps
    optimal = find_optimal_steps(results['lpips_scores'])
    results['optimal_steps'] = optimal
    results['lpips_at_optimal'] = results['lpips_scores'][optimal]
    
    # Calculate speedup
    speedup_pct = ((Config.BASELINE_STEPS - optimal) / Config.BASELINE_STEPS) * 100
    
    print(f"\n{'='*80}")
    print(f"📊 RESULTS FOR THIS PROMPT:")
    print(f"   Optimal steps: {optimal}/{Config.BASELINE_STEPS}")
    print(f"   LPIPS at optimal: {results['lpips_at_optimal']:.4f}")
    print(f"   Speedup: {speedup_pct:.1f}%")
    print(f"{'='*80}\n")
    
    return results

def main():
    """Main data collection pipeline"""
    print("\n" + "="*80)
    print("WEEK 5 DATA COLLECTION - SD 1.5 ADAPTIVE SAMPLING")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Setup
    setup_directories()
    prompts = load_prompts()
    checkpoint = load_checkpoint()
    
    # Initialize models
    pipe = initialize_pipeline()
    lpips_model = initialize_lpips()
    
    # Track overall results
    all_results = checkpoint.get('results', [])
    completed_prompts = set(checkpoint.get('completed_prompts', []))
    
    print(f"📊 COLLECTION PLAN:")
    print(f"   Total prompts: {len(prompts)}")
    print(f"   Step counts: {Config.STEP_COUNTS}")
    print(f"   Images per prompt: {len(Config.STEP_COUNTS)}")
    print(f"   Total images: {len(prompts) * len(Config.STEP_COUNTS)}")
    print(f"   Already completed: {len(completed_prompts)} prompts")
    print(f"   Remaining: {len(prompts) - len(completed_prompts)} prompts")
    print()
    
    estimated_time = (len(prompts) - len(completed_prompts)) * len(Config.STEP_COUNTS) * 5  # ~5s per image (SD 1.5)
    print(f"⏱️  Estimated time: {estimated_time // 3600:.1f} hours ({estimated_time // 60:.0f} minutes)")
    print(f"   (SD 1.5 is faster than SDXL - should complete in ~45 minutes)")
    print()
    
    input("Press ENTER to start collection...")
    print()
    
    # Main loop
    try:
        for idx, prompt in enumerate(prompts):
            # Skip if already completed
            if idx in completed_prompts:
                print(f"⏭️  Skipping prompt {idx + 1} (already completed)")
                continue
            
            # Collect data for this prompt
            result = collect_data_for_prompt(pipe, lpips_model, prompt, idx, len(prompts))
            
            # Store results
            all_results.append({
                'prompt_idx': idx,
                'prompt': prompt,
                'optimal_steps': result['optimal_steps'],
                'lpips_at_optimal': result['lpips_at_optimal'],
                'lpips_scores': result['lpips_scores']
            })
            completed_prompts.add(idx)
            
            # Save checkpoint
            save_checkpoint({
                'completed_prompts': list(completed_prompts),
                'results': all_results
            })
            
            # Periodic cache clearing
            if (idx + 1) % Config.CLEAR_CACHE_FREQUENCY == 0:
                gc.collect()
                torch.cuda.empty_cache()
                print("\n🧹 Cleared CUDA cache\n")
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Collection interrupted by user")
        print("💾 Progress saved to checkpoint file")
        print("   Run script again to resume from where you left off\n")
        return
    
    except Exception as e:
        print(f"\n\n❌ Error occurred: {e}")
        print("💾 Progress saved to checkpoint file")
        print("   Run script again to resume from where you left off\n")
        raise
    
    # Save final results to CSV
    print("\n" + "="*80)
    print("💾 SAVING RESULTS TO CSV")
    print("="*80)
    
    df_data = []
    for result in all_results:
        row = {
            'prompt': result['prompt'],
            'optimal_steps': result['optimal_steps'],
            'lpips_at_optimal': result['lpips_at_optimal'],
        }
        # Add individual step LPIPS scores
        for steps in Config.STEP_COUNTS:
            row[f'lpips_at_{steps}'] = result['lpips_scores'].get(steps, None)
        df_data.append(row)
    
    df = pd.DataFrame(df_data)
    df.to_csv(Config.CSV_OUTPUT, index=False)
    print(f"✅ Saved to {Config.CSV_OUTPUT}")
    
    # Summary statistics
    print("\n" + "="*80)
    print("📊 COLLECTION SUMMARY")
    print("="*80)
    print(f"Total prompts collected: {len(all_results)}")
    print(f"\nOptimal steps distribution:")
    print(df['optimal_steps'].value_counts().sort_index())
    print(f"\nAverage optimal steps: {df['optimal_steps'].mean():.1f}")
    print(f"Median optimal steps: {df['optimal_steps'].median():.0f}")
    print(f"\nAverage speedup: {((Config.BASELINE_STEPS - df['optimal_steps'].mean()) / Config.BASELINE_STEPS * 100):.1f}%")
    
    # Quality check
    quality_maintained = (df['lpips_at_optimal'] < Config.LPIPS_THRESHOLD).sum()
    print(f"\nQuality maintained (LPIPS < {Config.LPIPS_THRESHOLD}):")
    print(f"   {quality_maintained}/{len(df)} prompts ({quality_maintained/len(df)*100:.1f}%)")
    
    print(f"\n✅ COLLECTION COMPLETE!")
    print(f"   Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")
    
    # Clean up checkpoint
    Path(Config.CHECKPOINT_FILE).unlink(missing_ok=True)
    print("🧹 Checkpoint file removed")

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()