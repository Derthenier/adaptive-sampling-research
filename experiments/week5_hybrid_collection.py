"""
Week 5 Hybrid Data Collection - Reuses Week 1 Results
=======================================================

Intelligently reuses Week 1 data where possible:
- Extracts 10 existing prompts from week1_results/
- Copies existing images for steps [15, 20, 25, 30]
- Generates missing steps [18, 22] for Week 1 prompts
- Computes LPIPS for all (Week 1 didn't have this metric)
- Adds 60 new diverse prompts with full generation

Hardware: RTX 5070 Ti (16GB VRAM)
Model: Stable Diffusion 1.5
Output: CSV with prompt, optimal_steps, LPIPS metrics
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
import shutil
import gc

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Model settings
    MODEL_ID = "runwayml/stable-diffusion-v1-5"
    DEVICE = "cuda"
    DTYPE = torch.float16
    
    # Step counts to test (SD 1.5)
    STEP_COUNTS = [15, 18, 20, 22, 25, 30]
    BASELINE_STEPS = 30
    
    # Week 1 integration
    WEEK1_DIR = "results/week1_results"
    WEEK1_STEPS_AVAILABLE = [10, 15, 20, 25, 30, 40, 50]  # From Week 1
    WEEK1_STEPS_TO_REUSE = [15, 20, 25, 30]  # Overlap with Week 5 needs
    WEEK1_STEPS_TO_GENERATE = [18, 22]  # Missing from Week 1
    
    # Generation settings
    GUIDANCE_SCALE = 7.5
    WIDTH = 512
    HEIGHT = 512
    SEED = 42
    
    # Quality thresholds
    LPIPS_THRESHOLD = 0.10
    
    # Paths
    NEW_PROMPTS_FILE = "data/training_prompts_new_60.txt"
    OUTPUT_DIR = "week5_training_data"
    CSV_OUTPUT = "data/training_data.csv"
    CHECKPOINT_FILE = "data/collection_checkpoint.json"
    
    # Performance
    ENABLE_ATTENTION_SLICING = True
    CLEAR_CACHE_FREQUENCY = 5

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

def extract_week1_prompts():
    """Extract prompts from Week 1 results"""
    week1_path = Path(Config.WEEK1_DIR)
    
    if not week1_path.exists():
        print(f"⚠️  Week 1 directory not found: {Config.WEEK1_DIR}")
        return []
    
    prompts = []
    prompt_dirs = [d for d in week1_path.iterdir() if d.is_dir()]
    
    for prompt_dir in sorted(prompt_dirs):
        # Read the results.json to get the actual prompt text
        results_file = prompt_dir / "results.json"
        if results_file.exists():
            with open(results_file, 'r') as f:
                data = json.load(f)
                prompt_text = data.get('prompt', '')
                if prompt_text:
                    prompts.append({
                        'text': prompt_text,
                        'dir': prompt_dir.name,
                        'path': prompt_dir
                    })
    
    print(f"✅ Extracted {len(prompts)} prompts from Week 1")
    return prompts

def load_new_prompts():
    """Load 60 new prompts from file"""
    prompt_file = Path(Config.NEW_PROMPTS_FILE)
    if not prompt_file.exists():
        print(f"⚠️  New prompts file not found: {Config.NEW_PROMPTS_FILE}")
        print("    Create this file with 60 new prompts (one per line)")
        return []
    
    with open(prompt_file, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]
    
    print(f"✅ Loaded {len(prompts)} new prompts from {Config.NEW_PROMPTS_FILE}")
    return prompts

def copy_week1_image(week1_prompt_info, step_count, output_idx):
    """Copy existing Week 1 image to Week 5 directory"""
    src_path = week1_prompt_info['path'] / f"steps_{step_count}.png"
    dst_path = Path(f"{Config.OUTPUT_DIR}/steps_{step_count}/prompt_{output_idx:03d}.png")
    
    if src_path.exists():
        shutil.copy2(src_path, dst_path)
        return True
    return False

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
        safety_checker=None
    )
    pipe = pipe.to(Config.DEVICE)
    
    if Config.ENABLE_ATTENTION_SLICING:
        pipe.enable_attention_slicing()
        print("   ✅ Attention slicing enabled")
    
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
    def pil_to_tensor(img):
        img_array = np.array(img).astype(np.float32) / 255.0
        img_array = img_array * 2.0 - 1.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
        return img_tensor.to(Config.DEVICE)
    
    tensor1 = pil_to_tensor(img1)
    tensor2 = pil_to_tensor(img2)
    
    with torch.no_grad():
        distance = lpips_model(tensor1, tensor2)
    
    return distance.item()

def find_optimal_steps(lpips_scores):
    """Find minimum steps that maintain quality below threshold"""
    sorted_steps = sorted(lpips_scores.keys())
    
    for steps in sorted_steps:
        if steps == Config.BASELINE_STEPS:
            continue
        
        if lpips_scores[steps] < Config.LPIPS_THRESHOLD:
            return steps
    
    return Config.BASELINE_STEPS

# ============================================================================
# DATA COLLECTION FUNCTIONS
# ============================================================================

def process_week1_prompt(pipe, lpips_model, week1_info, prompt_idx, total_prompts):
    """
    Process a Week 1 prompt:
    - Copy existing images for steps [15, 20, 25, 30]
    - Generate missing steps [18, 22]
    - Compute LPIPS for all steps
    """
    prompt = week1_info['text']
    
    print(f"\n{'='*80}")
    print(f"Prompt {prompt_idx + 1}/{total_prompts} [WEEK 1 REUSE]")
    print(f"{'='*80}")
    print(f"📝 {prompt}")
    print(f"📂 Week 1 folder: {week1_info['dir']}")
    print()
    
    results = {
        'prompt': prompt,
        'source': 'week1',
        'lpips_scores': {},
        'images': {}
    }
    
    # Step 1: Copy/load existing images
    print("📋 Reusing Week 1 images:")
    for steps in Config.WEEK1_STEPS_TO_REUSE:
        if copy_week1_image(week1_info, steps, prompt_idx):
            img_path = Path(f"{Config.OUTPUT_DIR}/steps_{steps}/prompt_{prompt_idx:03d}.png")
            results['images'][steps] = Image.open(img_path)
            print(f"   ✅ Copied step {steps}")
        else:
            print(f"   ⚠️  Missing step {steps} - will regenerate")
    
    # Step 2: Generate baseline (30 steps) if not already copied
    if 30 not in results['images']:
        print(f"\n🎨 Generating baseline ({Config.BASELINE_STEPS} steps)...")
        baseline_img = generate_image(pipe, prompt, Config.BASELINE_STEPS, Config.SEED)
        baseline_path = f"{Config.OUTPUT_DIR}/steps_{Config.BASELINE_STEPS}/prompt_{prompt_idx:03d}.png"
        baseline_img.save(baseline_path)
        results['images'][Config.BASELINE_STEPS] = baseline_img
        print(f"   ✅ Saved to {baseline_path}")
    else:
        results['images'][Config.BASELINE_STEPS] = Image.open(
            f"{Config.OUTPUT_DIR}/steps_{Config.BASELINE_STEPS}/prompt_{prompt_idx:03d}.png"
        )
    
    baseline_img = results['images'][Config.BASELINE_STEPS]
    results['lpips_scores'][Config.BASELINE_STEPS] = 0.0
    
    # Step 3: Generate missing steps
    print(f"\n🎨 Generating missing steps:")
    for steps in Config.WEEK1_STEPS_TO_GENERATE:
        print(f"   Generating {steps} steps...")
        img = generate_image(pipe, prompt, steps, Config.SEED)
        
        img_path = f"{Config.OUTPUT_DIR}/steps_{steps}/prompt_{prompt_idx:03d}.png"
        img.save(img_path)
        results['images'][steps] = img
        
        lpips_dist = compute_lpips(lpips_model, img, baseline_img)
        results['lpips_scores'][steps] = lpips_dist
        
        status = "✅ PASS" if lpips_dist < Config.LPIPS_THRESHOLD else "⚠️  FAIL"
        print(f"   LPIPS: {lpips_dist:.4f} {status}")
    
    # Step 4: Compute LPIPS for all existing images
    print(f"\n📊 Computing LPIPS for reused images:")
    for steps in Config.WEEK1_STEPS_TO_REUSE:
        if steps == Config.BASELINE_STEPS:
            continue
        if steps in results['images']:
            lpips_dist = compute_lpips(lpips_model, results['images'][steps], baseline_img)
            results['lpips_scores'][steps] = lpips_dist
            
            status = "✅ PASS" if lpips_dist < Config.LPIPS_THRESHOLD else "⚠️  FAIL"
            print(f"   Step {steps}: LPIPS {lpips_dist:.4f} {status}")
    
    # Find optimal steps
    optimal = find_optimal_steps(results['lpips_scores'])
    results['optimal_steps'] = optimal
    results['lpips_at_optimal'] = results['lpips_scores'][optimal]
    
    speedup_pct = ((Config.BASELINE_STEPS - optimal) / Config.BASELINE_STEPS) * 100
    
    print(f"\n{'='*80}")
    print(f"📊 RESULTS:")
    print(f"   Optimal steps: {optimal}/{Config.BASELINE_STEPS}")
    print(f"   LPIPS at optimal: {results['lpips_at_optimal']:.4f}")
    print(f"   Speedup: {speedup_pct:.1f}%")
    print(f"{'='*80}\n")
    
    return results

def process_new_prompt(pipe, lpips_model, prompt, prompt_idx, total_prompts):
    """Generate all step counts for a new prompt"""
    print(f"\n{'='*80}")
    print(f"Prompt {prompt_idx + 1}/{total_prompts} [NEW]")
    print(f"{'='*80}")
    print(f"📝 {prompt[:100]}..." if len(prompt) > 100 else f"📝 {prompt}")
    print()
    
    results = {
        'prompt': prompt,
        'source': 'new',
        'lpips_scores': {},
        'images': {}
    }
    
    # Generate baseline
    print(f"🎨 Generating baseline ({Config.BASELINE_STEPS} steps)...")
    baseline_img = generate_image(pipe, prompt, Config.BASELINE_STEPS, Config.SEED)
    results['images'][Config.BASELINE_STEPS] = baseline_img
    results['lpips_scores'][Config.BASELINE_STEPS] = 0.0
    
    baseline_path = f"{Config.OUTPUT_DIR}/steps_{Config.BASELINE_STEPS}/prompt_{prompt_idx:03d}.png"
    baseline_img.save(baseline_path)
    print(f"   ✅ Saved to {baseline_path}")
    
    # Generate at other step counts
    for steps in Config.STEP_COUNTS:
        if steps == Config.BASELINE_STEPS:
            continue
        
        print(f"\n🎨 Generating with {steps} steps...")
        img = generate_image(pipe, prompt, steps, Config.SEED)
        results['images'][steps] = img
        
        lpips_dist = compute_lpips(lpips_model, img, baseline_img)
        results['lpips_scores'][steps] = lpips_dist
        
        img_path = f"{Config.OUTPUT_DIR}/steps_{steps}/prompt_{prompt_idx:03d}.png"
        img.save(img_path)
        
        status = "✅ PASS" if lpips_dist < Config.LPIPS_THRESHOLD else "⚠️  FAIL"
        print(f"   LPIPS: {lpips_dist:.4f} {status}")
        print(f"   Saved to {img_path}")
    
    # Find optimal steps
    optimal = find_optimal_steps(results['lpips_scores'])
    results['optimal_steps'] = optimal
    results['lpips_at_optimal'] = results['lpips_scores'][optimal]
    
    speedup_pct = ((Config.BASELINE_STEPS - optimal) / Config.BASELINE_STEPS) * 100
    
    print(f"\n{'='*80}")
    print(f"📊 RESULTS:")
    print(f"   Optimal steps: {optimal}/{Config.BASELINE_STEPS}")
    print(f"   LPIPS at optimal: {results['lpips_at_optimal']:.4f}")
    print(f"   Speedup: {speedup_pct:.1f}%")
    print(f"{'='*80}\n")
    
    return results

# ============================================================================
# MAIN COLLECTION PIPELINE
# ============================================================================

def main():
    """Main data collection pipeline"""
    print("\n" + "="*80)
    print("WEEK 5 HYBRID DATA COLLECTION - SD 1.5 ADAPTIVE SAMPLING")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Setup
    setup_directories()
    week1_prompts = extract_week1_prompts()
    new_prompts = load_new_prompts()
    
    total_prompts = len(week1_prompts) + len(new_prompts)
    
    if total_prompts == 0:
        print("❌ No prompts found! Please check your files.")
        return
    
    checkpoint = load_checkpoint()
    
    # Initialize models
    pipe = initialize_pipeline()
    lpips_model = initialize_lpips()
    
    # Track results
    all_results = checkpoint.get('results', [])
    completed_prompts = set(checkpoint.get('completed_prompts', []))
    
    print(f"📊 COLLECTION PLAN:")
    print(f"   Week 1 prompts (reuse): {len(week1_prompts)}")
    print(f"   New prompts: {len(new_prompts)}")
    print(f"   Total prompts: {total_prompts}")
    print(f"   Already completed: {len(completed_prompts)}")
    print(f"   Remaining: {total_prompts - len(completed_prompts)}")
    print()
    
    # Estimate time
    week1_remaining = len([i for i in range(len(week1_prompts)) if i not in completed_prompts])
    new_remaining = len([i for i in range(len(week1_prompts), total_prompts) if i not in completed_prompts])
    
    week1_time = week1_remaining * 2 * 5  # 2 images × 5s each
    new_time = new_remaining * 6 * 5  # 6 images × 5s each
    total_time = (week1_time + new_time) // 60
    
    print(f"⏱️  Estimated time: {total_time} minutes")
    print(f"   (Week 1 prompts: ~{week1_time // 60} min, New prompts: ~{new_time // 60} min)")
    print()
    
    input("Press ENTER to start collection...")
    print()
    
    # Main loop
    try:
        # Process Week 1 prompts
        for idx, week1_info in enumerate(week1_prompts):
            if idx in completed_prompts:
                print(f"⏭️  Skipping prompt {idx + 1} (already completed)")
                continue
            
            result = process_week1_prompt(pipe, lpips_model, week1_info, idx, total_prompts)
            
            all_results.append({
                'prompt_idx': idx,
                'prompt': result['prompt'],
                'source': 'week1',
                'optimal_steps': result['optimal_steps'],
                'lpips_at_optimal': result['lpips_at_optimal'],
                'lpips_scores': result['lpips_scores']
            })
            completed_prompts.add(idx)
            
            save_checkpoint({
                'completed_prompts': list(completed_prompts),
                'results': all_results
            })
            
            if (idx + 1) % Config.CLEAR_CACHE_FREQUENCY == 0:
                gc.collect()
                torch.cuda.empty_cache()
                print("\n🧹 Cleared CUDA cache\n")
        
        # Process new prompts
        for idx, prompt in enumerate(new_prompts):
            global_idx = len(week1_prompts) + idx
            
            if global_idx in completed_prompts:
                print(f"⏭️  Skipping prompt {global_idx + 1} (already completed)")
                continue
            
            result = process_new_prompt(pipe, lpips_model, prompt, global_idx, total_prompts)
            
            all_results.append({
                'prompt_idx': global_idx,
                'prompt': result['prompt'],
                'source': 'new',
                'optimal_steps': result['optimal_steps'],
                'lpips_at_optimal': result['lpips_at_optimal'],
                'lpips_scores': result['lpips_scores']
            })
            completed_prompts.add(global_idx)
            
            save_checkpoint({
                'completed_prompts': list(completed_prompts),
                'results': all_results
            })
            
            if (global_idx + 1) % Config.CLEAR_CACHE_FREQUENCY == 0:
                gc.collect()
                torch.cuda.empty_cache()
                print("\n🧹 Cleared CUDA cache\n")
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Collection interrupted by user")
        print("💾 Progress saved to checkpoint file")
        return
    
    except Exception as e:
        print(f"\n\n❌ Error occurred: {e}")
        print("💾 Progress saved to checkpoint file")
        raise
    
    # Save final results
    print("\n" + "="*80)
    print("💾 SAVING RESULTS TO CSV")
    print("="*80)
    
    df_data = []
    for result in all_results:
        row = {
            'prompt': result['prompt'],
            'source': result.get('source', 'unknown'),
            'optimal_steps': result['optimal_steps'],
            'lpips_at_optimal': result['lpips_at_optimal'],
        }
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
    print(f"   From Week 1 (reused): {len([r for r in all_results if r.get('source') == 'week1'])}")
    print(f"   New prompts: {len([r for r in all_results if r.get('source') == 'new'])}")
    
    print(f"\nOptimal steps distribution:")
    print(df['optimal_steps'].value_counts().sort_index())
    print(f"\nAverage optimal steps: {df['optimal_steps'].mean():.1f}")
    print(f"Median optimal steps: {df['optimal_steps'].median():.0f}")
    print(f"\nAverage speedup: {((Config.BASELINE_STEPS - df['optimal_steps'].mean()) / Config.BASELINE_STEPS * 100):.1f}%")
    
    quality_maintained = (df['lpips_at_optimal'] < Config.LPIPS_THRESHOLD).sum()
    print(f"\nQuality maintained (LPIPS < {Config.LPIPS_THRESHOLD}):")
    print(f"   {quality_maintained}/{len(df)} prompts ({quality_maintained/len(df)*100:.1f}%)")
    
    print(f"\n✅ COLLECTION COMPLETE!")
    print(f"   Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")
    
    Path(Config.CHECKPOINT_FILE).unlink(missing_ok=True)
    print("🧹 Checkpoint file removed")

if __name__ == "__main__":
    main()