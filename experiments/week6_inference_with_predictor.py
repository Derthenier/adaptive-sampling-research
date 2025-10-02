"""
Inference with Binary Step Predictor
=====================================

Generate images using the trained predictor to determine optimal step count.

Usage:
    python experiments/inference_with_predictor.py --prompt "a cat sitting"
    python experiments/inference_with_predictor.py --batch test_prompts.txt
    python experiments/inference_with_predictor.py --prompt "a cat" --compare

Features:
- Predicts optimal steps (20 or 30) before generation
- Shows prediction confidence
- Optionally generates baseline (30 steps) for comparison
- Computes LPIPS and timing metrics
- Saves images with clear naming
"""

import torch
import torch.nn as nn
from diffusers import StableDiffusionPipeline
from transformers import CLIPTokenizer, CLIPTextModel
from PIL import Image
import lpips
import numpy as np
from pathlib import Path
from datetime import datetime
import argparse
import time

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Model paths
    MODEL_PATH = "models/binary_step_predictor_v1.pth"
    SD_MODEL = "runwayml/stable-diffusion-v1-5"
    CLIP_MODEL = "openai/clip-vit-large-patch14"
    
    # Generation settings
    ACCELERATED_STEPS = 20
    BASELINE_STEPS = 30
    GUIDANCE_SCALE = 7.5
    WIDTH = 512
    HEIGHT = 512
    SEED = 42  # For reproducibility
    
    # Output
    OUTPUT_DIR = "results/week6_inference_results"
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.float16

# ============================================================================
# MODEL ARCHITECTURE (must match training)
# ============================================================================

class BinaryStepPredictor(nn.Module):
    """Binary classifier for step prediction"""
    
    def __init__(self, input_dim=768, hidden_dims=[256, 128], dropout=0.3):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x).squeeze(-1)

# ============================================================================
# PREDICTOR PIPELINE
# ============================================================================

class StepPredictorPipeline:
    """Complete pipeline for prediction and generation"""
    
    def __init__(self):
        print("🔄 Initializing Step Predictor Pipeline...")
        print()
        
        # Load predictor
        print("   Loading binary predictor...")
        self.predictor = BinaryStepPredictor()
        checkpoint = torch.load(Config.MODEL_PATH, weights_only=False)
        self.predictor.load_state_dict(checkpoint['model_state_dict'])
        self.predictor = self.predictor.to(Config.DEVICE)
        self.predictor.eval()
        print(f"   ✅ Predictor loaded (Val Acc: {checkpoint['val_acc']:.1%})")
        
        # Load CLIP for embeddings
        print("   Loading CLIP text encoder...")
        self.tokenizer = CLIPTokenizer.from_pretrained(Config.CLIP_MODEL)
        self.text_encoder = CLIPTextModel.from_pretrained(Config.CLIP_MODEL)
        self.text_encoder = self.text_encoder.to(Config.DEVICE)
        self.text_encoder.eval()
        print("   ✅ CLIP loaded")
        
        # Load SD pipeline
        print("   Loading Stable Diffusion 1.5...")
        self.sd_pipe = StableDiffusionPipeline.from_pretrained(
            Config.SD_MODEL,
            torch_dtype=Config.DTYPE,
            safety_checker=None
        )
        self.sd_pipe = self.sd_pipe.to(Config.DEVICE)
        self.sd_pipe.enable_attention_slicing()
        print("   ✅ SD 1.5 loaded")
        
        # Load LPIPS for quality measurement
        print("   Loading LPIPS metric...")
        self.lpips_model = lpips.LPIPS(net='alex').to(Config.DEVICE)
        print("   ✅ LPIPS loaded")
        
        print()
        print("✅ Pipeline ready!")
        print()
    
    def extract_embedding(self, prompt):
        """Extract CLIP embedding from prompt"""
        with torch.no_grad():
            inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="pt"
            )
            inputs = {k: v.to(Config.DEVICE) for k, v in inputs.items()}
            outputs = self.text_encoder(**inputs)
            embedding = outputs.pooler_output
        return embedding
    
    def predict_steps(self, prompt):
        """Predict optimal steps for prompt"""
        # Get embedding
        embedding = self.extract_embedding(prompt)
        
        # Predict
        with torch.no_grad():
            probability = self.predictor(embedding).item()
        
        # Decision
        can_accelerate = probability > 0.5
        predicted_steps = Config.ACCELERATED_STEPS if can_accelerate else Config.BASELINE_STEPS
        
        return predicted_steps, probability, can_accelerate
    
    def generate_image(self, prompt, num_steps):
        """Generate image with specified steps"""
        generator = torch.Generator(device=Config.DEVICE).manual_seed(Config.SEED)
        
        start_time = time.time()
        
        image = self.sd_pipe(
            prompt=prompt,
            num_inference_steps=num_steps,
            guidance_scale=Config.GUIDANCE_SCALE,
            width=Config.WIDTH,
            height=Config.HEIGHT,
            generator=generator
        ).images[0]
        
        generation_time = time.time() - start_time
        
        return image, generation_time
    
    def compute_lpips(self, img1, img2):
        """Compute LPIPS between two images"""
        def pil_to_tensor(img):
            img_array = np.array(img).astype(np.float32) / 255.0
            img_array = img_array * 2.0 - 1.0
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
            return img_tensor.to(Config.DEVICE)
        
        tensor1 = pil_to_tensor(img1)
        tensor2 = pil_to_tensor(img2)
        
        with torch.no_grad():
            distance = self.lpips_model(tensor1, tensor2)
        
        return distance.item()
    
    def generate_with_predictor(self, prompt, compare=False):
        """
        Full pipeline: predict steps, generate, optionally compare
        
        Args:
            prompt: Text prompt
            compare: If True, also generate baseline (30 steps) for comparison
        
        Returns:
            dict with results
        """
        print(f"📝 Prompt: '{prompt}'")
        print()
        
        # Predict optimal steps
        predicted_steps, probability, can_accelerate = self.predict_steps(prompt)
        
        confidence_pct = probability * 100 if can_accelerate else (1 - probability) * 100
        
        print(f"🧠 Prediction:")
        print(f"   Can accelerate: {'YES ✅' if can_accelerate else 'NO ❌'}")
        print(f"   Predicted steps: {predicted_steps}")
        print(f"   Confidence: {confidence_pct:.1f}%")
        print()
        
        # Generate with predicted steps
        print(f"🎨 Generating with {predicted_steps} steps...")
        pred_image, pred_time = self.generate_image(prompt, predicted_steps)
        print(f"   ✅ Generated in {pred_time:.2f}s")
        print()
        
        results = {
            'prompt': prompt,
            'predicted_steps': predicted_steps,
            'can_accelerate': can_accelerate,
            'confidence': confidence_pct,
            'pred_image': pred_image,
            'pred_time': pred_time,
        }
        
        # Optionally generate baseline for comparison
        if compare:
            print(f"🎨 Generating baseline ({Config.BASELINE_STEPS} steps)...")
            baseline_image, baseline_time = self.generate_image(prompt, Config.BASELINE_STEPS)
            print(f"   ✅ Generated in {baseline_time:.2f}s")
            print()
            
            # Compute quality difference
            if predicted_steps != Config.BASELINE_STEPS:
                lpips_dist = self.compute_lpips(pred_image, baseline_image)
                speedup = ((baseline_time - pred_time) / baseline_time) * 100
                
                print(f"📊 Comparison:")
                print(f"   Time saved: {baseline_time - pred_time:.2f}s ({speedup:.1f}% faster)")
                print(f"   LPIPS distance: {lpips_dist:.4f}")
                print(f"   Quality: {'✅ Excellent' if lpips_dist < 0.10 else '👍 Good' if lpips_dist < 0.12 else '⚠️  Noticeable'}")
                print()
                
                results['baseline_image'] = baseline_image
                results['baseline_time'] = baseline_time
                results['lpips'] = lpips_dist
                results['speedup_pct'] = speedup
            else:
                print(f"📊 Predictor chose baseline ({Config.BASELINE_STEPS} steps) - no comparison needed")
                print()
                results['baseline_image'] = None
                results['baseline_time'] = None
                results['lpips'] = 0.0
                results['speedup_pct'] = 0.0
        
        return results

# ============================================================================
# VISUALIZATION
# ============================================================================

def save_results(results, output_dir):
    """Save images and create comparison visualization"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Create safe filename from prompt
    safe_prompt = "".join(c if c.isalnum() or c in ' -_' else '' for c in results['prompt'])
    safe_prompt = safe_prompt[:50].strip().replace(' ', '_')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save predicted image
    pred_filename = f"{timestamp}_{safe_prompt}_pred{results['predicted_steps']}.png"
    pred_path = output_path / pred_filename
    results['pred_image'].save(pred_path)
    print(f"💾 Saved: {pred_path}")
    
    # Save baseline if exists
    if results.get('baseline_image'):
        baseline_filename = f"{timestamp}_{safe_prompt}_baseline30.png"
        baseline_path = output_path / baseline_filename
        results['baseline_image'].save(baseline_path)
        print(f"💾 Saved: {baseline_path}")
        
        # Create side-by-side comparison
        comparison_filename = f"{timestamp}_{safe_prompt}_comparison.png"
        comparison_path = output_path / comparison_filename
        create_comparison_image(results, comparison_path)
        print(f"💾 Saved comparison: {comparison_path}")
    
    print()

def create_comparison_image(results, save_path):
    """Create side-by-side comparison with metrics"""
    from PIL import ImageDraw, ImageFont
    
    pred_img = results['pred_image']
    baseline_img = results['baseline_image']
    
    # Create canvas
    width = pred_img.width * 2 + 60  # Space between images
    height = pred_img.height + 120  # Space for text
    
    canvas = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(canvas)
    
    # Try to load a font, fallback to default
    try:
        font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except:
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    # Paste images
    canvas.paste(pred_img, (20, 80))
    canvas.paste(baseline_img, (pred_img.width + 40, 80))
    
    # Add labels
    draw.text((20, 20), f"Predictor ({results['predicted_steps']} steps)", fill='black', font=font_large)
    draw.text((20, 45), f"Time: {results['pred_time']:.2f}s", fill='black', font=font_small)
    
    draw.text((pred_img.width + 40, 20), f"Baseline (30 steps)", fill='black', font=font_large)
    draw.text((pred_img.width + 40, 45), f"Time: {results['baseline_time']:.2f}s", fill='black', font=font_small)
    
    # Add metrics at bottom
    y_offset = pred_img.height + 90
    speedup_text = f"Speedup: {results['speedup_pct']:.1f}% faster"
    lpips_text = f"LPIPS: {results['lpips']:.4f}"
    quality_text = f"Quality: {'Excellent' if results['lpips'] < 0.10 else 'Good' if results['lpips'] < 0.12 else 'Noticeable diff'}"
    
    draw.text((20, y_offset), f"{speedup_text} | {lpips_text} | {quality_text}", fill='black', font=font_small)
    
    canvas.save(save_path)

# ============================================================================
# MAIN CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate images with step predictor')
    parser.add_argument('--prompt', type=str, help='Single prompt to generate')
    parser.add_argument('--batch', type=str, help='File with prompts (one per line)')
    parser.add_argument('--compare', action='store_true', help='Generate baseline for comparison')
    parser.add_argument('--output', type=str, default=Config.OUTPUT_DIR, help='Output directory')
    
    args = parser.parse_args()
    
    if not args.prompt and not args.batch:
        parser.error('Either --prompt or --batch must be specified')
    
    print("\n" + "="*80)
    print("INFERENCE WITH BINARY STEP PREDICTOR")
    print("="*80)
    print()
    
    # Initialize pipeline
    pipeline = StepPredictorPipeline()
    
    # Prepare prompts
    if args.prompt:
        prompts = [args.prompt]
    else:
        with open(args.batch, 'r') as f:
            prompts = [line.strip() for line in f if line.strip()]
        print(f"📂 Loaded {len(prompts)} prompts from {args.batch}")
        print()
    
    # Generate
    print("="*80)
    print()
    
    all_results = []
    total_pred_time = 0
    total_baseline_time = 0
    
    for i, prompt in enumerate(prompts, 1):
        print(f"[{i}/{len(prompts)}]")
        print()
        
        results = pipeline.generate_with_predictor(prompt, compare=args.compare)
        all_results.append(results)
        
        total_pred_time += results['pred_time']
        if args.compare and results.get('baseline_time'):
            total_baseline_time += results['baseline_time']
        
        # Save results
        save_results(results, args.output)
        
        print("="*80)
        print()
    
    # Summary
    print("📊 SUMMARY")
    print("="*80)
    print()
    print(f"Total prompts: {len(prompts)}")
    print(f"Accelerated: {sum(1 for r in all_results if r['can_accelerate'])} ({sum(1 for r in all_results if r['can_accelerate'])/len(prompts)*100:.1f}%)")
    print(f"Full quality: {sum(1 for r in all_results if not r['can_accelerate'])} ({sum(1 for r in all_results if not r['can_accelerate'])/len(prompts)*100:.1f}%)")
    print()
    
    if args.compare and total_baseline_time > 0:
        time_saved = total_baseline_time - total_pred_time
        speedup = (time_saved / total_baseline_time) * 100
        
        print(f"Total time (predictor): {total_pred_time:.1f}s")
        print(f"Total time (baseline): {total_baseline_time:.1f}s")
        print(f"Time saved: {time_saved:.1f}s ({speedup:.1f}% faster)")
        
        if len([r for r in all_results if r.get('lpips')]) > 0:
            avg_lpips = np.mean([r['lpips'] for r in all_results if r.get('lpips') and r['lpips'] > 0])
            print(f"Average LPIPS: {avg_lpips:.4f}")
    
    print()
    print(f"✅ Results saved to: {args.output}/")
    print()
    print("="*80)

if __name__ == "__main__":
    main()