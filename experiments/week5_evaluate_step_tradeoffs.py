"""
Evaluate Step Count Tradeoffs for Binary Predictor
===================================================

Tests different step counts (20, 22, 25) when accelerating to find
the optimal balance between speed and quality.

For each step count, calculates:
- Overall speedup
- Quality maintained (% with LPIPS < threshold)
- Mean LPIPS for accelerated prompts
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Data
    TRAIN_CSV = "data/training_data_012.csv"
    EMBEDDINGS_CACHE = "data/clip_embeddings.pt"
    MODEL_PATH = "models/binary_step_predictor_v1.pth"
    
    # Test different acceleration step counts
    STEP_COUNTS_TO_TEST = [20, 22, 25]
    BASELINE_STEPS = 30
    ACCELERATION_THRESHOLD = 25  # Same as training
    
    # Quality thresholds
    LPIPS_STRICT = 0.10
    LPIPS_RELAXED = 0.12
    
    # Output
    RESULTS_DIR = "results"
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================================
# LOAD MODEL
# ============================================================================

class BinaryStepPredictor(torch.nn.Module):
    """Binary classifier - must match training architecture"""
    
    def __init__(self, input_dim=768, hidden_dims=[256, 128], dropout=0.3):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                torch.nn.Linear(prev_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
                torch.nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        layers.append(torch.nn.Linear(prev_dim, 1))
        layers.append(torch.nn.Sigmoid())
        
        self.network = torch.nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x).squeeze(-1)

# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_with_step_count(model, df, embeddings, accelerated_steps, device):
    """Evaluate using specific step count for acceleration"""
    model.eval()
    
    # Get predictions
    predictions = []
    with torch.no_grad():
        for emb in embeddings:
            prob = model(emb.unsqueeze(0).to(device))
            predictions.append(1 if prob.item() > 0.5 else 0)
    
    predictions = np.array(predictions)
    
    # Calculate which steps we'd actually use
    predicted_steps = np.where(predictions == 1, accelerated_steps, Config.BASELINE_STEPS)
    
    # Speedup
    total_steps_baseline = len(df) * Config.BASELINE_STEPS
    total_steps_predicted = predicted_steps.sum()
    speedup_pct = ((total_steps_baseline - total_steps_predicted) / total_steps_baseline) * 100
    
    # Quality metrics for accelerated prompts
    accelerated_mask = predictions == 1
    num_accelerated = accelerated_mask.sum()
    
    if num_accelerated > 0:
        # Get LPIPS at the accelerated step count
        lpips_col = f'lpips_at_{accelerated_steps}'
        
        if lpips_col in df.columns:
            accelerated_lpips = df.loc[accelerated_mask, lpips_col].values
            mean_lpips = accelerated_lpips.mean()
            
            # Quality maintenance at different thresholds
            quality_strict = (accelerated_lpips < Config.LPIPS_STRICT).sum() / len(accelerated_lpips)
            quality_relaxed = (accelerated_lpips < Config.LPIPS_RELAXED).sum() / len(accelerated_lpips)
        else:
            print(f"⚠️  Warning: Column {lpips_col} not found in data")
            mean_lpips = None
            quality_strict = None
            quality_relaxed = None
    else:
        mean_lpips = None
        quality_strict = None
        quality_relaxed = None
    
    results = {
        'accelerated_steps': accelerated_steps,
        'speedup_pct': speedup_pct,
        'num_accelerated': num_accelerated,
        'pct_accelerated': (num_accelerated / len(df)) * 100,
        'mean_lpips': mean_lpips,
        'quality_strict': quality_strict,
        'quality_relaxed': quality_relaxed,
        'avg_steps': predicted_steps.mean()
    }
    
    return results

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_tradeoff_analysis(results_list, save_path):
    """Plot speed vs quality tradeoff"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    step_counts = [r['accelerated_steps'] for r in results_list]
    speedups = [r['speedup_pct'] for r in results_list]
    quality_strict = [r['quality_strict']*100 if r['quality_strict'] else 0 for r in results_list]
    quality_relaxed = [r['quality_relaxed']*100 if r['quality_relaxed'] else 0 for r in results_list]
    mean_lpips = [r['mean_lpips'] if r['mean_lpips'] else 0 for r in results_list]
    
    # 1. Speedup vs Step Count
    ax1.plot(step_counts, speedups, 'o-', linewidth=3, markersize=10, color='#2E86AB')
    ax1.set_xlabel('Accelerated Step Count', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Overall Speedup (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Speedup vs Step Count', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(step_counts)
    for i, (x, y) in enumerate(zip(step_counts, speedups)):
        ax1.text(x, y+0.3, f'{y:.1f}%', ha='center', fontweight='bold')
    
    # 2. Quality Maintained vs Step Count
    ax2.plot(step_counts, quality_strict, 'o-', linewidth=3, markersize=10, 
             color='#A23B72', label='LPIPS < 0.10 (Strict)')
    ax2.plot(step_counts, quality_relaxed, 's-', linewidth=3, markersize=10, 
             color='#F18F01', label='LPIPS < 0.12 (Relaxed)')
    ax2.axhline(y=90, color='green', linestyle='--', alpha=0.5, label='90% Target')
    ax2.set_xlabel('Accelerated Step Count', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Quality Maintained (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Quality vs Step Count', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(step_counts)
    ax2.set_ylim([0, 105])
    
    # 3. Mean LPIPS vs Step Count
    ax3.plot(step_counts, mean_lpips, 'o-', linewidth=3, markersize=10, color='#C73E1D')
    ax3.axhline(y=Config.LPIPS_STRICT, color='red', linestyle='--', alpha=0.5, label='Strict threshold (0.10)')
    ax3.axhline(y=Config.LPIPS_RELAXED, color='orange', linestyle='--', alpha=0.5, label='Relaxed threshold (0.12)')
    ax3.set_xlabel('Accelerated Step Count', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Mean LPIPS', fontsize=12, fontweight='bold')
    ax3.set_title('Perceptual Distance vs Step Count', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(step_counts)
    for i, (x, y) in enumerate(zip(step_counts, mean_lpips)):
        ax3.text(x, y+0.002, f'{y:.4f}', ha='center', fontweight='bold', fontsize=9)
    
    # 4. Tradeoff Summary (Quality vs Speedup)
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    for i, (speed, qual, step, color) in enumerate(zip(speedups, quality_relaxed, step_counts, colors)):
        ax4.scatter(speed, qual, s=500, alpha=0.7, color=color, edgecolors='black', linewidth=2)
        ax4.text(speed, qual, f'{step}', ha='center', va='center', 
                fontweight='bold', fontsize=14, color='white')
    
    ax4.axhline(y=90, color='green', linestyle='--', alpha=0.5, linewidth=2, label='90% Quality Target')
    ax4.set_xlabel('Speedup (%)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Quality Maintained (%) [LPIPS < 0.12]', fontsize=12, fontweight='bold')
    ax4.set_title('Speed vs Quality Tradeoff', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim([speedups[-1]-1, speedups[0]+1])
    ax4.set_ylim([min(quality_relaxed)-5, 105])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved tradeoff analysis to {save_path}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*80)
    print("STEP COUNT TRADEOFF ANALYSIS")
    print("="*80)
    print()
    
    # Load data
    print("📂 Loading data...")
    df = pd.read_csv(Config.TRAIN_CSV)
    print(f"✅ Loaded {len(df)} prompts")
    
    # Load embeddings
    cache = torch.load(Config.EMBEDDINGS_CACHE)
    embeddings = cache['embeddings']
    print(f"✅ Loaded embeddings: {embeddings.shape}")
    print()
    
    # Load model
    print("🔄 Loading trained model...")
    model = BinaryStepPredictor()
    checkpoint = torch.load(Config.MODEL_PATH, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(Config.DEVICE)
    model.eval()
    print(f"✅ Model loaded (Val Acc: {checkpoint['val_acc']:.3f})")
    print()
    
    # Evaluate at different step counts
    print("🧪 Testing different step counts...")
    print("="*80)
    print()
    
    results_list = []
    
    for steps in Config.STEP_COUNTS_TO_TEST:
        print(f"Testing with {steps} steps for acceleration:")
        print("-" * 60)
        
        results = evaluate_with_step_count(model, df, embeddings, steps, Config.DEVICE)
        results_list.append(results)
        
        print(f"  Speedup: {results['speedup_pct']:.1f}%")
        print(f"  Prompts accelerated: {results['num_accelerated']}/{len(df)} ({results['pct_accelerated']:.1f}%)")
        print(f"  Average steps used: {results['avg_steps']:.1f}")
        
        if results['mean_lpips'] is not None:
            print(f"  Mean LPIPS: {results['mean_lpips']:.4f}")
            print(f"  Quality (LPIPS < 0.10): {results['quality_strict']*100:.1f}%")
            print(f"  Quality (LPIPS < 0.12): {results['quality_relaxed']*100:.1f}%")
        else:
            print(f"  ⚠️  LPIPS data not available for this step count")
        
        print()
    
    # Summary comparison
    print("="*80)
    print("SUMMARY COMPARISON")
    print("="*80)
    print()
    
    print(f"{'Step Count':<12} | {'Speedup':<8} | {'Mean LPIPS':<12} | {'Quality (0.10)':<15} | {'Quality (0.12)':<15}")
    print("-" * 80)
    
    for r in results_list:
        if r['mean_lpips'] is not None:
            print(f"{r['accelerated_steps']:<12} | {r['speedup_pct']:>6.1f}%  | {r['mean_lpips']:>10.4f}  | "
                  f"{r['quality_strict']*100:>13.1f}%  | {r['quality_relaxed']*100:>13.1f}%")
        else:
            print(f"{r['accelerated_steps']:<12} | {r['speedup_pct']:>6.1f}%  | {'N/A':>10}  | "
                  f"{'N/A':>13}  | {'N/A':>13}")
    
    print()
    
    # Recommendation
    print("="*80)
    print("💡 RECOMMENDATION")
    print("="*80)
    print()
    
    # Find best tradeoff (quality > 85% at relaxed, maximize speedup)
    valid_results = [r for r in results_list if r['quality_relaxed'] is not None and r['quality_relaxed'] > 0.85]
    
    if valid_results:
        best = max(valid_results, key=lambda r: r['speedup_pct'])
        
        print(f"✨ Recommended: {best['accelerated_steps']} steps")
        print()
        print(f"   Speedup: {best['speedup_pct']:.1f}%")
        print(f"   Mean LPIPS: {best['mean_lpips']:.4f}")
        print(f"   Quality (strict): {best['quality_strict']*100:.1f}%")
        print(f"   Quality (relaxed): {best['quality_relaxed']*100:.1f}%")
        print()
        print(f"   Rationale: Best speedup while maintaining >85% quality at 0.12 threshold")
    else:
        print("⚠️  No configuration meets the quality threshold")
    
    print()
    
    # Plot tradeoff analysis
    plot_path = Path(Config.RESULTS_DIR) / "step_count_tradeoff_analysis.png"
    plot_tradeoff_analysis(results_list, plot_path)
    print()
    
    # Save detailed results
    results_path = Path(Config.RESULTS_DIR) / "step_count_tradeoff_results.txt"
    with open(results_path, 'w') as f:
        f.write("STEP COUNT TRADEOFF ANALYSIS\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"{'Step Count':<12} | {'Speedup':<8} | {'Mean LPIPS':<12} | {'Quality (0.10)':<15} | {'Quality (0.12)':<15}\n")
        f.write("-" * 80 + "\n")
        
        for r in results_list:
            if r['mean_lpips'] is not None:
                f.write(f"{r['accelerated_steps']:<12} | {r['speedup_pct']:>6.1f}%  | {r['mean_lpips']:>10.4f}  | "
                       f"{r['quality_strict']*100:>13.1f}%  | {r['quality_relaxed']*100:>13.1f}%\n")
        
        f.write("\n")
        
        if valid_results:
            best = max(valid_results, key=lambda r: r['speedup_pct'])
            f.write(f"Recommended: {best['accelerated_steps']} steps\n")
            f.write(f"  Speedup: {best['speedup_pct']:.1f}%\n")
            f.write(f"  Quality: {best['quality_relaxed']*100:.1f}%\n")
    
    print(f"✅ Saved detailed results to {results_path}")
    print()
    
    print("="*80)
    print("✅ ANALYSIS COMPLETE")
    print("="*80)
    print()

if __name__ == "__main__":
    main()