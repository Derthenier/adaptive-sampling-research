"""
Re-label Training Data with 0.12 LPIPS Threshold
=================================================

Creates training_data_012.csv with relaxed quality threshold.
Compares results between 0.10 and 0.12 thresholds.

Usage:
    python experiments/relabel_threshold.py
"""

import pandas as pd
from pathlib import Path

# Configuration
INPUT_CSV = "data/training_data.csv"
OUTPUT_CSV_010 = "data/training_data_010.csv"  # Keep original as 0.10
OUTPUT_CSV_012 = "data/training_data_012.csv"  # New 0.12 version

THRESHOLD_010 = 0.10
THRESHOLD_012 = 0.12
STEP_COUNTS = [15, 18, 20, 22, 25, 30]
BASELINE_STEPS = 30

def find_optimal_steps(row, threshold):
    """Find minimum steps that meet quality threshold"""
    for steps in STEP_COUNTS:
        if steps == BASELINE_STEPS:
            continue
        
        lpips_col = f'lpips_at_{steps}'
        if lpips_col in row.index and pd.notna(row[lpips_col]):
            if row[lpips_col] < threshold:
                return steps
    
    return BASELINE_STEPS

def main():
    print("="*80)
    print("RE-LABELING DATASET WITH MULTIPLE THRESHOLDS")
    print("="*80)
    print()
    
    # Load original data
    print(f"📂 Loading {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    print(f"✅ Loaded {len(df)} prompts")
    print()
    
    # Create 0.10 threshold version (rename original)
    df_010 = df.copy()
    
    # Create 0.12 threshold version
    df_012 = df.copy()
    
    print("🔄 Re-labeling with 0.12 threshold...")
    df_012['optimal_steps'] = df_012.apply(lambda row: find_optimal_steps(row, THRESHOLD_012), axis=1)
    df_012['lpips_at_optimal'] = df_012.apply(
        lambda row: row[f'lpips_at_{int(row["optimal_steps"])}'] if f'lpips_at_{int(row["optimal_steps"])}' in row.index else 0.0,
        axis=1
    )
    print("✅ Re-labeling complete")
    print()
    
    # Save both versions
    print("💾 Saving datasets...")
    df_010.to_csv(OUTPUT_CSV_010, index=False)
    print(f"   ✅ Saved {OUTPUT_CSV_010}")
    
    df_012.to_csv(OUTPUT_CSV_012, index=False)
    print(f"   ✅ Saved {OUTPUT_CSV_012}")
    print()
    
    # Comparison analysis
    print("="*80)
    print("THRESHOLD COMPARISON")
    print("="*80)
    print()
    
    print("📊 THRESHOLD 0.10 (Conservative):")
    print("-" * 40)
    print(df_010['optimal_steps'].value_counts().sort_index())
    speedup_010 = ((BASELINE_STEPS - df_010['optimal_steps'].mean()) / BASELINE_STEPS) * 100
    acceleratable_010 = (df_010['optimal_steps'] < BASELINE_STEPS).sum()
    print(f"\nMean optimal steps: {df_010['optimal_steps'].mean():.1f}")
    print(f"Can accelerate: {acceleratable_010}/{len(df_010)} ({acceleratable_010/len(df_010)*100:.1f}%)")
    print(f"Average speedup: {speedup_010:.1f}%")
    print(f"Mean LPIPS: {df_010['lpips_at_optimal'].mean():.4f}")
    print()
    
    print("📊 THRESHOLD 0.12 (Balanced):")
    print("-" * 40)
    print(df_012['optimal_steps'].value_counts().sort_index())
    speedup_012 = ((BASELINE_STEPS - df_012['optimal_steps'].mean()) / BASELINE_STEPS) * 100
    acceleratable_012 = (df_012['optimal_steps'] < BASELINE_STEPS).sum()
    print(f"\nMean optimal steps: {df_012['optimal_steps'].mean():.1f}")
    print(f"Can accelerate: {acceleratable_012}/{len(df_012)} ({acceleratable_012/len(df_012)*100:.1f}%)")
    print(f"Average speedup: {speedup_012:.1f}%")
    print(f"Mean LPIPS: {df_012['lpips_at_optimal'].mean():.4f}")
    print()
    
    # Difference analysis
    print("📈 IMPROVEMENT FROM 0.10 → 0.12:")
    print("-" * 40)
    print(f"Additional acceleratable prompts: +{acceleratable_012 - acceleratable_010}")
    print(f"Speedup improvement: +{speedup_012 - speedup_010:.1f}% (absolute)")
    print(f"Relative improvement: +{((speedup_012 - speedup_010) / speedup_010 * 100):.1f}%")
    
    # Which prompts changed?
    changed_prompts = df_010['optimal_steps'] != df_012['optimal_steps']
    num_changed = changed_prompts.sum()
    print(f"\nPrompts with changed optimal steps: {num_changed}")
    
    if num_changed > 0:
        print("\nExamples of changes:")
        changes_df = pd.DataFrame({
            'prompt': df_010.loc[changed_prompts, 'prompt'].str[:50] + '...',
            'steps_010': df_010.loc[changed_prompts, 'optimal_steps'].astype(int),
            'steps_012': df_012.loc[changed_prompts, 'optimal_steps'].astype(int),
            'lpips_010': df_010.loc[changed_prompts, 'lpips_at_optimal'].round(4),
            'lpips_012': df_012.loc[changed_prompts, 'lpips_at_optimal'].round(4)
        })
        print(changes_df.head(10).to_string(index=False))
    
    print()
    print("="*80)
    print("✅ RE-LABELING COMPLETE")
    print("="*80)
    print()
    print("Next steps:")
    print("1. Train predictor on training_data_012.csv (better speedup)")
    print("2. Validate against both thresholds")
    print("3. Users can choose conservative (0.10) or balanced (0.12) mode")
    print()

if __name__ == "__main__":
    main()