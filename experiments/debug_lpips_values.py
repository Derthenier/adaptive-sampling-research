#!/usr/bin/env python3
"""
Debug: Analyze LPIPS values from Phase A to understand why nothing stopped
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def analyze_detection_logs():
    """Examine actual LPIPS values measured during generation"""
    
    results_dir = Path('week3_phase_a_results')
    
    print("🔍 Analyzing LPIPS Detection Logs")
    print("="*60)
    
    all_lpips_values = []
    categories_data = {}
    
    # Load all results
    with open(results_dir / 'all_results.json') as f:
        all_results = json.load(f)
    
    for category, prompts in all_results.items():
        category_lpips = []
        
        print(f"\n📂 {category.upper()}")
        
        for i, prompt_data in enumerate(prompts, 1):
            detection_log = prompt_data.get('detection_log', [])
            
            if not detection_log:
                print(f"  ⚠️  Prompt {i}: No detection log!")
                continue
            
            # Extract LPIPS values
            lpips_values = [entry['lpips_change'] for entry in detection_log]
            min_lpips = min(lpips_values) if lpips_values else None
            
            all_lpips_values.extend(lpips_values)
            category_lpips.extend(lpips_values)
            
            print(f"  Prompt {i}: LPIPS range {min(lpips_values):.4f} - {max(lpips_values):.4f}, "
                  f"min: {min_lpips:.4f}, stopped: {prompt_data['stopped_early']}")
        
        categories_data[category] = category_lpips
    
    # Overall statistics
    print("\n" + "="*60)
    print("📊 LPIPS STATISTICS")
    print("="*60)
    
    if all_lpips_values:
        print(f"\nOverall LPIPS values:")
        print(f"  Minimum: {min(all_lpips_values):.4f}")
        print(f"  Maximum: {max(all_lpips_values):.4f}")
        print(f"  Mean: {np.mean(all_lpips_values):.4f}")
        print(f"  Median: {np.median(all_lpips_values):.4f}")
        print(f"  25th percentile: {np.percentile(all_lpips_values, 25):.4f}")
        print(f"  10th percentile: {np.percentile(all_lpips_values, 10):.4f}")
        print(f"  5th percentile: {np.percentile(all_lpips_values, 5):.4f}")
        
        print(f"\n🎯 Current threshold: 0.020")
        print(f"   % of measurements below threshold: {sum(1 for v in all_lpips_values if v < 0.02) / len(all_lpips_values) * 100:.1f}%")
        
        # Suggest new thresholds
        print("\n💡 SUGGESTED THRESHOLDS TO TEST:")
        percentiles = [95, 90, 85, 80, 75, 70]
        for p in percentiles:
            thresh = np.percentile(all_lpips_values, p)
            print(f"   {p}th percentile: {thresh:.4f} ({100-p}% of measurements would trigger stop)")
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Distribution of LPIPS values
    ax1.hist(all_lpips_values, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax1.axvline(0.02, color='red', linestyle='--', linewidth=2, label='Current threshold (0.02)')
    ax1.set_xlabel('LPIPS Change Value', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Distribution of LPIPS Changes', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Box plot by category
    category_names = list(categories_data.keys())
    category_values = [categories_data[cat] for cat in category_names]
    
    ax2.boxplot(category_values, labels=category_names)
    ax2.axhline(0.02, color='red', linestyle='--', linewidth=2, label='Threshold (0.02)')
    ax2.set_ylabel('LPIPS Change Value', fontsize=12)
    ax2.set_title('LPIPS Changes by Category', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / 'lpips_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\n📊 Visualization saved: {results_dir / 'lpips_analysis.png'}")
    
    # Critical analysis
    print("\n" + "="*60)
    print("🔍 CRITICAL ANALYSIS")
    print("="*60)
    
    min_value = min(all_lpips_values)
    if min_value > 0.02:
        print(f"\n❌ THRESHOLD TOO LOW!")
        print(f"   Even the MINIMUM LPIPS value ({min_value:.4f}) is above threshold (0.02)")
        print(f"   NO prompt could possibly stop early with this threshold")
        print(f"\n   Recommendation: Try threshold around {np.percentile(all_lpips_values, 80):.4f}")
    elif min_value < 0.01:
        print(f"\n✅ Some measurements are low enough")
        print(f"   Minimum: {min_value:.4f}")
        print(f"   But most values are higher - threshold might need adjustment")
    
    # Compare to Week 2 (if we can infer)
    print("\n💭 Hypothesis:")
    if np.median(all_lpips_values) > 0.03:
        print("   These prompts have HIGHER perceptual changes than Week 2 prompts")
        print("   Possible reasons:")
        print("   1. More complex prompts = larger visual changes")
        print("   2. Week 2 prompts were unusually convergent")
        print("   3. Detection parameters (check_every, window) need adjustment")

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║  DEBUG: LPIPS VALUES ANALYSIS                           ║
    ║                                                          ║
    ║  Understanding why threshold 0.02 didn't work           ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    analyze_detection_logs()
    
    print("\n" + "="*60)
    print("✅ Analysis complete!")
    print("\nNext step: Run Phase B with adjusted threshold range")
