#!/usr/bin/env python3
"""
Simple Results Analyzer - Helps you understand Week 1 data

This script makes it EASY to see patterns in your results
No PhD required - just run it!
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt

def analyze_category(category_name, results_list):
    """Analyze results for one category (simple/medium/complex)"""
    
    print(f"\n{'='*60}")
    print(f"📊 ANALYZING: {category_name.upper()}")
    print(f"{'='*60}\n")
    
    for result in results_list:
        prompt = result['prompt']
        print(f"Prompt: '{prompt}'")
        print("-" * 60)
        
        experiments = result['experiments']
        
        # Find baseline (30 steps)
        baseline = next((e for e in experiments if e['steps'] == 30), None)
        
        if baseline:
            baseline_time = baseline['generation_time']
            baseline_score = baseline['clip_score']
            
            print(f"📌 BASELINE (30 steps):")
            print(f"   Time: {baseline_time:.2f}s")
            print(f"   CLIP Score: {baseline_score:.2f}")
            print()
            
            # Find "good enough" options (>=99% quality)
            threshold = baseline_score * 0.99
            
            print(f"🎯 LOOKING FOR: CLIP ≥ {threshold:.2f} (99% of baseline)")
            print()
            
            good_enough = []
            for exp in experiments:
                if exp['clip_score'] >= threshold:
                    good_enough.append(exp)
            
            if good_enough:
                # Find fastest "good enough" option
                best = min(good_enough, key=lambda x: x['steps'])
                
                time_saved = baseline_time - best['generation_time']
                speedup_pct = (time_saved / baseline_time) * 100
                quality_loss = ((baseline_score - best['clip_score']) / baseline_score) * 100
                
                print(f"✅ SWEET SPOT FOUND:")
                print(f"   Steps: {best['steps']} (vs 30 baseline)")
                print(f"   Time: {best['generation_time']:.2f}s (vs {baseline_time:.2f}s)")
                print(f"   CLIP: {best['clip_score']:.2f} (vs {baseline_score:.2f})")
                print(f"   ⚡ SPEEDUP: {speedup_pct:.1f}%")
                print(f"   📉 Quality Loss: {quality_loss:.1f}%")
                
                if speedup_pct > 20:
                    print(f"   🎉 SIGNIFICANT SPEEDUP POTENTIAL!")
                
            else:
                print("❌ No options found with ≥99% quality")
                print("   → This prompt might need all 30 steps")
        
        print("\n")

def analyze_all_results():
    """Analyze all Week 1 results and show patterns"""
    
    results_file = Path("week1_results/benchmark_results.json")
    
    if not results_file.exists():
        print("❌ Results file not found!")
        print("   Make sure you ran: python experiments/week1_experiments.py")
        return
    
    # Load results
    with open(results_file, 'r') as f:
        all_results = json.load(f)
    
    print("="*60)
    print("🔬 WEEK 1 RESULTS ANALYSIS")
    print("="*60)
    print()
    print("This will help you understand:")
    print("  1. Which prompts can use fewer steps")
    print("  2. How much speedup is possible")
    print("  3. What the quality tradeoff looks like")
    print()
    
    # Analyze each category
    for category in ['simple', 'medium', 'complex']:
        if category in all_results:
            analyze_category(category, all_results[category])
    
    # Summary analysis
    print("\n" + "="*60)
    print("🎯 SUMMARY & HYPOTHESIS FORMATION")
    print("="*60)
    print()
    
    # Collect all "sweet spots"
    all_speedups = []
    for category in ['simple', 'medium', 'complex']:
        if category not in all_results:
            continue
            
        for result in all_results[category]:
            experiments = result['experiments']
            baseline = next((e for e in experiments if e['steps'] == 30), None)
            
            if baseline:
                threshold = baseline['clip_score'] * 0.99
                good_enough = [e for e in experiments if e['clip_score'] >= threshold]
                
                if good_enough:
                    best = min(good_enough, key=lambda x: x['steps'])
                    speedup = ((baseline['generation_time'] - best['generation_time']) 
                              / baseline['generation_time']) * 100
                    
                    all_speedups.append({
                        'category': category,
                        'prompt': result['prompt'],
                        'steps': best['steps'],
                        'speedup': speedup
                    })
    
    if all_speedups:
        # Group by category
        for category in ['simple', 'medium', 'complex']:
            cat_speedups = [s for s in all_speedups if s['category'] == category]
            
            if cat_speedups:
                avg_speedup = sum(s['speedup'] for s in cat_speedups) / len(cat_speedups)
                avg_steps = sum(s['steps'] for s in cat_speedups) / len(cat_speedups)
                
                print(f"📊 {category.upper()}:")
                print(f"   Average optimal steps: {avg_steps:.1f}")
                print(f"   Average speedup: {avg_speedup:.1f}%")
                print()
        
        # Overall
        overall_speedup = sum(s['speedup'] for s in all_speedups) / len(all_speedups)
        print(f"🎯 OVERALL POTENTIAL: {overall_speedup:.1f}% average speedup")
        print()
        
        # Suggest hypothesis
        print("💡 SUGGESTED HYPOTHESIS:")
        print()
        
        simple_speedups = [s['speedup'] for s in all_speedups if s['category'] == 'simple']
        complex_speedups = [s['speedup'] for s in all_speedups if s['category'] == 'complex']
        
        if simple_speedups and complex_speedups:
            avg_simple = sum(simple_speedups) / len(simple_speedups)
            avg_complex = sum(complex_speedups) / len(complex_speedups)
            
            if avg_simple > avg_complex + 10:
                print("   'Simple prompts converge faster than complex ones.'")
                print("   'Text complexity (length, number of objects) may predict")
                print("   optimal step count. Should build predictor based on")
                print("   prompt analysis.'")
            elif abs(avg_simple - avg_complex) < 10:
                print("   'Step requirements don't vary much by text complexity.'")
                print("   'May need early-step features (latent analysis) instead")
                print("   of text features to predict convergence.'")
        
        print()
        print("="*60)
        print()
        print("✅ NEXT STEPS:")
        print("   1. Look at the images visually")
        print("   2. Verify these numbers match what you see")
        print("   3. Write your hypothesis in planning/hypothesis.md")
        print("   4. Move to Week 2: Feature engineering!")
        print()

def plot_results():
    """Create visual plots of the results"""
    
    results_file = Path("week1_results/benchmark_results.json")
    
    if not results_file.exists():
        print("❌ Results file not found!")
        return
    
    with open(results_file, 'r') as f:
        all_results = json.load(f)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Week 1 Results Analysis', fontsize=16, fontweight='bold')
    
    categories = ['simple', 'medium', 'complex']
    colors = ['green', 'blue', 'red']
    
    # Plot 1: Steps vs CLIP Score
    ax1 = axes[0, 0]
    for category, color in zip(categories, colors):
        if category in all_results:
            for result in all_results[category]:
                steps = [e['steps'] for e in result['experiments']]
                scores = [e['clip_score'] for e in result['experiments']]
                ax1.plot(steps, scores, 'o-', alpha=0.5, color=color, label=category)
    
    ax1.set_xlabel('Number of Steps')
    ax1.set_ylabel('CLIP Score (Quality)')
    ax1.set_title('Quality vs Steps')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Steps vs Time
    ax2 = axes[0, 1]
    for category, color in zip(categories, colors):
        if category in all_results:
            for result in all_results[category]:
                steps = [e['steps'] for e in result['experiments']]
                times = [e['generation_time'] for e in result['experiments']]
                ax2.plot(steps, times, 'o-', alpha=0.5, color=color, label=category)
    
    ax2.set_xlabel('Number of Steps')
    ax2.set_ylabel('Generation Time (seconds)')
    ax2.set_title('Speed vs Steps')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Category comparison
    ax3 = axes[1, 0]
    category_optimal_steps = []
    for category in categories:
        if category in all_results:
            optimal_steps = []
            for result in all_results[category]:
                baseline = next((e for e in result['experiments'] if e['steps'] == 30))
                threshold = baseline['clip_score'] * 0.99
                good = [e for e in result['experiments'] if e['clip_score'] >= threshold]
                if good:
                    best = min(good, key=lambda x: x['steps'])
                    optimal_steps.append(best['steps'])
            
            if optimal_steps:
                avg = sum(optimal_steps) / len(optimal_steps)
                category_optimal_steps.append(avg)
    
    if category_optimal_steps:
        ax3.bar(categories, category_optimal_steps, color=colors)
        ax3.set_ylabel('Average Optimal Steps')
        ax3.set_title('Optimal Steps by Category')
        ax3.axhline(y=30, color='red', linestyle='--', label='Baseline (30)')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Speedup potential
    ax4 = axes[1, 1]
    category_speedups = []
    for category in categories:
        if category in all_results:
            speedups = []
            for result in all_results[category]:
                baseline = next((e for e in result['experiments'] if e['steps'] == 30))
                threshold = baseline['clip_score'] * 0.99
                good = [e for e in result['experiments'] if e['clip_score'] >= threshold]
                if good:
                    best = min(good, key=lambda x: x['steps'])
                    speedup = ((baseline['generation_time'] - best['generation_time']) 
                              / baseline['generation_time']) * 100
                    speedups.append(speedup)
            
            if speedups:
                avg = sum(speedups) / len(speedups)
                category_speedups.append(avg)
    
    if category_speedups:
        ax4.bar(categories, category_speedups, color=colors)
        ax4.set_ylabel('Average Speedup (%)')
        ax4.set_title('Potential Speedup by Category')
        ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save plot
    plot_file = Path("week1_results/analysis_plots.png")
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"📊 Plots saved to: {plot_file}")
    
    # Show plot
    plt.show()

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║        WEEK 1 RESULTS ANALYZER                          ║
    ║                                                          ║
    ║  This will help you understand your experiment results  ║
    ║  and form your first hypothesis!                        ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    print("\n1. Analyzing text results...")
    analyze_all_results()
    
    print("\n2. Creating visualizations...")
    try:
        plot_results()
    except Exception as e:
        print(f"⚠️  Plotting failed: {e}")
        print("   (That's okay - text analysis above is most important)")
    
    print("\n" + "="*60)
    print("🎉 ANALYSIS COMPLETE!")
    print("="*60)
    print()
    print("Now you have:")
    print("  ✅ Text analysis of results")
    print("  ✅ Visual plots (if matplotlib worked)")
    print("  ✅ Suggested hypothesis")
    print()
    print("Next: Look at the images yourself and verify!")
