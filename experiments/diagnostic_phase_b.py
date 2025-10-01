#!/usr/bin/env python3
"""
Diagnostic Phase B: Test wide threshold range to find working zone
Based on Phase A failure, test much wider threshold range
"""

import torch
from diffusers import StableDiffusionPipeline
import lpips
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

class DiagnosticTester:
    """Test wide threshold range to understand LPIPS behavior"""
    
    def __init__(self):
        print("🔬 Diagnostic Testing Mode")
        
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            safety_checker=None,
        ).to("cuda")
        self.pipe.enable_attention_slicing()
        
        self.lpips_model = lpips.LPIPS(net='alex').cuda()
        
        self.results_dir = Path('diagnostic_phase_b')
        self.results_dir.mkdir(exist_ok=True)
    
    def generate_with_full_logging(self, prompt, threshold, min_steps=15, check_every=2):
        """Generate with extensive logging"""
        
        previous_image = None
        detection_log = []
        stopped_step = None
        
        def callback(step, timestep, latents):
            nonlocal previous_image, stopped_step
            
            if step < min_steps or step % check_every != 0:
                return
            
            latents_scaled = latents / self.pipe.vae.config.scaling_factor
            current_image = self.pipe.vae.decode(latents_scaled, return_dict=False)[0]
            
            if previous_image is not None:
                with torch.no_grad():
                    img1_norm = previous_image * 2 - 1
                    img2_norm = current_image * 2 - 1
                    change = self.lpips_model(img1_norm, img2_norm).item()
                
                detection_log.append({
                    'step': step,
                    'lpips': change,
                    'below_threshold': change < threshold
                })
                
                if change < threshold and stopped_step is None:
                    stopped_step = step
                    return False
            
            previous_image = current_image.clone()
        
        generator = torch.Generator("cuda").manual_seed(42)
        result = self.pipe(
            prompt=prompt,
            num_inference_steps=30,
            guidance_scale=7.5,
            generator=generator,
            callback=callback,
            callback_steps=1,
        )
        
        return {
            'image': result.images[0],
            'stopped_step': stopped_step,
            'actual_steps': stopped_step if stopped_step else 30,
            'detection_log': detection_log,
        }
    
    def run_diagnostic(self):
        """Run comprehensive diagnostic"""
        
        print("="*60)
        print("DIAGNOSTIC: Wide Threshold Range Test")
        print("="*60)
        
        # Diagnostic prompts - mix of complexities
        prompts = [
            "a simple red apple",
            "a portrait of a woman",
            "a mountain landscape",
            "a busy city street with cars",
        ]
        
        # Wide threshold range - from very aggressive to very conservative
        thresholds = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.15]
        
        all_results = {}
        
        for prompt in prompts:
            print(f"\n📝 Testing: {prompt}")
            print("-" * 60)
            
            prompt_results = []
            
            for threshold in thresholds:
                result = self.generate_with_full_logging(prompt, threshold)
                
                prompt_results.append({
                    'threshold': threshold,
                    'stopped_step': result['stopped_step'],
                    'actual_steps': result['actual_steps'],
                    'stopped_early': result['stopped_step'] is not None,
                    'detection_log': result['detection_log'],
                })
                
                status = f"✅ {result['stopped_step']}" if result['stopped_step'] else "❌ 30"
                min_lpips = min((e['lpips'] for e in result['detection_log']), default=None)
                print(f"  {threshold:5.2f}: {status:8s} (min LPIPS: {min_lpips:.4f})")
            
            all_results[prompt] = prompt_results
        
        # Save results
        with open(self.results_dir / 'diagnostic_results.json', 'w') as f:
            json.dump(all_results, f, indent=2)
        
        # Analysis
        self.analyze_diagnostic(all_results, thresholds)
        
        return all_results
    
    def analyze_diagnostic(self, results, thresholds):
        """Analyze diagnostic results"""
        
        print("\n" + "="*60)
        print("📊 DIAGNOSTIC ANALYSIS")
        print("="*60)
        
        # Find working threshold range
        threshold_success = {t: 0 for t in thresholds}
        threshold_avg_steps = {t: [] for t in thresholds}
        
        for prompt, prompt_results in results.items():
            for result in prompt_results:
                threshold = result['threshold']
                if result['stopped_early']:
                    threshold_success[threshold] += 1
                    threshold_avg_steps[threshold].append(result['actual_steps'])
        
        total_prompts = len(results)
        
        print("\n📈 Success Rate by Threshold:")
        print(f"{'Threshold':<12} {'Success Rate':<15} {'Avg Steps':<12}")
        print("-" * 45)
        
        working_thresholds = []
        
        for threshold in thresholds:
            success_rate = threshold_success[threshold] / total_prompts
            avg_steps = np.mean(threshold_avg_steps[threshold]) if threshold_avg_steps[threshold] else 30
            
            marker = "✅" if success_rate >= 0.75 else "⚠️" if success_rate >= 0.5 else "❌"
            print(f"{marker} {threshold:<10.2f} {success_rate:<15.1%} {avg_steps:<12.1f}")
            
            if success_rate >= 0.75:
                working_thresholds.append(threshold)
        
        print("\n" + "="*60)
        
        if working_thresholds:
            print(f"✅ WORKING THRESHOLDS FOUND: {working_thresholds}")
            optimal = working_thresholds[0]  # Lowest that works = most aggressive
            print(f"\n🎯 RECOMMENDED THRESHOLD: {optimal:.2f}")
            
            # Calculate expected performance
            expected_steps = []
            for prompt_results in results.values():
                for result in prompt_results:
                    if result['threshold'] == optimal:
                        expected_steps.append(result['actual_steps'])
            
            avg_expected = np.mean(expected_steps)
            speedup = ((30 - avg_expected) / 30) * 100
            
            print(f"   Expected avg steps: {avg_expected:.1f}/30")
            print(f"   Expected speedup: {speedup:.1f}%")
            
            print("\n💡 Next Steps:")
            print(f"   1. Use threshold {optimal:.2f} for full Phase A re-run")
            print(f"   2. Test {len(working_thresholds)} working thresholds in detail")
            print(f"   3. Proceed to Week 4 with validated threshold")
            
        else:
            print("❌ NO WORKING THRESHOLDS FOUND")
            print("\n🔍 This means:")
            print("   • LPIPS changes are consistently too high")
            print("   • OR detection method needs fundamental revision")
            print("   • OR need different perceptual metric")
            
            print("\n💡 Options:")
            print("   1. Try even higher thresholds (0.20, 0.30)")
            print("   2. Check if detection is working (examine logs)")
            print("   3. Consider alternative metrics (SSIM, MS-SSIM)")
            print("   4. Re-examine Week 2 results (were they anomalous?)")
        
        # Create visualization
        self.plot_diagnostic(results, thresholds)
    
    def plot_diagnostic(self, results, thresholds):
        """Visualize diagnostic results"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: LPIPS trajectories for different prompts
        colors = ['red', 'blue', 'green', 'orange']
        for (prompt, prompt_results), color in zip(results.items(), colors):
            # Get LPIPS trajectory for threshold 0.05 (middle value)
            for result in prompt_results:
                if result['threshold'] == 0.05:
                    steps = [e['step'] for e in result['detection_log']]
                    lpips_values = [e['lpips'] for e in result['detection_log']]
                    ax1.plot(steps, lpips_values, 'o-', color=color, 
                            label=prompt[:30], alpha=0.7)
                    break
        
        ax1.axhline(0.05, color='red', linestyle='--', label='Threshold 0.05')
        ax1.set_xlabel('Step', fontsize=12)
        ax1.set_ylabel('LPIPS Change', fontsize=12)
        ax1.set_title('LPIPS Trajectories', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Success rate vs threshold
        success_rates = []
        for threshold in thresholds:
            success = sum(1 for prompt_results in results.values() 
                         for r in prompt_results 
                         if r['threshold'] == threshold and r['stopped_early'])
            success_rates.append(success / len(results))
        
        ax2.plot(thresholds, success_rates, 'bo-', linewidth=2, markersize=8)
        ax2.axhline(0.75, color='green', linestyle='--', alpha=0.5, label='Target (75%)')
        ax2.set_xlabel('Threshold', fontsize=12)
        ax2.set_ylabel('Success Rate', fontsize=12)
        ax2.set_title('Success Rate vs Threshold', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'diagnostic_analysis.png', dpi=150)
        print(f"\n📊 Visualization: {self.results_dir / 'diagnostic_analysis.png'}")


if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║  DIAGNOSTIC: Wide Threshold Range Test                  ║
    ║                                                          ║
    ║  Testing thresholds: 0.01 to 0.15                       ║
    ║  Goal: Find working threshold range                     ║
    ║                                                          ║
    ║  Expected time: 15-20 minutes                           ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    input("Press ENTER to start diagnostic...")
    
    tester = DiagnosticTester()
    results = tester.run_diagnostic()
    
    print("\n🎉 Diagnostic complete!")
    print("\nUse recommended threshold for Phase A re-run")