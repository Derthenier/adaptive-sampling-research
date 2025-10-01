# ContentAware: Adaptive Step Diffusion

> **Research Project**: Making Stable Diffusion faster through content-aware adaptive sampling

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8+-ee4c2c.svg)](https://pytorch.org/)

---

## 🎯 The Problem

Current text-to-image diffusion models (Stable Diffusion, SDXL) use a **fixed number of denoising steps** for all images:

- Simple prompts ("a red apple") → 30 steps → **Overkill** ⚠️
- Complex prompts ("cyberpunk city at night with rain") → 30 steps → **Could use more** ⚠️
- Everyone gets the same treatment → **Inefficient** ⚠️

**Result**: Wasted computation on easy images, insufficient quality on hard ones.

---

## 💡 Our Solution

**ContentAware** dynamically predicts the optimal number of denoising steps based on:
- Prompt complexity (text analysis)
- Content difficulty (early-step latent analysis)
- Real-time convergence detection

### Key Innovation
Unlike distillation methods (LCM, progressive distillation) that require expensive retraining, our approach:
- ✅ Works at **inference time**
- ✅ No model retraining needed
- ✅ Lightweight predictor (<1% overhead)
- ✅ Adapts per-image in real-time
- ✅ Maintains quality while improving speed

---

## 📊 Expected Results

| Metric | Target | Status |
|--------|--------|--------|
| Speed Improvement | 30-50% | 🔬 In Progress |
| Quality Retention | >95% | 🔬 In Progress |
| Predictor Overhead | <0.1s | 🔬 In Progress |
| Generalization | 90%+ prompts | 🔬 In Progress |

---

## 🏗️ Architecture

```
Input Prompt
    ↓
┌─────────────────────────┐
│  Feature Extraction     │
│  - Text analysis        │
│  - Semantic embeddings  │
└─────────────────────────┘
    ↓
┌─────────────────────────┐
│  Step Predictor         │
│  (Lightweight ML model) │
└─────────────────────────┘
    ↓
┌─────────────────────────┐
│  Adaptive Diffusion     │
│  - Dynamic step count   │
│  - Early stopping       │
│  - Quality monitoring   │
└─────────────────────────┘
    ↓
Output Image (Faster!)
```

---

## Results

### 🔬 Week 1 Results (Baseline Analysis)

**Hardware**: RTX 5070 Ti 16GB
**Baseline**: 2.39s for 512x512 @ 30 steps

### Key Findings:
- Simple prompts: X steps sufficient
- Complex prompts: Y steps needed
- Potential speedup: Z%

[See detailed results →](week1_results/)

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/Derthenier/adaptive-sampling-research.git
cd adaptive-sampling-research

# Install dependencies
pip install -r requirements.txt

# Verify setup
python experiments/verify_setup.py
```

### Run Baseline Analysis

```bash
# Test different step counts (Week 1 experiments)
python experiments/week1_experiments.py

# Results saved to week1_results/
```

### Generate with Adaptive Sampling (Coming Soon)

```bash
# Once predictor is trained
python generate_adaptive.py --prompt "your prompt here"
```

---

## 📁 Project Structure

```
adaptive-sampling-research/
├── experiments/          # Research experiments
│   ├── week1_experiments.py
│   ├── experiment_framework.py
│   └── verify_setup.py
├── generation/           # SD generation scripts
│   ├── generate_image.py
│   └── generate_optimized.py
├── planning/             # Research planning docs
│   ├── master_plan.md
│   ├── adaptive_sampling_plan.md
│   └── research_areas.md
├── papers/               # Literature review
│   ├── essential_papers.md
│   └── adaptive_sampling_papers.md
└── results/              # Experiment results (gitignored)
```

---

## 🔬 Research Timeline

**12-Week Research Project** (Current: Week 1)

- **Phase 1** (Weeks 1-3): Foundation & baseline analysis ← **YOU ARE HERE**
- **Phase 2** (Weeks 4-6): Predictor development
- **Phase 3** (Weeks 7-9): Validation & optimization
- **Phase 4** (Weeks 10-12): Publication & community release

[See detailed timeline →](planning/adaptive_sampling_plan.md)

---

## 📚 Key Papers

This research builds on:

- **DDIM** (Song et al., 2020) - Deterministic sampling
- **DPM-Solver** (Lu et al., 2022) - Efficient ODE solvers
- **Latent Consistency Models** (Luo et al., 2023) - 4-step generation
- **Progressive Distillation** (Salimans & Ho, 2022) - Step reduction

[Full reading list →](papers/adaptive_sampling_papers.md)

---

## 🎯 Success Metrics

### Minimum Viable Contribution
- [ ] 20% speed improvement
- [ ] <10% quality loss
- [ ] Works on 80% of prompts
- [ ] Open source release

### Target Success
- [ ] 30-40% speed improvement
- [ ] <5% quality loss
- [ ] Works on 90% of prompts
- [ ] 100+ GitHub stars
- [ ] Community adoption

### Stretch Goals
- [ ] Paper publication
- [ ] Integration into ComfyUI/A1111
- [ ] Becomes standard practice

---

## 🛠️ Hardware Requirements

**Minimum**: 
- NVIDIA GPU with 8GB+ VRAM
- CUDA 11.8+
- 16GB+ system RAM

**Recommended** (Used in this research):
- NVIDIA RTX 5070 Ti (16GB VRAM)
- CUDA 12.9
- PyTorch 2.9

---

## 🤝 Contributing

This is an active research project. Contributions welcome after initial results are published!

**Areas for contribution**:
- Testing on different hardware
- Additional evaluation metrics
- SDXL integration
- Different sampling methods

---

## 📖 Documentation

- [Master Plan](planning/master_plan.md) - Overall research strategy
- [Week-by-Week Guide](planning/adaptive_sampling_plan.md) - Detailed implementation
- [Paper Reading List](papers/adaptive_sampling_papers.md) - Literature review
- [Technical Deep Dive](planning/technical_deepdive.md) - Architecture details

---

## 📝 Citation

If you use this work in your research, please cite:

```bibtex
@misc{contentaware2025,
  title={ContentAware: Adaptive Step Diffusion for Stable Diffusion},
  author={Aditya Vennelakanti},
  year={2025},
  publisher={GitHub},
  url={https://github.com/Derthenier/adaptive-sampling-research}
}
```

---

## 📬 Contact

- **Researcher**: Aditya Vennelakanti
- **Email**: avennelakanti@gmail.com

---

## 📄 License

MIT License - See [LICENSE](LICENSE) for details

---

## 🙏 Acknowledgments

- Stability AI for Stable Diffusion
- HuggingFace for diffusers library
- Research community for foundational papers
- Claude AI for research guidance

---

## 🔗 Related Projects

- [Latent Consistency Models](https://github.com/luosiallen/latent-consistency-model)
- [DPM-Solver](https://github.com/LuChengTHU/dpm-solver)
- [ControlNet](https://github.com/lllyasviel/ControlNet)

---

**Status**: 🔬 Active Research (Week 1 of 12)

**Last Updated**: October 2025

---

<div align="center">

### ⭐ Star this repo to follow the research progress!

**Building the future of efficient diffusion models, one step at a time.**

</div>
