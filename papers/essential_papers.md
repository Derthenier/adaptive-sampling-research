# ESSENTIAL PAPERS - PRIORITIZED READING LIST

Reading strategy: DEPTH over breadth. Understand core papers deeply before expanding.

---

## 🔥 TIER 0: ABSOLUTE MUST-READ (Week 1-2)
**Read these FIRST. Understand them DEEPLY.**

### 1. Denoising Diffusion Probabilistic Models (DDPM)
**Ho et al., 2020 | NeurIPS**
- **Why**: Foundation of everything
- **Key insight**: Diffusion as iterative denoising
- **Read for**: Math, algorithm, intuition
- **Time**: 8-10 hours (read 3x)
- **ArXiv**: https://arxiv.org/abs/2006.11239
- **Action**: Implement toy version on MNIST

### 2. High-Resolution Image Synthesis with Latent Diffusion Models
**Rombach et al., 2022 | CVPR (The Stable Diffusion Paper)**
- **Why**: This IS Stable Diffusion
- **Key insight**: Diffusion in latent space
- **Read for**: Architecture, VAE + UNet
- **Time**: 6-8 hours (read 2x)
- **ArXiv**: https://arxiv.org/abs/2112.10752
- **Action**: Diagram the full pipeline

### 3. SDXL: Improving Latent Diffusion Models
**Podell et al., 2023 | Stability AI**
- **Why**: Current state-of-the-art baseline
- **Key insight**: Scale + refinement
- **Read for**: What changed from SD 1.5
- **Time**: 3-4 hours
- **ArXiv**: https://arxiv.org/abs/2307.01952
- **Action**: List all improvements

---

## ⭐ TIER 1: CORE FOUNDATION (Week 3-4)
**Essential background and context**

### 4. Denoising Diffusion Implicit Models (DDIM)
**Song et al., 2020 | ICLR**
- **Why**: Faster sampling, deterministic
- **Key insight**: Skip steps intelligently
- **Time**: 4-5 hours
- **ArXiv**: https://arxiv.org/abs/2010.02502

### 5. Classifier-Free Diffusion Guidance
**Ho & Salimans, 2022**
- **Why**: How text guidance works
- **Key insight**: Unconditional + conditional
- **Time**: 3-4 hours
- **ArXiv**: https://arxiv.org/abs/2207.12598

### 6. Attention Is All You Need (Transformers)
**Vaswani et al., 2017 | NeurIPS**
- **Why**: Understand UNet's attention
- **Key insight**: Self-attention mechanism
- **Time**: 6-8 hours (famous but dense)
- **ArXiv**: https://arxiv.org/abs/1706.03762

### 7. Learning Transferable Visual Models From Natural Language Supervision (CLIP)
**Radford et al., 2021 | OpenAI**
- **Why**: Text encoder in SD
- **Key insight**: Vision-language alignment
- **Time**: 4-5 hours
- **ArXiv**: https://arxiv.org/abs/2103.00020

---

## 📚 TIER 2: SPECIALIZATION (Month 2)
**Pick based on your chosen research area**

### For Efficient Attention Research:

#### FlashAttention: Fast and Memory-Efficient Exact Attention
**Dao et al., 2022 | NeurIPS**
- **Why**: 2-3x speedup for attention
- **Key insight**: IO-aware algorithm
- **ArXiv**: https://arxiv.org/abs/2205.14135

#### Linformer: Self-Attention with Linear Complexity
**Wang et al., 2020**
- **Why**: O(n) instead of O(n²)
- **ArXiv**: https://arxiv.org/abs/2006.04768

### For Sampling Research:

#### DPM-Solver: Fast Solver for Guided Sampling
**Lu et al., 2022 | NeurIPS**
- **Why**: Better samplers
- **ArXiv**: https://arxiv.org/abs/2206.00927

#### Latent Consistency Models
**Luo et al., 2023**
- **Why**: 4-step generation
- **ArXiv**: https://arxiv.org/abs/2310.04378

### For Control/Conditioning Research:

#### Adding Conditional Control to Text-to-Image Diffusion (ControlNet)
**Zhang et al., 2023**
- **Why**: Standard for control
- **ArXiv**: https://arxiv.org/abs/2302.05543

#### IP-Adapter: Text Compatible Image Prompt Adapter
**Ye et al., 2023**
- **Why**: Image-based conditioning
- **ArXiv**: https://arxiv.org/abs/2308.06721

#### T2I-Adapter: Learning Adapters to Dig out More Controllable Ability
**Mou et al., 2023**
- **Why**: Lightweight control
- **ArXiv**: https://arxiv.org/abs/2302.08453

### For Training/Architecture Research:

#### Progressive Distillation for Fast Sampling
**Salimans & Ho, 2022**
- **Why**: Train fast models
- **ArXiv**: https://arxiv.org/abs/2202.00512

#### Elucidating the Design Space of Diffusion-Based Generative Models (EDM)
**Karras et al., 2022 | NeurIPS**
- **Why**: Design principles
- **ArXiv**: https://arxiv.org/abs/2206.00364

---

## 📖 TIER 3: ADVANCED TOPICS (Month 3+)
**For deeper specialization**

### LoRA: Low-Rank Adaptation
**Hu et al., 2021 | ICLR**
- **Why**: Efficient fine-tuning
- **ArXiv**: https://arxiv.org/abs/2106.09685

### DreamBooth: Fine Tuning Text-to-Image Diffusion Models
**Ruiz et al., 2022 | Google**
- **Why**: Personalization technique
- **ArXiv**: https://arxiv.org/abs/2208.12242

### Score-Based Generative Modeling through SDEs
**Song et al., 2021 | ICLR**
- **Why**: Theoretical foundation
- **ArXiv**: https://arxiv.org/abs/2011.13456

### Scalable Diffusion Models with Transformers (DiT)
**Peebles & Xie, 2023 | Meta**
- **Why**: Transformer-based diffusion
- **ArXiv**: https://arxiv.org/abs/2212.09748

---

## 🎯 READING STRATEGY

### First Pass (1 hour)
- [ ] Read abstract
- [ ] Look at figures
- [ ] Read conclusion
- [ ] Skim introduction
- **Goal**: Understand main idea

### Second Pass (2-3 hours)
- [ ] Read methodology carefully
- [ ] Understand experiments
- [ ] Note key equations
- [ ] Try to reproduce figures mentally
- **Goal**: Understand how it works

### Third Pass (4-6 hours)
- [ ] Implement key algorithms
- [ ] Reproduce experiments (small scale)
- [ ] Compare to related work
- [ ] Identify limitations
- **Goal**: Master the technique

### Active Reading
**Don't just read - DO:**
- ✍️ Take notes in your own words
- 💻 Implement key algorithms
- 🤔 Ask "why?" at each step
- 📊 Reproduce key figures
- 💬 Discuss with others

---

## 📝 PAPER ANNOTATION TEMPLATE

For each paper, create a document:

```markdown
# [Paper Title]

## One-Sentence Summary


## Problem Being Solved


## Key Insight


## Method (High Level)


## Method (Technical Details)


## Experiments & Results


## Strengths
-
-

## Weaknesses
-
-

## Connections to Other Work


## Ideas for My Research


## Code Available?


## Questions / Confusions


## Implementation Difficulty
[ ] Easy [ ] Medium [ ] Hard [ ] Very Hard

## Priority for My Research
[ ] Critical [ ] Important [ ] Nice to have [ ] Not relevant
```

---

## 🔍 WHERE TO FIND PAPERS

### Primary Sources
- **ArXiv.org**: Preprints (search: cs.CV, cs.LG)
- **Papers with Code**: Papers + implementations
- **Google Scholar**: Citations and related work
- **Semantic Scholar**: AI-powered search

### Staying Current
- **Twitter**: Follow key researchers
- **Reddit**: r/MachineLearning, r/StableDiffusion
- **Discord**: Stable Diffusion servers
- **Newsletters**: Import AI, The Batch

### Key Researchers to Follow
- **Diffusion Models**: 
  - Jonathan Ho (Google)
  - Jiaming Song (NVIDIA)
  - Prafulla Dhariwal (OpenAI)
  - Robin Rombach (Stability AI)
  
- **Efficient Methods**:
  - Tri Dao (Princeton - FlashAttention)
  - William Peebles (Meta - DiT)
  
- **Control Methods**:
  - Lvmin Zhang (ControlNet)
  - Various Stability AI researchers

---

## 📅 READING SCHEDULE

### Week 1: Foundations
- Mon-Tue: DDPM
- Wed-Thu: Latent Diffusion (SD)
- Fri: SDXL
- Weekend: Review and implement toy model

### Week 2: Core Concepts
- Mon: DDIM
- Tue: Classifier-Free Guidance
- Wed-Thu: Transformers/Attention
- Fri: CLIP
- Weekend: Experiment with SD modifications

### Week 3-4: Specialization
- Pick 3-4 papers from Tier 2 based on research direction
- Deep dive into each
- Start prototyping ideas

### Month 2+: Continuous Learning
- Read 1-2 new papers per week
- Focus on your specific area
- Track new ArXiv postings
- Engage with community discussions

---

## 🎓 UNDERSTANDING CHECKLIST

After reading a paper, you should be able to:
- [ ] Explain it to a friend in 2 minutes
- [ ] Draw the architecture from memory
- [ ] Implement key algorithm (even if slow)
- [ ] Identify 3 limitations
- [ ] Suggest 2 improvements
- [ ] Connect it to 3 other papers

If you can't do these, read again!

---

## 💡 READING TIPS

1. **Don't read linearly**: Abstract → Figures → Intro → Method → Experiments
2. **Skip the fluff**: Focus on method and experiments
3. **Math is scary but necessary**: Work through equations step by step
4. **Implement to understand**: Code > passive reading
5. **Discuss**: Join reading groups, explain to others
6. **Take breaks**: Dense papers need mental processing time
7. **Revisit**: Understanding deepens on second/third read

---

## 🚫 COMMON PITFALLS

**Don't:**
- ❌ Read too many papers superficially
- ❌ Just skim abstracts
- ❌ Get stuck on one section
- ❌ Forget to implement
- ❌ Read without taking notes
- ❌ Skip the math entirely
- ❌ Ignore related work section

**Do:**
- ✅ Master core papers deeply
- ✅ Read actively with pen/code
- ✅ Move on if stuck (come back later)
- ✅ Implement key ideas
- ✅ Annotate and summarize
- ✅ Work through equations
- ✅ Mine related work for ideas

---

**Next Step**: Start with DDPM paper TODAY. Spend 2-3 hours on first pass.

Set a timer. Take notes. Come back with questions!