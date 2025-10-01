# ADAPTIVE SAMPLING - ESSENTIAL READING LIST

Papers specifically relevant to your adaptive sampling research, prioritized.

---

## 🔥 MUST READ (Week 1-2)

### 1. Denoising Diffusion Implicit Models (DDIM)
**Song et al., 2020**
- **Why crucial**: Deterministic sampling, can skip steps
- **Key for you**: Understand how to reduce steps safely
- **Focus on**: Section 4 (Sampling), Figure 2
- **Time**: 4 hours
- **ArXiv**: https://arxiv.org/abs/2010.02502
- **Read for**: How sampling schedules work

### 2. DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling
**Lu et al., 2022**
- **Why crucial**: 20 steps → 10 steps with better sampler
- **Key for you**: Alternative to reducing steps
- **Focus on**: Algorithm 1, experimental results
- **Time**: 3 hours
- **ArXiv**: https://arxiv.org/abs/2206.00927
- **Read for**: Efficient sampling mathematics

### 3. Fast Sampling of Diffusion Models with Exponential Integrator
**Zhang et al., 2022**
- **Why crucial**: Mathematical foundation for fewer steps
- **Key for you**: Theory behind step reduction
- **Time**: 3 hours
- **ArXiv**: https://arxiv.org/abs/2204.13902

---

## ⭐ HIGHLY RELEVANT (Week 3-4)

### 4. Latent Consistency Models
**Luo et al., 2023**
- **Why relevant**: 4-step generation!
- **Key for you**: How they achieved drastic reduction
- **Focus on**: LCM training, consistency distillation
- **Time**: 4 hours
- **ArXiv**: https://arxiv.org/abs/2310.04378
- **Note**: Requires training, but insights applicable

### 5. Progressive Distillation for Fast Sampling
**Salimans & Ho, 2022**
- **Why relevant**: Distill 1000 steps → 4 steps
- **Key for you**: Alternative approach to speed
- **Focus on**: Progressive training methodology
- **Time**: 3 hours
- **ArXiv**: https://arxiv.org/abs/2202.00512
- **Note**: Training-based, but understand principles

### 6. On Distillation of Guided Diffusion Models
**Meng et al., 2023**
- **Why relevant**: Speed up with guidance
- **Time**: 2 hours
- **ArXiv**: https://arxiv.org/abs/2210.03142

---

## 📚 SUPPORTING KNOWLEDGE (Week 5+)

### 7. gDDIM: Generalized DDIM for Diffusion Models
**Zhang & Chen, 2022**
- **Why relevant**: Flexible sampling schedules
- **ArXiv**: https://arxiv.org/abs/2206.05564

### 8. Accelerating Diffusion Models via Early Stop
**Lyu et al., 2022**
- **Why relevant**: DIRECTLY relevant! Early stopping research
- **Key insight**: Not all steps contribute equally
- **CRITICAL**: Read this carefully!
- **ArXiv**: https://arxiv.org/abs/2205.12524
- **Note**: This might be closest to your idea - read to differentiate

### 9. Fast Sampling via De-randomization for Discrete Diffusion Models
**Sun et al., 2023**
- **Why relevant**: Alternative perspective on speed
- **ArXiv**: https://arxiv.org/abs/2305.13352

---

## 🎯 RELATED TECHNIQUES (Optional)

### 10. Consistency Models
**Song et al., 2023**
- **Why relevant**: Single-step generation
- **Note**: Different approach but insightful
- **ArXiv**: https://arxiv.org/abs/2303.01469

### 11. Elucidating the Design Space of Diffusion-Based Generative Models
**Karras et al., 2022**
- **Why relevant**: Design principles, sampling schedules
- **Focus on**: Section 5 (sampling)
- **ArXiv**: https://arxiv.org/abs/2206.00364

---

## 📖 READING STRATEGY FOR YOUR PROJECT

### Phase 1: Understanding (Week 1)
**Read in this order:**
1. DDIM (how to skip steps)
2. DPM-Solver (better sampling)
3. Your baseline experiments

**Goal**: Understand sampling mechanics

### Phase 2: Related Work (Week 2)
**Read:**
4. LCM (extreme reduction)
5. Progressive Distillation (training approach)
6. **Accelerating via Early Stop** (closest to your idea!)

**Goal**: Know what exists, find your gap

### Phase 3: Deep Dives (Week 3-4)
**As needed based on your approach:**
- If doing learned stopping: Read early stop paper deeply
- If using better samplers: Deep dive DPM-Solver
- If considering distillation: Study progressive distillation

---

## 🔍 KEY QUESTIONS TO ANSWER

As you read, ask:

### From DDIM:
- How does sampling schedule affect quality?
- Can we skip steps non-uniformly?
- What determines step importance?

### From DPM-Solver:
- Why is solver choice important?
- Can we change solvers dynamically?
- Trade-offs vs step reduction?

### From LCM:
- How did they validate 4 steps?
- What quality metrics matter?
- Where does it fail?

### From Early Stop paper:
- How do they decide when to stop?
- What features predict convergence?
- Why is this different from your idea?

---

## 💡 YOUR INNOVATION vs EXISTING WORK

**Existing:**
- Fixed better samplers (DPM-Solver): Everyone gets optimized path
- Distillation (LCM): Requires expensive retraining
- Early stop: May not be content-aware

**Your contribution:**
- **Content-aware**: Different prompts get different steps
- **Inference-time**: No retraining base model
- **Dynamic**: Adapt during generation
- **Lightweight**: Minimal overhead predictor

**This is your differentiator!**

---

## 📋 READING CHECKLIST

### Week 1 (MUST DO):
- [ ] DDIM paper - first pass
- [ ] DDIM paper - deep read
- [ ] DPM-Solver - skim
- [ ] Note key sampling concepts

### Week 2:
- [ ] DPM-Solver - deep read
- [ ] LCM - skim for ideas
- [ ] Accelerating via Early Stop - CRITICAL READ
- [ ] Document how your approach differs

### Week 3-4 (as needed):
- [ ] Progressive Distillation (if considering training)
- [ ] gDDIM (if exploring schedules)
- [ ] Consistency Models (for perspective)

---

## 🎓 PAPER NOTES TEMPLATE

For each paper, create notes:

```markdown
# [Paper Title]

## Problem They Solve
[What problem?]

## Their Solution
[High-level approach]

## Key Technical Details
[Math, algorithm, architecture]

## Results
[What improvement? On what metrics?]

## Limitations
[What doesn't work?]

## Relevant to My Research
[How does this inform adaptive sampling?]

## Ideas Sparked
[New ideas from reading this]

## Can I Use This?
[Directly applicable? Need to modify?]
```

---

## 🚀 START TODAY

**Your immediate reading assignment:**

1. **Tonight (2 hours)**: Read DDIM abstract, intro, and Section 4
   - Focus: How does DDIM enable step skipping?
   
2. **This week**: Complete DDIM deep read
   - Implement toy DDIM sampler
   - Understand noise schedules

3. **By Week 2**: Read "Accelerating via Early Stop"
   - This is closest to your idea
   - Understand to differentiate your approach

---

## 💬 DISCUSSION RESOURCES

**Communities to join:**
- r/StableDiffusion - Reddit
- Stability AI Discord
- HuggingFace Discord
- AI/ML Twitter (follow authors)

**When you read papers:**
- Tweet about insights (build presence)
- Discuss in Discord
- Ask questions

**Build network early** - it'll help when you publish!

---

**Next**: Start with DDIM tonight. Take notes. Come back with questions! 📚