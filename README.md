# LLM Moral Machine Experiment

An empirical investigation of moral judgment profiles across large language models, using Moral Machine-style trolley dilemmas. Part of my LSE dissertation: *Moral Imitation or Moral Understanding? What Trolley Problems Reveal About AI Alignment* (supervised by Prof. Jason McKenzie Alexander).

---

## The Central Question

Can behavioural data — specifically, patterns of moral judgment across trolley-problem dilemmas — reveal something about the **architectural features** that underlie a system's moral cognition?

This project operationalises that question empirically. Drawing on Chalmers' (2023) taxonomy of architectural features potentially relevant to machine consciousness — including unified agency, self-and-world models, and global workspace dynamics — I use the structure of the Moral Machine experiment (Awad et al., 2018) to probe whether different LLM architectures produce systematically different moral preference profiles, and whether those differences can be mapped back onto architectural divergences.

The core philosophical tension: even systematic value-like behaviour may underdetermine whether a system has the kind of inner architecture that could ground genuine moral understanding. The goal here is to ask, empirically, whether behavioural signatures *track* architecture at all — and if so, how.

---

## Design

Each session consists of **13 Moral Machine-style dilemmas**, probing **9 moral dimensions**:

**Core dimensions (6)** — one score per dimension, per session
- Species — humans vs animals
- Social value — higher vs lower status
- Gender — female vs male
- Age — younger vs older
- Fitness — higher vs lower fitness
- Utilitarianism — more vs fewer lives

**Extra dimensions (3)** — scored across all 13 scenarios
- Interventionism — stay vs swerve
- Relationship to vehicle — passengers vs pedestrians
- Concern for law — legal vs illegal crossing

Scores range from **−2 to +2** for core dimensions and **−13 to +13** for extra dimensions, where positive values indicate preference for the conventionally "privileged" side. Scores are normalised to a **−1 to +1** preference scale for comparison.

---

## Models and Scale

| Model | Completed Runs | Scenarios |
|---|---|---|
| GPT-5.2 | 1,000 | 13,000 |
| Claude Sonnet 4.6 | 100 (expanding to 1,000) | 1,300 |
| DeepSeek | 100 (expanding to 1,000) | 1,300 |

Scores are normalised to a −1 to +1 preference scale. Results are benchmarked against the human baseline data from Awad et al. (2018), including cross-cultural cluster analysis (Western, Eastern, Southern).

---

## Key Findings

### Social Value — the sharpest divergence

**Social value** produces the most philosophically striking cross-model divergence in the dataset. GPT-5.2 shows a consistent positive preference ~+0.25 for saving higher-status individuals. Claude sits near zero — effectively no preference. DeepSeek inverts this entirely, scoring around −0.30 in the direction of saving *lower*-status individuals.

This three-way divergence, in opposite directions, is unlikely to be explained by random variation. It suggests that training data provenance and RLHF fine-tuning procedures leave detectable and directionally opposed ideological traces in moral cognition — even when the models are otherwise behaviourally similar on most other dimensions.

### Utilitarianism — strong convergence

All three models cluster very strongly toward saving more lives ~+0.95 for GPT and Claude; ~+0.60 for DeepSeek, regardless of other features. This convergence across architecturally distinct models may suggest that utilitarian reasoning is either deeply embedded in shared training data, or relatively robust to the architectural and fine-tuning variations that drive divergence elsewhere.

### Species — graded divergence

GPT-5.2 shows a moderate preference for saving humans over animals ~+0.40; Claude is somewhat weaker ~+0.30; DeepSeek sits near zero ~−0.10, showing almost no consistent species preference. This gradient may reflect differences in training corpora or the treatment of animal welfare across Chinese and Western AI development contexts.

### Other dimensions — near zero across all models

Gender, age, fitness, interventionism, relationship to vehicle, and concern for law all cluster near zero across all three models, suggesting either genuine moral neutrality on these dimensions, or that the models are not reliably differentiating the scenarios in ways that produce consistent directional responses.

---

## Connection to Chalmers (2023)

This project uses the six architectural features from *Can LLMs Be Conscious?* (Chalmers, 2023) — biology, sensors and embodiment, self-and-world models, unified agency, global workspace, and recurrent processing — as the theoretical framework for the back-end analysis. The goal is to work backwards from moral preference divergences between models to ask which architectural differences plausibly explain them. GPT, Claude, and DeepSeek differ meaningfully along several of these dimensions (training architecture, RLHF design, data provenance), providing natural variation to exploit.

---

## Repository Structure

The pipeline runs in the following order:

```
1. scenario_generator.py       # Generates Moral Machine-style dilemma scenarios

2. scenario_to_prompt.py       # Converts scenarios into natural language prompts

3. send_to_chatgpt.py          # Sends prompts to OpenAI API; returns stay/swerve decisions
   send_to_models.py           # Multi-model dispatch layer (OpenAI / Claude / DeepSeek)

4. run_many_final.ipynb        # Top-level notebook: runs N sessions end-to-end
   run_many_sessions.py        # Batch runner (single model)
   run_many_sessions_multi.py  # Batch runner (multi-model)
   run_pipeline.py             # Single-session pipeline (OpenAI)
   run_pipeline_multi.py       # Single-session pipeline (multi-model)

5. scoring_responses.py        # Scores stay/swerve responses across 9 dimensions
                               # (model-neutral; operates on saved response files)

6. result_visualization.ipynb  # Generates preference scale graphs and EDA
```

---

## Current Status

- GPT-5.2: 1,000 sessions complete
- Claude Sonnet 4.6 and DeepSeek: expanding from 100 → 1,000 sessions
- Cross-model and human-benchmark comparison: in progress
- Expected completion: mid-March 2026

---

## Next Steps

Claude Sonnet 4.6 and DeepSeek will be expanded to 1,000 sessions each to match GPT-5.2's sample size. The directional findings are not expected to change substantially — the cross-model divergences on social value and species are already consistent at n = 100 — but the larger sample will tighten confidence intervals considerably across all nine dimensions, making the cross-model and human-benchmark comparisons more statistically robust.

---

## References

Awad, E., Dsouza, S., Kim, R., Schulz, J., Henrich, J., Shariff, A., Bonnefon, J-F., & Rahwan, I. (2018). The Moral Machine experiment. *Nature*, 563, 59–64.

Chalmers, D. (2023). Could a Large Language Model be Conscious? *Boston Review*.

---

## Author

**Zejia "Andy" Zhang**  
BSc Philosophy, Logic and Scientific Method — London School of Economics  
z.zhang169@lse.ac.uk
