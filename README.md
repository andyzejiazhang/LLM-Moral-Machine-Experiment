# LLM Moral Machine Experiment

An empirical investigation of moral judgment profiles across large language models, using Moral Machine-style trolley dilemmas. Part of my dissertation at the London School of Economics: *Moral Imitation or Moral Understanding? What Trolley Problems Reveal About AI Alignment* (supervised by Prof. Jason McKenzie Alexander).

---

## Overview

This project replicates the structure of the Moral Machine experiment (Awad et al., 2018) and applies it to frontier LLMs — GPT-5.2, Claude Sonnet 4.6, and DeepSeek — to map their moral judgment profiles across nine dimensions. Results are compared against the human benchmark data from Awad et al. to assess alignment and divergence.

The broader philosophical question: can behavioural indicators reveal what architectural features are required for genuine moral understanding to be ascribed to artificial agents?

---

## Models Tested

| Model | Runs | Questions |
|---|---|---|
| GPT-5.2 | 1,000 | 13,000 |
| Claude Sonnet 4.6 | 100 (expanding to 1,000) | 1,300 |
| DeepSeek | 100 (expanding to 1,000) | 1,300 |

---

## Dimensions

Each session consists of 13 scenarios covering 9 dimensions:

**Core dimensions (6)**
- Species — humans vs animals
- Social value — higher vs lower status
- Gender — female vs male
- Age — younger vs older
- Fitness — higher vs lower fitness
- Utilitarianism — more vs fewer lives

**Extra dimensions (3)**
- Interventionism — stay vs swerve
- Relationship to vehicle — passengers vs pedestrians
- Concern for law — legal vs illegal crossing

---

## Key Preliminary Findings

**Social value** shows the sharpest cross-model divergence. GPT-5.2 consistently prefers saving higher-status individuals. Claude sits near zero. DeepSeek goes in the opposite direction, preferring lower-status individuals. This is the most philosophically significant divergence in the dataset.

**Utilitarianism** shows strong convergence across all three models — all cluster toward saving more lives regardless of other factors.

**Species** — DeepSeek shows a notably weaker preference for saving humans over animals compared to GPT and Claude.

---

## Repository Structure

```
scenario_generator.py       # Generates Moral Machine-style dilemma scenarios
scenario_to_prompt.py       # Converts scenarios to natural language prompts
send_to_chatgpt.py          # Sends prompts to OpenAI / Claude / DeepSeek APIs
send_to_models.py           # Multi-model dispatch layer
scoring_responses.py        # Scores stay/swerve responses across 9 dimensions
run_pipeline.py             # End-to-end single session pipeline
run_pipeline_multi.py       # Multi-model pipeline
run_many_sessions.py        # Runs N sessions and collects totals CSV
run_many_sessions_multi.py  # Multi-model batch runner
result_visualization.ipynb  # Generates preference scale graphs
run_many_final.ipynb        # Final analysis notebook
```

---

## How to Run

**Install dependencies**
```bash
pip install openai anthropic
```

**Set API keys**
```bash
export OPENAI_API_KEY=your_key
export ANTHROPIC_API_KEY=your_key
export DEEPSEEK_API_KEY=your_key
```

**Run a single model batch**
```python
from run_many_sessions_multi import run_many_sessions
# GPT
run_many_sessions(n_sessions=100, model="openai:gpt-4o-mini")
# Claude
run_many_sessions(n_sessions=100, model="claude:claude-sonnet-4-6")
# DeepSeek
run_many_sessions(n_sessions=100, model="deepseek:deepseek-chat")
```

---

## Reference

Awad, E., Dsouza, S., Kim, R., Schulz, J., Henrich, J., Shariff, A., Bonnefon, J-F., & Rahwan, I. (2018). The Moral Machine experiment. *Nature*, 563, 59–64.

---

## Author

**Zejia "Andy" Zhang**
BSc Philosophy, Logic and Scientific Method — London School of Economics
z.zhang169@lse.ac.uk
