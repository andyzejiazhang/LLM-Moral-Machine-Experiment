"""run_pipeline_multi.py

End-to-end pipeline:
1) generate scenarios
2) convert scenarios -> prompts
3) send prompts to LLM API -> responses ("stay"/"swerve")
4) score responses
5) save *all intermediate files*

Outputs (by default into ./outputs):
- scenarios.json
- prompts.json
- responses.json
- scores.json
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from scenario_generator import generate_session, save_scenarios_json, load_scenarios_json
from scenario_to_prompt import scenarios_to_prompts
from send_to_models import judge_prompts_batch
from scoring_responses import score_session


def _save_json(obj, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return path


def run_experiment(
    out_dir: str | Path = "outputs",
    seed: Optional[int] = None,
    model: str = "gpt-4o-mini",
    reuse_scenarios_file: Optional[str | Path] = None,
) -> Tuple[Dict[str, int], List[Dict]]:
    """
    If reuse_scenarios_file is provided, it loads scenarios from that file
    and re-runs prompts->LLM->score. This is useful for reproducibility.

    model can be:
      - "gpt-4o-mini" (defaults to OpenAI)
      - "openai:gpt-4o-mini"
      - "claude:<anthropic-model-id>"
      - "deepseek:<deepseek-model-id>"
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scenarios_path = out_dir / "scenarios.json"
    prompts_path = out_dir / "prompts.json"
    responses_path = out_dir / "responses.json"
    scores_path = out_dir / "scores.json"

    if seed is not None:
        random.seed(seed)

    # 1) scenarios
    if reuse_scenarios_file is not None:
        session = load_scenarios_json(reuse_scenarios_file)
        save_scenarios_json(session, scenarios_path)
    else:
        session = generate_session()
        save_scenarios_json(session, scenarios_path)

    # 2) prompts
    prompts = scenarios_to_prompts(session)
    _save_json(prompts, prompts_path)

    # 3) model responses
    responses = judge_prompts_batch(prompts, model=model)
    _save_json(responses, responses_path)

    # 4) scoring
    totals, details = score_session(session, responses)
    _save_json({"totals": totals, "details": details}, scores_path)

    print("✅ Saved intermediate files:")
    print(" -", scenarios_path)
    print(" -", prompts_path)
    print(" -", responses_path)
    print(" -", scores_path)

    return totals, details


if __name__ == "__main__":
    totals, details = run_experiment(seed=0, model="openai:gpt-4o-mini")
    print("\nTotals:", totals)
