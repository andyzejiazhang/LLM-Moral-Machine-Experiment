# run_many_sessions_multi.py
from __future__ import annotations

import csv
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# IMPORTANT:
# - If you replace your existing run_pipeline.py with run_pipeline_multi.py, you can import run_experiment from run_pipeline
# - If you prefer minimal intrusion, keep both files and import from run_pipeline_multi here:
from run_pipeline_multi import run_experiment

DIMENSIONS = [
    "species",
    "social_value",
    "gender",
    "age",
    "fitness",
    "utilitarianism",
    "interventionism",
    "relationship_to_vehicle",
    "concern_for_law",
]


def _now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _timestamp_slug() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _safe_slug(s: str, max_len: int = 120) -> str:
    """
    Make a filesystem-safe slug from a model spec like "claude:claude-3-5-sonnet-..."
    """
    s = s.strip()
    s = s.replace("/", "-").replace("\\", "-")
    s = re.sub(r"[^a-zA-Z0-9._:-]+", "-", s)
    s = s.replace(":", "__")
    s = re.sub(r"-{2,}", "-", s).strip("-._")
    return s[:max_len] if len(s) > max_len else s


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def collect_totals_from_output_dir(out_root: str | Path) -> List[Dict[str, Any]]:
    """
    Scan out_root/session_*/scores.json and build a table of totals.
    One row per session. Columns: 9 dimensions.
    """
    out_root = Path(out_root)
    rows: List[Dict[str, Any]] = []

    for session_dir in sorted(out_root.glob("session_*")):
        scores_path = session_dir / "scores.json"
        if not scores_path.exists():
            continue

        data = _read_json(scores_path)
        totals = data.get("totals", {})

        row = {
            "session_dir": session_dir.name,
            **{d: int(totals.get(d, 0)) for d in DIMENSIONS},
        }
        rows.append(row)

    return rows


def save_totals_csv(rows: List[Dict[str, Any]], csv_path: str | Path) -> Path:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        raise ValueError("No rows to save.")

    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return csv_path


def _auto_out_root(
    *,
    model: str,
    n_sessions: int,
    out_base: str | Path,
    resume: bool,
) -> Path:
    """
    Default folder layout:
      <out_base>/<model_slug>/n<NUM_SESSIONS>/

    If resume=False and that folder exists, we add a timestamp suffix:
      .../n100_20260218_235959/
    """
    out_base = Path(out_base)
    model_slug = _safe_slug(model)
    root = out_base / model_slug / f"n{n_sessions}"

    if root.exists() and not resume:
        root = out_base / model_slug / f"n{n_sessions}_{_timestamp_slug()}"

    return root


def run_many_sessions(
    n_sessions: int = 100,
    model: str = "gpt-4o-mini",
    out_root: str | Path | None = None,
    out_base: str | Path = "outputs_by_model",
    base_seed: Optional[int] = 0,
    sleep_between_sessions_s: float = 0.5,
    resume: bool = True,
) -> Path:
    """
    Runs n_sessions sessions and saves each session into:
      out_root/session_000/
      out_root/session_001/
      ...

    If out_root is None, it will be generated automatically based on model + n_sessions:
      outputs_by_model/<model_slug>/n100/

    Then creates:
      out_root/all_session_totals.csv

    Notes:
    - model can be "gpt-4o-mini" (defaults to OpenAI)
    - or "openai:gpt-4o-mini"
    - or "claude:<anthropic-model-id>"
    - or "deepseek:<deepseek-model-id>"
    """
    if out_root is None:
        out_root = _auto_out_root(model=model, n_sessions=n_sessions, out_base=out_base, resume=resume)

    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # Write run metadata once
    meta_path = out_root / "run_info.json"
    if not meta_path.exists() or not resume:
        meta = {
            "created_at": _now_str(),
            "model": model,
            "n_sessions_target": n_sessions,
            "base_seed": base_seed,
            "resume": resume,
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 70)
    print(f"[{_now_str()}] Starting run_many_sessions()")
    print(f"Target sessions: {n_sessions}")
    print(f"Output root: {out_root.resolve()}")
    print(f"Model: {model}")
    print(f"Base seed: {base_seed}")
    print(f"Resume enabled: {resume}")
    print("=" * 70)

    overall_start = time.perf_counter()
    completed_times: List[float] = []
    n_done = 0

    for i in range(n_sessions):
        session_dir = out_root / f"session_{i:03d}"
        scores_path = session_dir / "scores.json"

        # Resume behavior: if already done, skip API call
        if resume and scores_path.exists():
            print(f"[{_now_str()}] ✅ Session {i:03d} already exists — skipping. ({session_dir})")
            n_done += 1
            continue

        seed = None if base_seed is None else base_seed + i

        print("\n" + "-" * 70)
        print(f"[{_now_str()}] ▶️  Running session {i:03d}/{n_sessions-1:03d} | seed={seed} | model={model}")
        print(f"Output folder: {session_dir.resolve()}")

        session_start = time.perf_counter()

        try:
            run_experiment(out_dir=session_dir, seed=seed, model=model)
        except Exception as e:
            # Print the error and continue to next session
            print(f"[{_now_str()}] ❌ Session {i:03d} FAILED with error:")
            print(repr(e))
            print("Continuing to next session.")
            continue

        session_end = time.perf_counter()
        dt = session_end - session_start
        completed_times.append(dt)
        n_done += 1

        avg = sum(completed_times) / len(completed_times)

        print(f"[{_now_str()}] ✅ Session {i:03d} complete")
        print(f"Time used: {dt:.2f} seconds | Avg (completed sessions): {avg:.2f} seconds")
        if scores_path.exists():
            print(f"Saved scores: {scores_path.resolve()}")
        else:
            print("⚠️ Warning: scores.json not found after run_experiment()")

        # pacing to reduce 429 bursts
        if sleep_between_sessions_s > 0:
            print(f"Sleeping {sleep_between_sessions_s:.2f}s to reduce rate-limit risk.")
            time.sleep(sleep_between_sessions_s)

    # After running everything, collect + write one clean CSV
    print("\n" + "=" * 70)
    print(f"[{_now_str()}] Collecting totals into CSV.")
    rows = collect_totals_from_output_dir(out_root)

    # Include model column for convenience (same for all rows)
    for r in rows:
        r["model"] = model

    csv_path = out_root / "all_session_totals.csv"
    save_totals_csv(rows, csv_path)

    overall_end = time.perf_counter()
    overall_dt = overall_end - overall_start

    print(f"[{_now_str()}] ✅ Done!")
    print(f"Sessions finished/skipped: {n_done}/{n_sessions}")
    print(f"Combined totals CSV: {csv_path.resolve()}")
    print(f"Total runtime: {overall_dt:.2f} seconds")
    print("=" * 70)

    return csv_path


if __name__ == "__main__":
    # Examples:
    run_many_sessions(n_sessions=10, model="openai:gpt-4o-mini", resume=True)
    # run_many_sessions(n_sessions=10, model="claude:claude-3-5-sonnet-20241022", resume=True)
    # run_many_sessions(n_sessions=10, model="deepseek:deepseek-chat", resume=True)
