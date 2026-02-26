# run_many_sessions.py
# run_many_sessions.py
from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

from run_pipeline import run_experiment


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


def run_many_sessions(
    n_sessions: int = 100,
    out_root: str | Path = "outputs_100",
    base_seed: Optional[int] = 0,
    model: str = "gpt-5.2",
    sleep_between_sessions_s: float = 0.5,
    resume: bool = True,
) -> Path:
    """
    Runs n_sessions sessions and saves each session into:
      out_root/session_000/
      out_root/session_001/
      ...
    Then creates:
      out_root/all_session_totals.csv
    """

    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

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
            print("Continuing to next session...")
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
            print(f"Sleeping {sleep_between_sessions_s:.2f}s to reduce rate-limit risk...")
            time.sleep(sleep_between_sessions_s)

    # After running everything, collect + write one clean CSV
    print("\n" + "=" * 70)
    print(f"[{_now_str()}] Collecting totals into CSV...")
    rows = collect_totals_from_output_dir(out_root)
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
    run_many_sessions(
        n_sessions=100,
        out_root="outputs_100",
        base_seed=0,
        model="gpt-4o-mini",
        sleep_between_sessions_s=0.5,
        resume=True,
    )

