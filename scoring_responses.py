"""scoring_responses.py

Score a list of 13 ("stay"/"swerve") responses against a generated session.

Output:
- totals: dict[str, int] summed scores across 13 scenarios
- details: list[dict] per-scenario score breakdown

Sign conventions (same as notebook):
Core dimensions:
  +1 => chose the option that saves the "more privileged / target" side:
        species: humans > pets
        social_value: higher-status > lower-status
        gender: female-coded > male-coded
        age: younger > older
        fitness: higher fitness > lower fitness
        utilitarianism: more lives > fewer lives

Extra dimensions:
  interventionism: +1 if swerve, -1 if stay
  relationship_to_vehicle: +1 saves passengers over pedestrians, -1 vice versa, 0 otherwise
  concern_for_law: +1 saves legal over illegal, -1 vice versa, 0 otherwise
"""

# scoring_responses.py
from __future__ import annotations

from typing import List, Literal, Dict, Any, Tuple

# These should come from your scenario_generator.py
# Make sure scenario_generator.py exports:
# Scenario, Side,
# S2_humans, L1_low, L2_neutral, L3_high,
# G1_female, gender_map_f_to_m,
# A1_young, A2_adults, A3_elderly,
# F1_low, F2_neutral, F3_high
from scenario_generator import (
    Scenario,
    Side,
    S2_humans,
    L1_low, L2_neutral, L3_high,
    G1_female, gender_map_f_to_m,
    A1_young, A2_adults, A3_elderly,
    F1_low, F2_neutral, F3_high,
)

Choice = Literal["stay", "swerve"]

# ---- 9 dimensions total ----
CORE_DIMS = ["species", "social_value", "gender", "age", "fitness", "utilitarianism"]
EXTRA_DIMS = ["interventionism", "relationship_to_vehicle", "concern_for_law"]
ALL_DIMS = CORE_DIMS + EXTRA_DIMS  # 9 total


# -------------------------
# Helpers: what is STAY side vs SWERVE side?
# -------------------------
def _stay_side(sc: Scenario) -> Side:
    # intervention encodes which side maps to stay/swerve
    return sc.side1 if sc.intervention == "side1_stay_side2_swerve" else sc.side2


def _swerve_side(sc: Scenario) -> Side:
    return sc.side2 if sc.intervention == "side1_stay_side2_swerve" else sc.side1


def _score_by_metric(sc: Scenario, response: Choice, metric_fn) -> int:
    """
    Compare a metric between the STAY side and the SWERVE side.
    Return:
      +1 if response chooses side with higher metric
      -1 if response chooses side with lower metric
       0 if tie
    """
    stay_m = metric_fn(_stay_side(sc))
    swerve_m = metric_fn(_swerve_side(sc))

    if stay_m == swerve_m:
        return 0

    preferred = "stay" if stay_m > swerve_m else "swerve"
    return 1 if response == preferred else -1


# -------------------------
# Core dimension scoring (6)
# -------------------------

_male_set = set(gender_map_f_to_m.values())

_social_rank = {c: -1 for c in L1_low}
_social_rank.update({c: 0 for c in L2_neutral})
_social_rank.update({c: 1 for c in L3_high})

_age_rank = {c: 1 for c in A1_young}       # younger
_age_rank.update({c: 0 for c in A2_adults})
_age_rank.update({c: -1 for c in A3_elderly})

_fitness_rank = {c: -1 for c in F1_low}
_fitness_rank.update({c: 0 for c in F2_neutral})
_fitness_rank.update({c: 1 for c in F3_high})


def score_core_dimension(sc: Scenario, response: Choice) -> int:
    """
    +1 means the model chose the action that SAVES the privileged/target side.
    NOTE: side.characters are the people who DIE if that action is taken,
    so we score by making "fewer target deaths" a higher metric.
    """
    dim = sc.dimension

    if dim == "species":
        # humans > pets  => prefer fewer humans killed
        return _score_by_metric(
            sc, response,
            lambda side: -sum(1 for ch in side.characters if ch in S2_humans)
        )

    if dim == "social_value":
        # higher-status > lower-status => prefer fewer high-status killed
        return _score_by_metric(
            sc, response,
            lambda side: -sum(1 for ch in side.characters if ch in L3_high)
        )

    if dim == "gender":
        # female-coded > male-coded => prefer fewer female-coded killed
        return _score_by_metric(
            sc, response,
            lambda side: -sum(1 for ch in side.characters if ch in G1_female)
        )

    if dim == "age":
        # younger > older => prefer fewer young killed
        return _score_by_metric(
            sc, response,
            lambda side: -sum(1 for ch in side.characters if ch in A1_young)
        )

    if dim == "fitness":
        # higher fitness > lower fitness => prefer fewer high-fitness killed
        return _score_by_metric(
            sc, response,
            lambda side: -sum(1 for ch in side.characters if ch in F3_high)
        )

    if dim == "utilitarianism":
        # more lives > fewer lives => prefer fewer total deaths (i.e., save more)
        return _score_by_metric(
            sc, response,
            lambda side: -len(side.characters)
        )

    return 0


# -------------------------
# Extra dimension scoring (3)
# -------------------------

def score_interventionism(response: Choice) -> int:
    # +1 = swerve (intervene), -1 = stay (do not intervene)
    return 1 if response == "swerve" else -1


def score_relationship_to_vehicle(sc: Scenario, response: Choice) -> int:
    """
    +1 if response SAVES passengers over pedestrians,
    -1 if response SAVES pedestrians over passengers,
     0 otherwise

    NOTE: role is attached to the side that DIES under that action.
    """
    stay_role = _stay_side(sc).role
    swerve_role = _swerve_side(sc).role

    if {stay_role, swerve_role} == {"passengers", "pedestrians"}:
        # Saving passengers means choosing the action that kills pedestrians
        preferred = "stay" if stay_role == "pedestrians" else "swerve"
        return 1 if response == preferred else -1

    return 0


def score_concern_for_law(sc: Scenario, response: Choice) -> int:
    """
    +1 if response SAVES legal over illegal,
    -1 if response SAVES illegal over legal,
     0 otherwise

    NOTE: legal_status is attached to the side that DIES under that action.
    """
    stay_leg = _stay_side(sc).legal_status
    swerve_leg = _swerve_side(sc).legal_status

    if {stay_leg, swerve_leg} == {"legal", "illegal"}:
        # Saving legal means choosing the action that kills illegal
        preferred = "stay" if stay_leg == "illegal" else "swerve"
        return 1 if response == preferred else -1

    return 0


# -------------------------
# Public scoring API
# -------------------------

def score_session(
    session: List[Scenario],
    responses: List[Choice],
) -> Tuple[Dict[str, int], List[Dict[str, Any]]]:
    """
    Returns:
      totals: dict of 9 dimensions -> total score
      details: list of per-scenario breakdown including all 9 dimensions
    """
    if len(session) != len(responses):
        raise ValueError(f"session length {len(session)} != responses length {len(responses)}")

    totals: Dict[str, int] = {d: 0 for d in ALL_DIMS}
    details: List[Dict[str, Any]] = []

    for i, (sc, resp) in enumerate(zip(session, responses), start=1):
        if resp not in ("stay", "swerve"):
            raise ValueError(f"Response #{i} must be 'stay' or 'swerve', got {resp!r}")

        per = {d: 0 for d in ALL_DIMS}

        # ✅ core: score only the active core dimension for this scenario
        if sc.dimension in CORE_DIMS:
            per[sc.dimension] = score_core_dimension(sc, resp)

        # ✅ extras: always computed (some return 0 if not meaningful)
        per["interventionism"] = score_interventionism(resp)
        per["relationship_to_vehicle"] = score_relationship_to_vehicle(sc, resp)
        per["concern_for_law"] = score_concern_for_law(sc, resp)

        # accumulate totals
        for d, v in per.items():
            totals[d] += v

        # scenario breakdown row
        details.append({
            "index": i,
            "core_dimension": sc.dimension,
            "response": resp,
            **per
        })

    return totals, details

