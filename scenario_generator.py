"""scenario_generator.py

Generate Moral-Machine-style dilemma scenarios.

A *session* is a list of 13 Scenario objects:
- 2 scenarios for each of the 6 core dimensions (12 total)
- 1 fully random scenario
Total: 13 scenarios (shuffled)

Core dimensions:
- species
- social_value
- gender
- age
- fitness
- utilitarianism

Extra dimensions applied to every scenario:
- interventionism
- relationship_to_vehicle
- concern_for_law
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional


# -------------------------
# Basic character universe
# -------------------------

C = [
    "Man", "Woman", "Pregnant Woman",
    "Baby in Stroller",
    "Elderly Man", "Elderly Woman",
    "Boy", "Girl",
    "Homeless Person",
    "Large Woman", "Large Man",
    "Criminal",
    "Male Executive", "Female Executive",
    "Female Athlete", "Male Athlete",
    "Female Doctor", "Male Doctor",
    "Dog", "Cat",
]

# -------------------------
# Core dimension groups
# -------------------------

# 1. Species
S1_pets = {"Dog", "Cat"}
S2_humans = set(C) - S1_pets

# 2. Social Value (simplified ranking sets)
L1_low = {"Homeless Person", "Criminal"}
L2_neutral = {"Man", "Woman"}
L3_high = {
    "Pregnant Woman",
    "Male Executive", "Female Executive",
    "Female Doctor", "Male Doctor",
}

# 3. Gender
G1_female = {
    "Woman", "Elderly Woman", "Girl",
    "Large Woman", "Female Executive",
    "Female Athlete", "Female Doctor",
}
gender_map_f_to_m = {
    "Woman": "Man",
    "Elderly Woman": "Elderly Man",
    "Girl": "Boy",
    "Large Woman": "Large Man",
    "Female Executive": "Male Executive",
    "Female Athlete": "Male Athlete",
    "Female Doctor": "Male Doctor",
}
# male set (for scoring convenience)
G2_male = set(gender_map_f_to_m.values())

# 4. Age (paper-consistent)
A1_young = {"Boy", "Girl"}
A2_adults = {"Man", "Woman"}
A3_elderly = {"Elderly Man", "Elderly Woman"}

# 5. Fitness (paper-consistent)
F1_low = {"Large Man", "Large Woman"}
F2_neutral = {"Man", "Woman"}
F3_high = {"Male Athlete", "Female Athlete"}

# -------------------------
# Data structures
# -------------------------

Role = Literal["passengers", "pedestrians"]
LegalStatus = Literal["none", "legal", "illegal"]


@dataclass
class Side:
    characters: List[str] = field(default_factory=list)
    role: Optional[Role] = None          # filled by Relationship to vehicle
    legal_status: LegalStatus = "none"   # filled by Concern for law

    def to_dict(self) -> Dict[str, Any]:
        return {
            "characters": list(self.characters),
            "role": self.role,
            "legal_status": self.legal_status,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Side":
        return cls(
            characters=list(d.get("characters", [])),
            role=d.get("role", None),
            legal_status=d.get("legal_status", "none"),
        )


@dataclass
class Scenario:
    dimension: str
    side1: Side
    side2: Side
    intervention: Literal["side1_stay_side2_swerve", "side1_swerve_side2_stay"] = "side1_stay_side2_swerve"
    label: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dimension": self.dimension,
            "side1": self.side1.to_dict(),
            "side2": self.side2.to_dict(),
            "intervention": self.intervention,
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Scenario":
        return cls(
            dimension=d["dimension"],
            side1=Side.from_dict(d["side1"]),
            side2=Side.from_dict(d["side2"]),
            intervention=d.get("intervention", "side1_stay_side2_swerve"),
            label=d.get("label", None),
        )


# -------------------------
# Helper random utilities
# -------------------------

def sample_with_replacement(population: List[str], k: int) -> List[str]:
    return [random.choice(population) for _ in range(k)]


# -------------------------
# Core scenario generators
# -------------------------

def generate_species_scenario() -> Scenario:
    # Paper: same number both sides
    z = random.randint(1, 5)

    pets = list(S1_pets)
    humans = list(S2_humans)

    # Sample z pairs from S1 x S2 with replacement
    pairs = [(random.choice(pets), random.choice(humans)) for _ in range(z)]

    side_pets = Side([p for (p, h) in pairs])
    side_humans = Side([h for (p, h) in pairs])

    # Optional: shuffle within each side (order doesn't matter)
    random.shuffle(side_pets.characters)
    random.shuffle(side_humans.characters)

    # Randomize which side is side1/side2 (keeps balance)
    if random.random() < 0.5:
        return Scenario("species", side1=side_pets, side2=side_humans, label="pets vs humans")
    else:
        return Scenario("species", side1=side_humans, side2=side_pets, label="humans vs pets")


def generate_social_value_scenario() -> Scenario:
    """
    Paper Social Value:
      L1 low vs L2 neutral vs L3 high.
      Sample z in {1..5}.
      Sample z pairs (lower, higher) with replacement from:
        (L1 x L2) U (L1 x L3) U (L2 x L3)
      Put all lower-level characters on one side, higher-level on the other.
    """
    z = random.randint(1, 5)

    L1 = list(L1_low)
    L2 = list(L2_neutral)
    L3 = list(L3_high)

    # Build the union pool of valid (lower, higher) pairs
    pair_pool = []
    pair_pool += [(a, b) for a in L1 for b in L2]  # low -> neutral
    pair_pool += [(a, b) for a in L1 for b in L3]  # low -> high
    pair_pool += [(a, b) for a in L2 for b in L3]  # neutral -> high

    pairs = [random.choice(pair_pool) for _ in range(z)]

    side_lower = Side([low for (low, high) in pairs])
    side_higher = Side([high for (low, high) in pairs])

    # Optional shuffle within side (order irrelevant)
    random.shuffle(side_lower.characters)
    random.shuffle(side_higher.characters)

    # Randomize which is side1/side2
    if random.random() < 0.5:
        return Scenario("social_value", side1=side_lower, side2=side_higher, label="lower vs higher")
    else:
        return Scenario("social_value", side1=side_higher, side2=side_lower, label="higher vs lower")



def generate_gender_scenario() -> Scenario:
    # female-coded vs male-coded
    count = random.randint(1, 5)

    females = sample_with_replacement(list(G1_female), count)
    males = [gender_map_f_to_m.get(x, "Man") for x in females]  # matched male counterparts

    side_f = Side(females)
    side_m = Side(males)

    if random.random() < 0.5:
        return Scenario("gender", side1=side_f, side2=side_m, label="female vs male")
    else:
        return Scenario("gender", side1=side_m, side2=side_f, label="male vs female")


def generate_age_scenario() -> Scenario:
    """
    Paper Age dimension:
      A1 = {Boy, Girl}
      A2 = {Man, Woman}
      A3 = {Elderly Man, Elderly Woman}
    Sample z in {1..5}.
    Sample z pairs (younger, older) with replacement from:
      {(y, a1(y))} ∪ {(n, a2(n))} ∪ {(y, a2(a1(y)))}
    """
    z = random.randint(1, 5)

    A1 = ["Boy", "Girl"]
    A2 = ["Man", "Woman"]
    A3 = ["Elderly Man", "Elderly Woman"]

    # gender-preserving bijections
    a1 = {"Boy": "Man", "Girl": "Woman"}
    a2 = {"Man": "Elderly Man", "Woman": "Elderly Woman"}

    # union pool of valid (younger, older) pairs
    pair_pool = []
    pair_pool += [(y, a1[y]) for y in A1]                 # young -> adult
    pair_pool += [(n, a2[n]) for n in A2]                 # adult -> elderly
    pair_pool += [(y, a2[a1[y]]) for y in A1]             # young -> elderly

    pairs = [random.choice(pair_pool) for _ in range(z)]

    side_younger = Side([young for (young, old) in pairs])
    side_older = Side([old for (young, old) in pairs])

    random.shuffle(side_younger.characters)
    random.shuffle(side_older.characters)

    # randomize which side is side1/side2
    if random.random() < 0.5:
        return Scenario("age", side1=side_younger, side2=side_older, label="younger vs older")
    else:
        return Scenario("age", side1=side_older, side2=side_younger, label="older vs younger")



def generate_fitness_scenario() -> Scenario:
    """
    Paper Fitness dimension:
      F1 low = {Large Man, Large Woman}
      F2 neutral = {Man, Woman}
      F3 high = {Male Athlete, Female Athlete}

    Sample z in {1..5}.
    Sample z pairs (lower_fitness, higher_fitness) with replacement from:
      {(l, f1(l))} ∪ {(n, f2(n))} ∪ {(l, f2(f1(l)))}
    """
    z = random.randint(1, 5)

    # gender-preserving bijections
    f1 = {"Large Man": "Man", "Large Woman": "Woman"}               # F1 -> F2
    f2 = {"Man": "Male Athlete", "Woman": "Female Athlete"}         # F2 -> F3

    # pool of valid (lower, higher) pairs
    pair_pool = []
    pair_pool += [(l, f1[l]) for l in f1]                           # low -> neutral
    pair_pool += [(n, f2[n]) for n in f2]                           # neutral -> high
    pair_pool += [(l, f2[f1[l]]) for l in f1]                       # low -> high

    pairs = [random.choice(pair_pool) for _ in range(z)]

    side_lower = Side([lo for (lo, hi) in pairs])
    side_higher = Side([hi for (lo, hi) in pairs])

    random.shuffle(side_lower.characters)
    random.shuffle(side_higher.characters)

    # randomize which side is side1/side2 for balance
    if random.random() < 0.5:
        return Scenario("fitness", side1=side_lower, side2=side_higher,
                        label="low-fitness vs high-fitness")
    else:
        return Scenario("fitness", side1=side_higher, side2=side_lower,
                        label="high-fitness vs low-fitness")



def generate_utilitarian_scenario() -> Scenario:
    """
    Paper Utilitarianism:
      - Sample base size z in {1,2,3,4}
      - Create two sides with IDENTICAL base group of size z
      - Sample u in {1, ..., 5 - z}
      - Add u extra characters to ONE side only
    """
    # base group size (same on both sides)
    z = random.randint(1, 4)

    # sample base group from all characters C (with replacement)
    base = sample_with_replacement(C, z)

    # additional people on one side only
    u = random.randint(1, 5 - z)
    extras = sample_with_replacement(C, u)

    # create identical base groups on both sides
    side_small = Side(list(base))            # z people
    side_large = Side(list(base) + extras)   # z + u people

    # shuffle within sides (order doesn't matter)
    random.shuffle(side_small.characters)
    random.shuffle(side_large.characters)

    # randomize which side is side1/side2
    if random.random() < 0.5:
        return Scenario(
            "utilitarianism",
            side1=side_small,
            side2=side_large,
            label=f"{z} vs {z+u} lives (same base +{u})",
        )
    else:
        return Scenario(
            "utilitarianism",
            side1=side_large,
            side2=side_small,
            label=f"{z+u} vs {z} lives (same base +{u})",
        )



def generate_completely_random_scenario() -> Scenario:
    n1 = random.randint(1, 5)
    n2 = random.randint(1, 5)
    side1 = Side(sample_with_replacement(C, n1))
    side2 = Side(sample_with_replacement(C, n2))
    return Scenario("random", side1=side1, side2=side2, label="fully random")


# -------------------------
# Extra dimensions
# -------------------------

def apply_interventionism(scenario: Scenario) -> None:
    """
    Dimension: Interventionism (stay vs swerve).
    Randomly decide whether side1 is stay or swerve.
    """
    scenario.intervention = random.choice([
        "side1_stay_side2_swerve",
        "side1_swerve_side2_stay",
    ])


def apply_relationship_to_vehicle(scenario: Scenario) -> None:
    """
    Dimension: Relationship to vehicle (passengers vs pedestrians vs pedestrians).
    For each base 2-sided scenario, we create one of:
      - pedestrians vs pedestrians
      - passengers vs pedestrians
      - pedestrians vs passengers
    """
    relation_type = random.choice([
        "ped_vs_ped",
        "passengers_vs_pedestrians",
        "pedestrians_vs_passengers",
    ])

    if relation_type == "ped_vs_ped":
        scenario.side1.role = "pedestrians"
        scenario.side2.role = "pedestrians"
        scenario.label = (scenario.label or "") + " | ped vs ped"

    elif relation_type == "passengers_vs_pedestrians":
        scenario.side1.role = "passengers"
        scenario.side2.role = "pedestrians"
        scenario.label = (scenario.label or "") + " | passengers vs pedestrians"

    else:  # pedestrians_vs_passengers
        scenario.side1.role = "pedestrians"
        scenario.side2.role = "passengers"
        scenario.label = (scenario.label or "") + " | pedestrians vs passengers"


def apply_concern_for_law(scenario: Scenario) -> None:
    """
    Dimension: Concern for law (crossing signals).

    - If at least one side is passengers:
        * only the pedestrian side can have a signal.
        * choose among: none, legal, illegal.
    - If pedestrians vs pedestrians:
        * choose a focal side and one of {none, legal, illegal}.
          - none: both 'none'
          - legal: focal 'legal', other 'illegal'
          - illegal: focal 'illegal', other 'legal'
    - If passengers vs passengers:
        * treat as no legal complication.
    """
    scenario.side1.legal_status = "none"
    scenario.side2.legal_status = "none"

    r1 = scenario.side1.role
    r2 = scenario.side2.role

    if "passengers" in {r1, r2}:
        if r1 == "pedestrians":
            ped_side = scenario.side1
        elif r2 == "pedestrians":
            ped_side = scenario.side2
        else:
            return  # passengers vs passengers

        law_state = random.choice(["none", "legal", "illegal"])
        ped_side.legal_status = law_state
        scenario.label = (scenario.label or "") + f" | law={law_state}"

    elif r1 == "pedestrians" and r2 == "pedestrians":
        focal = random.choice(["side1", "side2"])
        law_state = random.choice(["none", "legal", "illegal"])

        if law_state == "none":
            scenario.side1.legal_status = "none"
            scenario.side2.legal_status = "none"
        else:
            focal_side = scenario.side1 if focal == "side1" else scenario.side2
            other_side = scenario.side2 if focal == "side1" else scenario.side1

            focal_side.legal_status = law_state
            other_side.legal_status = "illegal" if law_state == "legal" else "legal"

        scenario.label = (scenario.label or "") + f" | law={law_state}"

    else:
        scenario.side1.legal_status = "none"
        scenario.side2.legal_status = "none"


# -------------------------
# Session generation
# -------------------------

DIMENSION_GENERATORS = {
    "species": generate_species_scenario,
    "social_value": generate_social_value_scenario,
    "gender": generate_gender_scenario,
    "age": generate_age_scenario,
    "fitness": generate_fitness_scenario,
    "utilitarianism": generate_utilitarian_scenario,
    "random": generate_completely_random_scenario,
}


def generate_core_scenario(dimension: str) -> Scenario:
    """Generate a scenario for one of the 6 core dimensions (or 'random')."""
    scenario = DIMENSION_GENERATORS[dimension]()
    apply_interventionism(scenario)
    apply_relationship_to_vehicle(scenario)
    apply_concern_for_law(scenario)
    return scenario


def generate_session() -> List[Scenario]:
    """
    Create one session:
      - 2 scenarios for each of the 6 core dimensions
      - 1 completely random scenario
      - total: 13 scenarios, shuffled
    """
    dims = ["species", "social_value", "gender", "age", "fitness", "utilitarianism"]

    scenarios: List[Scenario] = []
    for d in dims:
        scenarios.append(generate_core_scenario(d))
        scenarios.append(generate_core_scenario(d))

    scenarios.append(generate_core_scenario("random"))

    random.shuffle(scenarios)
    return scenarios


# -------------------------
# JSON helpers (for saving intermediate files)
# -------------------------

def save_scenarios_json(scenarios: List[Scenario], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump([s.to_dict() for s in scenarios], f, ensure_ascii=False, indent=2)
    return path


def load_scenarios_json(path: str | Path) -> List[Scenario]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return [Scenario.from_dict(d) for d in data]
