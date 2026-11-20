"""Fixed, shared evaluation scenarios.

A scenario is just an environment reset seed with start perturbation on. Every controller
evaluated on a split sees *exactly the same* list of scenarios, so differences come from
the drivers, not from who got the easier starts.

Seed ranges are disjoint per split so a validation scenario can never leak into a test.
Training uses the environment's own seeding, which never touches these ranges as long as
training seeds stay below 1,000,000.
"""

from __future__ import annotations

SPLIT_BASE = {"validation": 1_000_000, "test": 2_000_000}


def scenario_seeds(split: str, count: int) -> list[int]:
    if split not in SPLIT_BASE:
        raise ValueError(f"split must be one of {sorted(SPLIT_BASE)}, got '{split}'")
    if count < 1:
        raise ValueError("count must be at least 1")
    base = SPLIT_BASE[split]
    return [base + i for i in range(count)]
