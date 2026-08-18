"""Stochastic prompt formatting (paper §2.2): template slots whose value is
sampled from a configured distribution — a cheap diversity knob."""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class StochasticSlot:
    name: str
    options: list[str]
    weights: list[float]

    @classmethod
    def from_config(cls, cfg: dict) -> StochasticSlot:
        options = [str(o) for o in cfg["options"]]
        weights = [float(w) for w in cfg.get("weights", [1.0] * len(options))]
        if len(weights) != len(options):
            raise ValueError(
                f"slot {cfg.get('name')}: {len(options)} options, {len(weights)} weights"
            )
        return cls(name=str(cfg["name"]), options=options, weights=weights)


def sample_slots(slots: list[StochasticSlot], rng: random.Random) -> dict[str, str]:
    return {slot.name: rng.choices(slot.options, weights=slot.weights, k=1)[0] for slot in slots}
