"""Task generator — picks a scenario, optionally applies domain randomization.

    - `domain_randomise=True`: perturb budget (±30%), time (±20%), noise,
      and other continuous parameters. Encourages sample-efficient policies.
    - `domain_randomise=False`: deterministic — for reproducible evals.
"""

from __future__ import annotations

import copy
import random
from typing import Optional

from models import TaskSpec
from server.tasks.scenarios import SCENARIOS, get_scenario


class TaskGenerator:
    """Yields TaskSpec instances with optional parameter perturbation."""

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)

    def generate(
        self,
        scenario_name: Optional[str] = None,
        domain_randomise: bool = False,
    ) -> TaskSpec:
        """Pick a scenario by name or at random; optionally randomize."""
        if scenario_name:
            base = get_scenario(scenario_name)
            if base is None:
                raise ValueError(f"Unknown scenario: {scenario_name}")
        else:
            base = self._rng.choice(SCENARIOS)

        if not domain_randomise:
            return base

        # Perturb budget (±30%) and time_limit (±20%). Skip if None.
        randomised = copy.deepcopy(base)
        if randomised.budget is not None:
            randomised.budget *= self._rng.uniform(0.7, 1.3)
        if randomised.time_limit is not None:
            randomised.time_limit *= self._rng.uniform(0.8, 1.2)
        # Tag the task_id so trajectories from randomized runs are distinguishable
        randomised.task_id = f"{randomised.task_id}_r{self._rng.randint(0, 999):03d}"
        return randomised
