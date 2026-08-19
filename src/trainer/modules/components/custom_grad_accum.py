from dataclasses import dataclass, field
from typing import Dict, List

__all__ = ["GradAccumSchedule"]

# Credit @shivvor: For now minimal changes until further adapted
@dataclass
class GradAccumSchedule:
    """Step-based gradient accumulation schedule.

    When provided to a LightningModule, overrides trainer.accumulate_grad_batches.
    Keys are optimizer steps (not batches, not epochs).

    Attributes:
        schedule: Mapping from optimizer step to accumulation value.
            If step 0 is not specified, defaults to 1.

    Example:
        1) Ramp up accumulation over training::

            GradAccumSchedule({0: 1, 1000: 2, 5000: 4})

        2) Constant accumulation::

            GradAccumSchedule({0: 4})

    Note:
        Step 0 is automatically added with value 1 if not specified.
        This ensures there's always a valid accumulation value.
    """

    schedule: Dict[int, int]
    _sorted_steps: List[int] = field(init=False, repr=False)

    # TODO: Instead of type checking in post init use type checker with adjustable default
    def __post_init__(self) -> None:
        """Validate schedule and precompute sorted steps."""
        if not self.schedule:
            raise ValueError("Schedule cannot be empty")

        for step, accum in self.schedule.items():
            if not isinstance(step, int) or step < 0:
                raise ValueError(
                    f"Step must be non-negative int, got {step} (type: {type(step).__name__})"
                )
            if not isinstance(accum, int) or accum < 1:
                raise ValueError(
                    f"Accumulation must be >= 1, got {accum} (type: {type(accum).__name__})"
                )

        # Add default step 0 if not specified
        if 0 not in self.schedule:
            self.schedule = {0: 1, **self.schedule}

        self._sorted_steps = sorted(self.schedule.keys())

    def adjust_accumulation_steps(self, target_memory, accum_factor, used_memory, max_steps):
        """ 
        Dynamically adjusts (increase or decrease) curr step pre-defined schedule for accumulation based off used memory, max and min bounds
        """

        if used_memory < target_memory * 0.9:
            return min(accum_factor + 1, max_steps)  # Gradually increase steps
        elif used_memory > target_memory:
            return max(accum_factor - 1, 1)  # Decrease steps if memory is overloaded
        return accum_factor

    def get_accum(self, step: int) -> int:
        """Get accumulation value for given optimizer step.

        Args:
            step: Current optimizer step (0-indexed).

        Returns:
            The gradient accumulation factor for this step.

        Example:
            >>> schedule = GradAccumSchedule({0: 1, 1000: 4})
            >>> schedule.get_accum(500)   # Returns 1
            >>> schedule.get_accum(1500)  # Returns 4
        """
        accum = 1
        for s in self._sorted_steps:
            if step >= s:
                accum = self.schedule[s]
            else:
                break
        return accum