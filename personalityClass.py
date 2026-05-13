from __future__ import annotations

import csv
from dataclasses import dataclass, fields
from pathlib import Path


@dataclass
class PersonalityVector:
    """
    Fixed traits that define an agent's disposition. Tuned by RL across games.
    All values are in [0, 1]. More fields will be added later.

    CSV file format (two rows):
        aggression,trust_baseline,lie_propensity
        0.9,0.1,0.8

    Unknown columns are ignored. Missing columns fall back to dataclass defaults,
    which makes the file format forward-compatible when we add traits later.
    """
    aggression: float = 0.5      # tendency to threaten, accuse, push hard
    trust_baseline: float = 0.5  # default willingness to believe the opponent
    lie_propensity: float = 0.5  # willingness to say things that don't match intent
    evasiveness: float = 0.5 # tendency to not respond to questions in the belief state
    cooperativeness: float = 0.5 # reaction to opponents intent to split

    def clamp(self) -> None:
        """Keep all trait values within [0, 1] after any RL update."""
        self.aggression = max(0.0, min(1.0, self.aggression))
        self.trust_baseline = max(0.0, min(1.0, self.trust_baseline))
        self.lie_propensity = max(0.0, min(1.0, self.lie_propensity))
        self.evasiveness = max(0.0, min(1.0, self.evasiveness))
        self.cooperativeness = max(0.0, min(1.0, self.cooperativeness))

    @classmethod
    def from_csv(cls, path: str | Path) -> "PersonalityVector":
        """
        Load a personality vector from a two-row CSV (header + one row of values).
        Raises FileNotFoundError if the path doesn't exist, ValueError if the file
        is malformed.
        """
        path = Path(path).expanduser()
        with path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            row = next(reader, None)

        if row is None:
            raise ValueError(f"Personality file {path} has no data row.")

        known = {f.name for f in fields(cls)}
        kwargs: dict[str, float] = {}
        for key, value in row.items():
            if key is None or key not in known:
                continue
            if value is None or value.strip() == "":
                continue
            try:
                kwargs[key] = float(value)
            except ValueError as e:
                raise ValueError(
                    f"Invalid float for '{key}' in {path}: {value!r}"
                ) from e

        instance = cls(**kwargs)
        instance.clamp()
        return instance

    def to_csv(self, path: str | Path) -> None:
        """
        Save the personality vector to a CSV file. Useful for persisting
        RL-tuned personalities between training runs.
        """
        path = Path(path).expanduser()
        field_names = [f.name for f in fields(self)]
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=field_names)
            writer.writeheader()
            writer.writerow({name: getattr(self, name) for name in field_names})