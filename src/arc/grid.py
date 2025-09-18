"""
grid.py — ARC Grid representation and core operations.

The Grid is the fundamental substrate of ARC-AGI:
a 2D array of cells with integer color values 0-9.
"""

from __future__ import annotations
import numpy as np
import json
from dataclasses import dataclass, field
from typing import Optional


# ARC color palette (standard 10 colors)
ARC_COLORS = {
    0: "black",     # background
    1: "blue",
    2: "red",
    3: "green",
    4: "yellow",
    5: "grey",
    6: "magenta",
    7: "orange",
    8: "cyan",
    9: "maroon",
}

# ANSI color codes for terminal display
ANSI_COLORS = {
    0: "\033[40m",    # black bg
    1: "\033[44m",    # blue
    2: "\033[41m",    # red
    3: "\033[42m",    # green
    4: "\033[43m",    # yellow
    5: "\033[47m",    # grey/white
    6: "\033[45m",    # magenta
    7: "\033[48;5;208m",  # orange
    8: "\033[46m",    # cyan
    9: "\033[48;5;52m",   # maroon
}
ANSI_RESET = "\033[0m"


@dataclass
class Grid:
    """
    A 2D ARC grid with integer color values.

    Wraps a numpy array with ARC-specific operations.
    """
    data: np.ndarray

    def __post_init__(self):
        if not isinstance(self.data, np.ndarray):
            self.data = np.array(self.data, dtype=np.int8)
        if self.data.ndim != 2:
            raise ValueError(f"Grid must be 2D, got {self.data.ndim}D")

    @classmethod
    def from_list(cls, data: list[list[int]]) -> Grid:
        return cls(np.array(data, dtype=np.int8))

    @classmethod
    def zeros(cls, h: int, w: int) -> Grid:
        return cls(np.zeros((h, w), dtype=np.int8))

    @classmethod
    def full(cls, h: int, w: int, color: int) -> Grid:
        return cls(np.full((h, w), color, dtype=np.int8))

    @property
    def height(self) -> int:
        return self.data.shape[0]

    @property
    def width(self) -> int:
        return self.data.shape[1]

    @property
    def shape(self) -> tuple[int, int]:
        return self.data.shape

    @property
    def size(self) -> int:
        return self.data.size

    def __getitem__(self, key):
        result = self.data[key]
        if isinstance(result, np.ndarray) and result.ndim == 2:
            return Grid(result)
        return int(result)

    def __setitem__(self, key, value):
        self.data[key] = value

    def __eq__(self, other):
        if isinstance(other, Grid):
            return self.shape == other.shape and np.array_equal(self.data, other.data)
        return False

    def __hash__(self):
        return hash(self.data.tobytes())

    def copy(self) -> Grid:
        return Grid(self.data.copy())

    def to_list(self) -> list[list[int]]:
        return self.data.tolist()

    def to_json(self) -> str:
        return json.dumps(self.to_list())

    @property
    def colors(self) -> set[int]:
        return set(int(x) for x in np.unique(self.data))

    @property
    def non_background_colors(self) -> set[int]:
        return self.colors - {0}

    def color_count(self, color: int) -> int:
        return int(np.sum(self.data == color))

    def display(self) -> str:
        """Terminal-friendly colored display."""
        lines = []
        for row in self.data:
            cells = []
            for val in row:
                color_code = ANSI_COLORS.get(int(val), "")
                cells.append(f"{color_code} {int(val)} {ANSI_RESET}")
            lines.append("".join(cells))
        return "\n".join(lines)

    def display_plain(self) -> str:
        """Plain text display (no ANSI)."""
        lines = []
        for row in self.data:
            lines.append(" ".join(str(int(v)) for v in row))
        return "\n".join(lines)

    def __repr__(self):
        return f"Grid({self.height}×{self.width}, colors={sorted(self.colors)})"


@dataclass
class TaskExample:
    """A single input→output example in an ARC task."""
    input: Grid
    output: Grid

    @classmethod
    def from_dict(cls, d: dict) -> TaskExample:
        return cls(
            input=Grid.from_list(d["input"]),
            output=Grid.from_list(d["output"]),
        )


@dataclass
class ArcTask:
    """
    A complete ARC-AGI task.

    Each task has:
    - train: list of input→output demonstration pairs
    - test: list of test inputs (with hidden outputs for evaluation)
    - task_id: unique identifier
    """
    task_id: str
    train: list[TaskExample]
    test: list[TaskExample]
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, task_id: str, d: dict) -> ArcTask:
        train = [TaskExample.from_dict(ex) for ex in d["train"]]
        test = [TaskExample.from_dict(ex) for ex in d["test"]]
        return cls(task_id=task_id, train=train, test=test)

    @property
    def n_train(self) -> int:
        return len(self.train)

    @property
    def n_test(self) -> int:
        return len(self.test)

    def train_inputs(self) -> list[Grid]:
        return [ex.input for ex in self.train]

    def train_outputs(self) -> list[Grid]:
        return [ex.output for ex in self.train]

    def test_inputs(self) -> list[Grid]:
        return [ex.input for ex in self.test]

    def test_outputs(self) -> list[Grid]:
        return [ex.output for ex in self.test]

    def summary(self) -> str:
        lines = [f"Task {self.task_id}: {self.n_train} train, {self.n_test} test"]
        for i, ex in enumerate(self.train):
            lines.append(f"  train[{i}]: {ex.input.shape} → {ex.output.shape}")
        for i, ex in enumerate(self.test):
            lines.append(f"  test[{i}]: {ex.input.shape} → {ex.output.shape}")
        return "\n".join(lines)

    def __repr__(self):
        return f"ArcTask({self.task_id}, train={self.n_train}, test={self.n_test})"
