"""
loader.py — ARC-AGI dataset loader.

Loads ARC-AGI-1 and ARC-AGI-2 from their JSON format.
Supports task filtering, train/eval/test splits, and LARC annotations.
"""

from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Optional
from src.arc.grid import ArcTask


class ArcDataset:
    """
    Loads and manages ARC-AGI task collections.

    Directory structure expected:
        data/arc-agi-1/
            training/
                <task_id>.json
            evaluation/
                <task_id>.json
        data/arc-agi-2/
            training/
                <task_id>.json
            evaluation/
                <task_id>.json
    """

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self._tasks: dict[str, ArcTask] = {}
        self._splits: dict[str, list[str]] = {}

    def load_split(self, split: str = "training") -> list[ArcTask]:
        """Load a split (training, evaluation, test)."""
        split_dir = self.data_dir / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        tasks = []
        task_ids = []
        for filepath in sorted(split_dir.glob("*.json")):
            task_id = filepath.stem
            with open(filepath) as f:
                data = json.load(f)
            task = ArcTask.from_dict(task_id, data)
            self._tasks[task_id] = task
            tasks.append(task)
            task_ids.append(task_id)

        self._splits[split] = task_ids
        return tasks

    def load_all(self) -> dict[str, list[ArcTask]]:
        """Load all available splits."""
        result = {}
        for split_name in ["training", "evaluation"]:
            split_dir = self.data_dir / split_name
            if split_dir.exists():
                result[split_name] = self.load_split(split_name)
        return result

    def get_task(self, task_id: str) -> Optional[ArcTask]:
        return self._tasks.get(task_id)

    def all_tasks(self) -> list[ArcTask]:
        return list(self._tasks.values())

    def task_ids(self, split: Optional[str] = None) -> list[str]:
        if split:
            return self._splits.get(split, [])
        return list(self._tasks.keys())

    def __len__(self) -> int:
        return len(self._tasks)

    def __getitem__(self, task_id: str) -> ArcTask:
        return self._tasks[task_id]

    def __contains__(self, task_id: str) -> bool:
        return task_id in self._tasks

    def summary(self) -> str:
        lines = [f"ArcDataset: {self.data_dir}"]
        lines.append(f"  Total tasks loaded: {len(self._tasks)}")
        for split, ids in self._splits.items():
            lines.append(f"  {split}: {len(ids)} tasks")
        return "\n".join(lines)


class LarcAnnotations:
    """
    LARC — Language-Annotated ARC dataset (Acquaviva et al. 2022).

    Provides natural-language task descriptions that serve as
    Carey-style placeholder structures for LAPS-style language-conditioned
    abstraction.
    """

    def __init__(self, larc_dir: str | Path):
        self.larc_dir = Path(larc_dir)
        self._annotations: dict[str, list[str]] = {}

    def load(self) -> None:
        """Load LARC annotations."""
        ann_file = self.larc_dir / "larc.json"
        if ann_file.exists():
            with open(ann_file) as f:
                data = json.load(f)
            for entry in data:
                task_id = entry.get("task_id", entry.get("id", ""))
                descriptions = entry.get("descriptions", [])
                if isinstance(descriptions, str):
                    descriptions = [descriptions]
                self._annotations[task_id] = descriptions
        else:
            # Try loading from per-task files
            for filepath in sorted(self.larc_dir.glob("*.json")):
                task_id = filepath.stem
                with open(filepath) as f:
                    data = json.load(f)
                descs = data if isinstance(data, list) else data.get("descriptions", [])
                self._annotations[task_id] = descs

    def get_descriptions(self, task_id: str) -> list[str]:
        return self._annotations.get(task_id, [])

    def has_annotation(self, task_id: str) -> bool:
        return task_id in self._annotations

    def annotated_task_ids(self) -> list[str]:
        return list(self._annotations.keys())

    def __len__(self) -> int:
        return len(self._annotations)


def load_arc_agi_1(base_dir: str | Path = "data/arc-agi-1") -> ArcDataset:
    """Convenience loader for ARC-AGI-1."""
    ds = ArcDataset(base_dir)
    ds.load_all()
    return ds


def load_arc_agi_2(base_dir: str | Path = "data/arc-agi-2") -> ArcDataset:
    """Convenience loader for ARC-AGI-2."""
    ds = ArcDataset(base_dir)
    ds.load_all()
    return ds
