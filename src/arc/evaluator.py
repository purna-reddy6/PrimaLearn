"""
evaluator.py — ARC-AGI task evaluation harness.

Measures solve rates, sample efficiency, and tracks per-task results.
"""

from __future__ import annotations
import json
import time
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Any
from src.arc.grid import ArcTask, Grid


@dataclass
class TaskResult:
    """Result of evaluating a solver on one ARC task."""
    task_id: str
    solved: bool
    predicted_outputs: list[np.ndarray]
    ground_truth: list[np.ndarray]
    search_attempts: int = 0
    solve_time_seconds: float = 0.0
    program_found: Optional[str] = None
    error: Optional[str] = None

    @property
    def accuracy(self) -> float:
        """Fraction of test cases correctly predicted."""
        if not self.ground_truth:
            return 0.0
        correct = 0
        for pred, gt in zip(self.predicted_outputs, self.ground_truth):
            if pred is not None and pred.shape == gt.shape and np.array_equal(pred, gt):
                correct += 1
        return correct / len(self.ground_truth)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "solved": self.solved,
            "accuracy": self.accuracy,
            "search_attempts": self.search_attempts,
            "solve_time_seconds": self.solve_time_seconds,
            "program": self.program_found,
            "error": self.error,
        }


@dataclass
class EvalResults:
    """Aggregated evaluation results across multiple tasks."""
    task_results: list[TaskResult] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def total_tasks(self) -> int:
        return len(self.task_results)

    @property
    def solved_count(self) -> int:
        return sum(1 for r in self.task_results if r.solved)

    @property
    def solve_rate(self) -> float:
        if not self.task_results:
            return 0.0
        return self.solved_count / self.total_tasks

    @property
    def mean_accuracy(self) -> float:
        if not self.task_results:
            return 0.0
        return np.mean([r.accuracy for r in self.task_results])

    @property
    def mean_solve_time(self) -> float:
        solved = [r for r in self.task_results if r.solved]
        if not solved:
            return 0.0
        return np.mean([r.solve_time_seconds for r in solved])

    @property
    def total_time(self) -> float:
        return sum(r.solve_time_seconds for r in self.task_results)

    def solved_task_ids(self) -> list[str]:
        return [r.task_id for r in self.task_results if r.solved]

    def unsolved_task_ids(self) -> list[str]:
        return [r.task_id for r in self.task_results if not r.solved]

    def summary(self) -> str:
        lines = [
            f"Evaluation Results:",
            f"  Tasks: {self.total_tasks}",
            f"  Solved: {self.solved_count} ({self.solve_rate:.1%})",
            f"  Mean accuracy: {self.mean_accuracy:.3f}",
            f"  Mean solve time: {self.mean_solve_time:.2f}s",
            f"  Total time: {self.total_time:.1f}s",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "total_tasks": self.total_tasks,
            "solved_count": self.solved_count,
            "solve_rate": self.solve_rate,
            "mean_accuracy": self.mean_accuracy,
            "mean_solve_time": self.mean_solve_time,
            "total_time": self.total_time,
            "metadata": self.metadata,
            "results": [r.to_dict() for r in self.task_results],
        }

    def save(self, path: str | Path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> EvalResults:
        with open(path) as f:
            data = json.load(f)
        results = cls(metadata=data.get("metadata", {}))
        for rd in data.get("results", []):
            results.task_results.append(TaskResult(
                task_id=rd["task_id"],
                solved=rd["solved"],
                predicted_outputs=[],
                ground_truth=[],
                search_attempts=rd.get("search_attempts", 0),
                solve_time_seconds=rd.get("solve_time_seconds", 0),
                program_found=rd.get("program"),
                error=rd.get("error"),
            ))
        return results


# ──────────────────────────────────────────────────────────────────────
# Solver Protocol
# ──────────────────────────────────────────────────────────────────────

class Solver:
    """Base class for ARC solvers. Subclass and implement solve()."""

    def solve(self, task: ArcTask, max_attempts: int = 100) -> TaskResult:
        """
        Given an ARC task (with train examples), predict test outputs.

        Args:
            task: The ARC task with train examples visible
            max_attempts: Maximum search attempts allowed

        Returns:
            TaskResult with predictions
        """
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self.__class__.__name__


# ──────────────────────────────────────────────────────────────────────
# Evaluation Runner
# ──────────────────────────────────────────────────────────────────────

class Evaluator:
    """Runs a solver on a set of ARC tasks and collects results."""

    def __init__(self, solver: Solver, verbose: bool = True):
        self.solver = solver
        self.verbose = verbose

    def evaluate(
        self,
        tasks: list[ArcTask],
        max_attempts: int = 100,
        timeout_per_task: float = 300.0,
    ) -> EvalResults:
        """Run solver on all tasks and return aggregated results."""
        results = EvalResults(metadata={
            "solver": self.solver.name,
            "max_attempts": max_attempts,
            "timeout_per_task": timeout_per_task,
            "n_tasks": len(tasks),
        })

        for i, task in enumerate(tasks):
            if self.verbose:
                print(f"[{i+1}/{len(tasks)}] {task.task_id}...", end=" ", flush=True)

            start = time.time()
            try:
                result = self.solver.solve(task, max_attempts=max_attempts)
                result.solve_time_seconds = time.time() - start
            except Exception as e:
                result = TaskResult(
                    task_id=task.task_id,
                    solved=False,
                    predicted_outputs=[],
                    ground_truth=[ex.output.data for ex in task.test],
                    solve_time_seconds=time.time() - start,
                    error=str(e),
                )

            results.task_results.append(result)

            if self.verbose:
                status = "✓" if result.solved else "✗"
                print(f"{status} ({result.solve_time_seconds:.1f}s)")

        if self.verbose:
            print(results.summary())

        return results

    def evaluate_at_budgets(
        self,
        tasks: list[ArcTask],
        budgets: list[int] = [10, 100, 1000],
    ) -> dict[int, EvalResults]:
        """
        Run evaluation at multiple search budgets.
        Returns solve-rate curve data for Figure 4.
        """
        results_by_budget = {}
        for budget in budgets:
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"  Budget: {budget} attempts")
                print(f"{'='*60}")
            results_by_budget[budget] = self.evaluate(tasks, max_attempts=budget)
        return results_by_budget


def sample_efficiency_curve(
    solver: Solver,
    all_tasks: list[ArcTask],
    train_sizes: list[int] = [10, 50, 100, 200],
    eval_tasks: Optional[list[ArcTask]] = None,
    max_attempts: int = 100,
    n_seeds: int = 3,
) -> dict[int, list[float]]:
    """
    Compute sample-efficiency curve: solve rate vs. training task count.

    For each train_size:
    1. Sample train_size tasks for wake-sleep training
    2. Evaluate on held-out eval tasks
    3. Record solve rate

    Returns: {train_size: [solve_rate_seed1, solve_rate_seed2, ...]}
    """
    import random
    if eval_tasks is None:
        eval_tasks = all_tasks

    results = {}
    for size in train_sizes:
        size_results = []
        for seed in range(n_seeds):
            random.seed(seed)
            train_subset = random.sample(all_tasks, min(size, len(all_tasks)))

            # In real usage, the solver would be trained on train_subset
            # then evaluated on eval_tasks. This is a placeholder for
            # the wake-sleep training loop.
            evaluator = Evaluator(solver, verbose=False)
            eval_result = evaluator.evaluate(eval_tasks, max_attempts=max_attempts)
            size_results.append(eval_result.solve_rate)

        results[size] = size_results

    return results
