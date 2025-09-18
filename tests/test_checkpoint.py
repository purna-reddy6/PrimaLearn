"""
test_checkpoint.py — Tests for checkpoint save/load round-trip.

Verifies that the wake-sleep engine can:
1. Save full state (library, programs, recognition weights, cycle results) after N cycles
2. Load that checkpoint and resume from cycle N+1
3. Produce consistent results after resumption
"""
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestRecognitionSaveLoad:
    """Test NeuralRecognitionNet save/load round-trip."""

    def test_save_load_round_trip(self):
        from src.engine.recognition import NeuralRecognitionNet

        net = NeuralRecognitionNet(n_features=50, hidden_dim=32)
        net.initialize(["prim_a", "prim_b", "prim_c", "prim_d"])

        # Modify weights to something non-initial
        net.W1[0, 0] = 42.0
        net.b2[1] = -3.14

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "recognition.npz")
            net.save_weights(path)

            # Verify files exist
            assert Path(path).exists()
            assert Path(path).with_suffix('.json').exists()

            # Load into a fresh network
            net2 = NeuralRecognitionNet(n_features=50, hidden_dim=32)
            loaded = net2.load_weights(path)

            assert loaded is True
            assert net2._initialized is True
            assert net2.primitive_names == ["prim_a", "prim_b", "prim_c", "prim_d"]
            np.testing.assert_array_equal(net2.W1, net.W1)
            np.testing.assert_array_equal(net2.b1, net.b1)
            np.testing.assert_array_equal(net2.W2, net.W2)
            np.testing.assert_array_equal(net2.b2, net.b2)
            np.testing.assert_array_equal(net2.vW1, net.vW1)
            np.testing.assert_array_equal(net2.vb2, net.vb2)
            assert net2.W1[0, 0] == 42.0
            assert net2.b2[1] == pytest.approx(-3.14)

    def test_load_nonexistent(self):
        from src.engine.recognition import NeuralRecognitionNet

        net = NeuralRecognitionNet()
        result = net.load_weights("/nonexistent/path.npz")
        assert result is False


class TestCheckpointRoundTrip:
    """Test full engine checkpoint save/load."""

    def test_checkpoint_saves_all_state(self):
        from src.engine.wake_sleep import WakeSleepEngine, WakeSleepConfig, CycleResult
        from src.engine.library import Library, Abstraction
        from src.engine.program import Program, LitNode
        from src.spelke_dsl import build_spelke_library
        from src.spelke_dsl.base import Arrow, tgrid

        with tempfile.TemporaryDirectory() as tmpdir:
            config = WakeSleepConfig(
                n_iterations=1,
                checkpoint_dir=tmpdir,
                verbose=False,
                use_enumerator=False,
                use_neural_recognition=True,
            )

            engine = WakeSleepEngine(config)
            registry = build_spelke_library()
            engine.initialize(registry)

            # Simulate a solved task
            stub = Program(root=LitNode("test"), task_id="task_001", source="ast_solver")
            engine._solved_programs["task_001"] = stub

            # Simulate an abstraction
            abs_ = Abstraction(
                name="abs_test_0",
                type_signature=Arrow(tgrid, tgrid),
                body=LitNode("body"),
                source_programs=["task_001"],
                systems_composed={"FORMS", "OBJECTS"},
                reuse_count=3,
                mdl_savings=15.0,
            )
            engine.library.add_abstraction(abs_)

            # Simulate cycle result
            cr = CycleResult(
                cycle=0, n_tasks=100, n_solved=10, solve_rate=0.1,
                n_heuristic=5, n_ast=4, n_enumerated=1,
                new_abstractions=[abs_.to_dict()],
                cross_system_abstractions=1,
                total_library_size=144,
                cycle_time_seconds=60.0,
                solved_task_ids=["task_001"],
            )
            engine.cycle_results.append(cr)

            # Save checkpoint
            engine._checkpoint(0)

            # Verify files were created
            ckpt_path = Path(tmpdir)
            assert (ckpt_path / "library_cycle_0.json").exists()
            assert (ckpt_path / "programs_cycle_0.json").exists()
            assert (ckpt_path / "cycle_results.json").exists()
            assert (ckpt_path / "results.json").exists()
            assert (ckpt_path / "recognition_cycle_0.npz").exists()
            assert (ckpt_path / "recognition_cycle_0.json").exists()

    def test_load_restores_state(self):
        from src.engine.wake_sleep import WakeSleepEngine, WakeSleepConfig, CycleResult
        from src.engine.library import Library, Abstraction
        from src.engine.program import Program, LitNode
        from src.spelke_dsl import build_spelke_library
        from src.spelke_dsl.base import Arrow, tgrid

        with tempfile.TemporaryDirectory() as tmpdir:
            # Setup and save
            config = WakeSleepConfig(
                n_iterations=1,
                checkpoint_dir=tmpdir,
                verbose=False,
                use_enumerator=False,
                use_neural_recognition=True,
            )

            engine1 = WakeSleepEngine(config)
            registry = build_spelke_library()
            engine1.initialize(registry)

            stub = Program(root=LitNode("test"), task_id="task_001", source="ast_solver")
            engine1._solved_programs["task_001"] = stub

            abs_ = Abstraction(
                name="abs_test_0",
                type_signature=Arrow(tgrid, tgrid),
                body=LitNode("body"),
                source_programs=["task_001"],
                systems_composed={"FORMS", "OBJECTS"},
                reuse_count=3,
                mdl_savings=15.0,
            )
            engine1.library.add_abstraction(abs_)

            cr = CycleResult(
                cycle=0, n_tasks=100, n_solved=10, solve_rate=0.1,
                n_heuristic=5, n_ast=4, n_enumerated=1,
                new_abstractions=[abs_.to_dict()],
                cross_system_abstractions=1,
                total_library_size=144,
                cycle_time_seconds=60.0,
                solved_task_ids=["task_001"],
            )
            engine1.cycle_results.append(cr)
            engine1._checkpoint(0)

            # Load into fresh engine
            config2 = WakeSleepConfig(
                n_iterations=3,
                verbose=False,
                use_enumerator=False,
                use_neural_recognition=True,
            )
            engine2 = WakeSleepEngine(config2)
            engine2.initialize(build_spelke_library())

            engine2.load_checkpoint(tmpdir, cycle=0)

            # Verify state
            assert "task_001" in engine2._solved_programs
            assert len(engine2.library.abstractions) >= 1
            abs_names = [a.name for a in engine2.library.abstractions]
            assert "abs_test_0" in abs_names
            assert engine2.library._cycle == 1  # advanced past loaded cycle
            assert len(engine2.cycle_results) == 1
            assert engine2.cycle_results[0].n_solved == 10
            assert engine2.cycle_results[0].solve_rate == 0.1

            # Recognition should be initialized
            if engine2._recognition is not None:
                assert engine2._recognition._initialized is True

