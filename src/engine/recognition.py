"""
recognition.py — Neural recognition model for guided search.

Closes the dreaming sleep loop: given an ARC task, predicts which
primitives are most likely relevant to solving it.

NeuralRecognitionNet: a 2-layer MLP trained with binary cross-entropy
and SGD with momentum.  Input: ~50-dim feature vector.
Output: per-primitive relevance probabilities.

TaskFeatures and helpers (_has_uniform_border, _check_periodic_fast)
are preserved intact for use by other modules.
"""

from __future__ import annotations
import numpy as np
from collections import Counter
from typing import Optional
from src.spelke_dsl.l_objects import _extract_objects


class TaskFeatures:
    """Feature vector extracted from an ARC task."""

    def __init__(self, task):
        self.features = {}
        pairs = [(ex.input.data, ex.output.data) for ex in task.train]
        if not pairs:
            return

        inp, out = pairs[0]
        ih, iw = inp.shape
        oh, ow = out.shape

        # ── Size features ──
        self.features["same_size"] = (ih == oh and iw == ow)
        self.features["h_ratio"] = oh / ih if ih > 0 else 0
        self.features["w_ratio"] = ow / iw if iw > 0 else 0
        self.features["size_ratio"] = (oh * ow) / (ih * iw) if ih * iw > 0 else 0
        self.features["output_smaller"] = oh * ow < ih * iw
        self.features["output_larger"] = oh * ow > ih * iw
        self.features["square_input"] = ih == iw
        self.features["square_output"] = oh == ow
        self.features["double_h"] = abs(oh - 2 * ih) < 2
        self.features["double_w"] = abs(ow - 2 * iw) < 2
        self.features["half_h"] = ih > 0 and abs(oh - ih // 2) < 2
        self.features["half_w"] = iw > 0 and abs(ow - iw // 2) < 2
        self.features["int_scale"] = (
            (oh % ih == 0 and ow % iw == 0) if ih > 0 and iw > 0 else False
        )

        # ── Color features ──
        in_colors = set(int(x) for x in inp.flat)
        out_colors = set(int(x) for x in out.flat)
        self.features["n_colors_in"] = len(in_colors)
        self.features["n_colors_out"] = len(out_colors)
        self.features["same_colors"] = in_colors == out_colors
        self.features["colors_removed"] = len(in_colors - out_colors)
        self.features["colors_added"] = len(out_colors - in_colors)
        self.features["single_color_out"] = len(out_colors - {0}) <= 1

        # ── Object features ──
        try:
            in_objs = _extract_objects(inp)
            out_objs = _extract_objects(out) if ih == oh and iw == ow else []
            self.features["n_objects_in"] = len(in_objs)
            self.features["n_objects_out"] = len(out_objs)
            self.features["objects_same_count"] = len(in_objs) == len(out_objs)
            self.features["single_object_out"] = len(out_objs) == 1
            self.features["many_objects"] = len(in_objs) > 3
            if in_objs:
                sizes = [o.size for o in in_objs]
                self.features["all_same_size"] = len(set(sizes)) == 1
                self.features["has_singleton"] = 1 in sizes
                obj_colors = set(o.color for o in in_objs)
                self.features["multicolor_objects"] = len(obj_colors) > 1
        except Exception:
            self.features["n_objects_in"] = 0
            self.features["n_objects_out"] = 0

        # ── Symmetry / diff features ──
        if ih == oh and iw == ow:
            self.features["diff_count"] = int(np.sum(inp != out))
            self.features["diff_ratio"] = self.features["diff_count"] / (ih * iw)
        else:
            self.features["diff_count"] = -1
            self.features["diff_ratio"] = -1

        self.features["input_h_sym"] = np.array_equal(inp, np.fliplr(inp))
        self.features["input_v_sym"] = np.array_equal(inp, np.flipud(inp))
        if ih == iw:
            self.features["input_rot90_sym"] = np.array_equal(inp, np.rot90(inp, k=-1))

        # ── Structural features ──
        self.features["has_border"] = _has_uniform_border(inp)
        self.features["is_periodic"] = _check_periodic_fast(inp)
        self.features["bg_dominant"] = np.sum(inp == 0) > ih * iw * 0.5

        # ── Cross-example consistency ──
        size_changes = set()
        for i, o in pairs:
            size_changes.add(
                (
                    o.shape[0] / i.shape[0] if i.shape[0] > 0 else 0,
                    o.shape[1] / i.shape[1] if i.shape[1] > 0 else 0,
                )
            )
        self.features["consistent_size_change"] = len(size_changes) == 1

        # ── Color histogram of input (colors 0-9) ──
        total_cells = ih * iw if ih * iw > 0 else 1
        for c in range(10):
            self.features[f"color_hist_{c}"] = float(np.sum(inp == c)) / total_cells


def _has_uniform_border(g):
    if g.shape[0] < 3 or g.shape[1] < 3:
        return False
    border_vals = set()
    border_vals.update(int(x) for x in g[0, :])
    border_vals.update(int(x) for x in g[-1, :])
    border_vals.update(int(x) for x in g[:, 0])
    border_vals.update(int(x) for x in g[:, -1])
    return len(border_vals) == 1 and border_vals != {0}


def _check_periodic_fast(g):
    h, w = g.shape
    for ph in [1, 2, 3]:
        if h % ph != 0:
            continue
        for pw in [1, 2, 3]:
            if w % pw != 0:
                continue
            if ph == h and pw == w:
                continue
            pat = g[:ph, :pw]
            if np.array_equal(np.tile(pat, (h // ph, w // pw)), g):
                return True
    return False


# ──────────────────────────────────────────────────────────────────────
# Feature extraction helpers
# ──────────────────────────────────────────────────────────────────────

# Feature names in fixed order — must match extract_feature_vector indices.
# Total: 50 dims (36 named features + 10 color histogram + 4 padding zeros).
_FEATURE_KEYS = [
    # [0-12] size features
    "same_size",           # 0
    "h_ratio",             # 1
    "w_ratio",             # 2
    "size_ratio",          # 3
    "output_smaller",      # 4
    "output_larger",       # 5
    "square_input",        # 6
    "square_output",       # 7
    "double_h",            # 8
    "double_w",            # 9
    "half_h",              # 10
    "half_w",              # 11
    "int_scale",           # 12
    # [13-18] color features
    "n_colors_in",         # 13  — divide by 10
    "n_colors_out",        # 14  — divide by 10
    "same_colors",         # 15
    "colors_removed",      # 16  — divide by 10
    "colors_added",        # 17  — divide by 10
    "single_color_out",    # 18
    # [19-26] object features
    "n_objects_in",        # 19  — divide by 10
    "n_objects_out",       # 20  — divide by 10
    "objects_same_count",  # 21
    "single_object_out",   # 22
    "many_objects",        # 23
    "all_same_size",       # 24
    "has_singleton",       # 25
    "multicolor_objects",  # 26
    # [27-28] diff features
    "diff_count",          # 27  — divide by 100, clip
    "diff_ratio",          # 28  — clip to [0,1]
    # [29-35] symmetry / structural
    "input_h_sym",         # 29
    "input_v_sym",         # 30
    "input_rot90_sym",     # 31
    "has_border",          # 32
    "is_periodic",         # 33
    "bg_dominant",         # 34
    "consistent_size_change",  # 35
]
# [36-45] color histogram features (color_hist_0 … color_hist_9)
# [46-49] padding zeros


def _features_to_vector(feats: dict, n_features: int = 50) -> np.ndarray:
    """
    Convert a TaskFeatures.features dict to a fixed-size float32 numpy array.

    Normalisation:
      - booleans           → 0.0 / 1.0
      - n_colors_*         → / 10.0
      - colors_removed/added → / 10.0
      - n_objects_*        → / 10.0
      - diff_count         → / 100.0, clipped to [0, 1]
      - diff_ratio         → clipped to [0, 1]  (already in [0,1] for same-size)
    """
    vec = np.zeros(n_features, dtype=np.float32)

    def _b(v):
        return 1.0 if v else 0.0

    # [0] same_size
    vec[0] = _b(feats.get("same_size", False))
    # [1] h_ratio
    vec[1] = float(feats.get("h_ratio", 0))
    # [2] w_ratio
    vec[2] = float(feats.get("w_ratio", 0))
    # [3] size_ratio
    vec[3] = float(feats.get("size_ratio", 0))
    # [4] output_smaller
    vec[4] = _b(feats.get("output_smaller", False))
    # [5] output_larger
    vec[5] = _b(feats.get("output_larger", False))
    # [6] square_input
    vec[6] = _b(feats.get("square_input", False))
    # [7] square_output
    vec[7] = _b(feats.get("square_output", False))
    # [8] double_h
    vec[8] = _b(feats.get("double_h", False))
    # [9] double_w
    vec[9] = _b(feats.get("double_w", False))
    # [10] half_h
    vec[10] = _b(feats.get("half_h", False))
    # [11] half_w
    vec[11] = _b(feats.get("half_w", False))
    # [12] int_scale
    vec[12] = _b(feats.get("int_scale", False))
    # [13] n_colors_in / 10.0
    vec[13] = float(feats.get("n_colors_in", 0)) / 10.0
    # [14] n_colors_out / 10.0
    vec[14] = float(feats.get("n_colors_out", 0)) / 10.0
    # [15] same_colors
    vec[15] = _b(feats.get("same_colors", False))
    # [16] colors_removed / 10.0
    vec[16] = float(feats.get("colors_removed", 0)) / 10.0
    # [17] colors_added / 10.0
    vec[17] = float(feats.get("colors_added", 0)) / 10.0
    # [18] single_color_out
    vec[18] = _b(feats.get("single_color_out", False))
    # [19] n_objects_in / 10.0
    vec[19] = float(feats.get("n_objects_in", 0)) / 10.0
    # [20] n_objects_out / 10.0
    vec[20] = float(feats.get("n_objects_out", 0)) / 10.0
    # [21] objects_same_count
    vec[21] = _b(feats.get("objects_same_count", False))
    # [22] single_object_out
    vec[22] = _b(feats.get("single_object_out", False))
    # [23] many_objects
    vec[23] = _b(feats.get("many_objects", False))
    # [24] all_same_size
    vec[24] = _b(feats.get("all_same_size", False))
    # [25] has_singleton
    vec[25] = _b(feats.get("has_singleton", False))
    # [26] multicolor_objects
    vec[26] = _b(feats.get("multicolor_objects", False))
    # [27] diff_count / 100.0, clipped to [0,1]
    dc = feats.get("diff_count", 0)
    vec[27] = float(np.clip(dc / 100.0, 0.0, 1.0)) if dc >= 0 else 0.0
    # [28] diff_ratio, clipped to [0,1]
    dr = feats.get("diff_ratio", 0)
    vec[28] = float(np.clip(dr, 0.0, 1.0)) if dr >= 0 else 0.0
    # [29] input_h_sym
    vec[29] = _b(feats.get("input_h_sym", False))
    # [30] input_v_sym
    vec[30] = _b(feats.get("input_v_sym", False))
    # [31] input_rot90_sym
    vec[31] = _b(feats.get("input_rot90_sym", False))
    # [32] has_border
    vec[32] = _b(feats.get("has_border", False))
    # [33] is_periodic
    vec[33] = _b(feats.get("is_periodic", False))
    # [34] bg_dominant
    vec[34] = _b(feats.get("bg_dominant", False))
    # [35] consistent_size_change
    vec[35] = _b(feats.get("consistent_size_change", False))
    # [36-45] color histogram (colors 0-9, already normalised in TaskFeatures)
    for c in range(10):
        vec[36 + c] = float(feats.get(f"color_hist_{c}", 0.0))
    # [46-49] padding zeros (already zero from np.zeros)

    return vec


# ──────────────────────────────────────────────────────────────────────
# NeuralRecognitionNet — 2-layer MLP with SGD + momentum
# ──────────────────────────────────────────────────────────────────────

class NeuralRecognitionNet:
    """
    2-layer MLP predicting which primitives are relevant for a task.

    Input:  ~50-dim feature vector from TaskFeatures
    Output: probability for each primitive in the library
    Training: binary cross-entropy, SGD with momentum
    """

    def __init__(self, n_features: int = 50, hidden_dim: int = 128):
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.primitive_names: list[str] = []
        self.W1 = None   # shape (n_features, hidden_dim)
        self.b1 = None   # shape (hidden_dim,)
        self.W2 = None   # shape (hidden_dim, n_primitives)
        self.b2 = None   # shape (n_primitives,)
        # Momentum terms
        self.vW1 = None
        self.vb1 = None
        self.vW2 = None
        self.vb2 = None
        self._initialized = False

    # ------------------------------------------------------------------
    def initialize(self, primitive_names: list[str]) -> None:
        """Initialize weights given the full list of primitive names."""
        self.primitive_names = list(primitive_names)
        n_out = len(self.primitive_names)
        rng = np.random.default_rng(42)

        # He initialisation for W1 (fan-in = n_features)
        scale1 = np.sqrt(2.0 / self.n_features)
        self.W1 = rng.normal(0.0, scale1, (self.n_features, self.hidden_dim)).astype(np.float32)
        self.b1 = np.zeros(self.hidden_dim, dtype=np.float32)

        # He initialisation for W2 (fan-in = hidden_dim)
        scale2 = np.sqrt(2.0 / self.hidden_dim)
        self.W2 = rng.normal(0.0, scale2, (self.hidden_dim, n_out)).astype(np.float32)
        self.b2 = np.zeros(n_out, dtype=np.float32)

        # Momentum buffers — zero initialised
        self.vW1 = np.zeros_like(self.W1)
        self.vb1 = np.zeros_like(self.b1)
        self.vW2 = np.zeros_like(self.W2)
        self.vb2 = np.zeros_like(self.b2)

        self._initialized = True

    # ------------------------------------------------------------------
    def extract_feature_vector(self, task) -> np.ndarray:
        """
        Extract fixed-size feature vector from a task.
        Returns shape (n_features,).
        """
        tf = TaskFeatures(task)
        return _features_to_vector(tf.features, self.n_features)

    # ------------------------------------------------------------------
    def _forward(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Forward pass.
        X shape: (batch, n_features)
        Returns (h, probs):
          h     shape (batch, hidden_dim)  — post-ReLU hidden activations
          probs shape (batch, n_primitives) — sigmoid output probabilities
        """
        z1 = X @ self.W1 + self.b1          # (batch, hidden)
        h = np.maximum(0.0, z1)             # ReLU
        logits = h @ self.W2 + self.b2      # (batch, n_primitives)
        clipped = np.clip(logits, -10.0, 10.0)
        probs = 1.0 / (1.0 + np.exp(-clipped))  # sigmoid
        return h, probs

    # ------------------------------------------------------------------
    def _backward(
        self,
        X: np.ndarray,
        h: np.ndarray,
        probs: np.ndarray,
        Y: np.ndarray,
        lr: float,
        momentum: float,
    ) -> float:
        """
        One gradient step with SGD + momentum.
        Returns the mean binary cross-entropy loss for this batch.
        """
        eps = 1e-7
        batch_size = X.shape[0]

        # Binary cross-entropy loss
        loss = -np.mean(
            Y * np.log(probs + eps) + (1.0 - Y) * np.log(1.0 - probs + eps)
        )

        # Gradient of loss w.r.t. logits (dL/dlogit = probs - Y for BCE + sigmoid)
        dlogits = (probs - Y) / batch_size   # (batch, n_primitives)

        # Gradients for W2, b2
        dW2 = h.T @ dlogits                  # (hidden, n_primitives)
        db2 = dlogits.sum(axis=0)            # (n_primitives,)

        # Backprop through W2 to h
        dh = dlogits @ self.W2.T             # (batch, hidden)

        # ReLU gradient
        dz1 = dh * (h > 0).astype(np.float32)  # (batch, hidden)

        # Gradients for W1, b1
        dW1 = X.T @ dz1                      # (n_features, hidden)
        db1 = dz1.sum(axis=0)               # (hidden,)

        # SGD with momentum updates
        self.vW2 = momentum * self.vW2 + lr * dW2
        self.vb2 = momentum * self.vb2 + lr * db2
        self.vW1 = momentum * self.vW1 + lr * dW1
        self.vb1 = momentum * self.vb1 + lr * db1

        self.W2 -= self.vW2
        self.b2 -= self.vb2
        self.W1 -= self.vW1
        self.b1 -= self.vb1

        return float(loss)

    # ------------------------------------------------------------------
    def train(
        self,
        dream_tasks: list,
        solved_tasks: list,
        n_epochs: int = 20,
        lr: float = 0.01,
        momentum: float = 0.9,
    ) -> float:
        """
        Train on dream tasks + solved tasks.

        dream_tasks:  list of DreamTask objects (from dream_generator).
                      Each DreamTask must expose .task and .primitives_used
                      (iterable of primitive name strings).
        solved_tasks: list of (task, primitive_names_used) tuples.

        Returns: final average loss (last epoch).
        """
        if not self._initialized:
            return 0.0

        n_prims = len(self.primitive_names)
        prim_index = {name: i for i, name in enumerate(self.primitive_names)}

        # ── Build (feature_vector, label_vector) pairs ──
        X_list: list[np.ndarray] = []
        Y_list: list[np.ndarray] = []

        def _add_sample(task, prim_names_used):
            try:
                x = self.extract_feature_vector(task)
            except Exception:
                return
            y = np.zeros(n_prims, dtype=np.float32)
            for pname in prim_names_used:
                if pname in prim_index:
                    y[prim_index[pname]] = 1.0
            X_list.append(x)
            Y_list.append(y)

        # From dream tasks
        for dt in dream_tasks:
            try:
                _add_sample(dt.task, dt.primitives_used)
            except Exception:
                pass

        # From solved tasks
        for item in solved_tasks:
            try:
                task, prim_names = item
                _add_sample(task, prim_names)
            except Exception:
                pass

        if not X_list:
            return 0.0

        X = np.array(X_list, dtype=np.float32)   # (N, n_features)
        Y = np.array(Y_list, dtype=np.float32)   # (N, n_primitives)

        final_loss = 0.0
        for epoch in range(n_epochs):
            # Shuffle
            perm = np.random.permutation(len(X))
            X_shuf = X[perm]
            Y_shuf = Y[perm]

            h, probs = self._forward(X_shuf)
            loss = self._backward(X_shuf, h, probs, Y_shuf, lr, momentum)
            final_loss = loss

        return final_loss

    # ------------------------------------------------------------------
    def extend_primitives(self, all_names: list[str]) -> None:
        """
        Add newly invented abstraction names to the output layer.

        Called each cycle after abstractions are registered. Existing weights
        are preserved; new output neurons are zero-initialized so new
        abstractions start with a neutral prior (0.5 after sigmoid) rather
        than 0.0, making them visible to the enumerator's sorting step.
        """
        if not self._initialized:
            return
        new_names = [n for n in all_names if n not in self.primitive_names]
        if not new_names:
            return

        n_new = len(new_names)
        self.primitive_names.extend(new_names)

        # Extend W2: append n_new zero columns (neutral logit → 0.5 sigmoid)
        new_cols = np.zeros((self.W2.shape[0], n_new), dtype=np.float32)
        self.W2 = np.concatenate([self.W2, new_cols], axis=1)
        self.b2 = np.concatenate([self.b2, np.zeros(n_new, dtype=np.float32)])

        # Extend momentum buffers to match
        self.vW2 = np.concatenate([self.vW2, np.zeros_like(new_cols)], axis=1)
        self.vb2 = np.concatenate([self.vb2, np.zeros(n_new, dtype=np.float32)])

    # ------------------------------------------------------------------
    def predict_primitive_priors(self, task) -> dict[str, float]:
        """
        Return {primitive_name: probability} for use as enumerator priors.
        Returns empty dict if not initialized.
        """
        if not self._initialized:
            return {}
        features = self.extract_feature_vector(task)
        X = features[np.newaxis, :]   # (1, n_features)
        h, probs = self._forward(X)
        probs_1d = probs[0]            # (n_primitives,)
        return {name: float(p) for name, p in zip(self.primitive_names, probs_1d)}

    # ------------------------------------------------------------------
    def save_weights(self, path: str) -> None:
        """
        Save recognition network weights + metadata to an .npz file.

        Saves: W1, b1, W2, b2, vW1, vb1, vW2, vb2 (numpy arrays)
               plus primitive_names as a JSON sidecar.
        """
        if not self._initialized:
            return
        import json
        from pathlib import Path
        np.savez(
            path,
            W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2,
            vW1=self.vW1, vb1=self.vb1, vW2=self.vW2, vb2=self.vb2,
        )
        # Save primitive name list alongside weights
        meta_path = Path(path).with_suffix('.json')
        with open(meta_path, 'w') as f:
            json.dump({
                'primitive_names': self.primitive_names,
                'n_features': self.n_features,
                'hidden_dim': self.hidden_dim,
            }, f, indent=2)

    # ------------------------------------------------------------------
    def load_weights(self, path: str) -> bool:
        """
        Load recognition network weights from an .npz checkpoint.

        Returns True if weights loaded successfully, False otherwise.
        Also loads primitive_names from the JSON sidecar.
        """
        import json
        from pathlib import Path
        npz_path = Path(path)
        meta_path = npz_path.with_suffix('.json')

        if not npz_path.exists():
            return False

        try:
            data = np.load(npz_path)
            self.W1 = data['W1']
            self.b1 = data['b1']
            self.W2 = data['W2']
            self.b2 = data['b2']
            self.vW1 = data['vW1']
            self.vb1 = data['vb1']
            self.vW2 = data['vW2']
            self.vb2 = data['vb2']

            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                self.primitive_names = meta.get('primitive_names', self.primitive_names)
                self.n_features = meta.get('n_features', self.n_features)
                self.hidden_dim = meta.get('hidden_dim', self.hidden_dim)

            self._initialized = True
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Legacy API shim — so old code that calls rank_strategies / record_outcome
    # / get_features still works without modification.
    # ------------------------------------------------------------------

    def rank_strategies(self, task) -> list[str]:
        """Return primitives ranked by predicted probability (desc)."""
        priors = self.predict_primitive_priors(task)
        if not priors:
            return list(self.primitive_names)
        return sorted(priors.keys(), key=lambda k: -priors[k])

    def record_outcome(self, strategy: str, solved: bool):
        """No-op — kept for backwards compatibility."""
        pass

    def get_features(self, task) -> dict:
        """Extract raw features dict for a task."""
        return TaskFeatures(task).features


# Backwards-compat alias so wake_sleep.py can import RecognitionModel unchanged.
RecognitionModel = NeuralRecognitionNet
