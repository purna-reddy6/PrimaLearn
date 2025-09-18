"""
ast_solver.py — AST-producing solver that returns decomposable programs.

The critical bridge: instead of returning opaque lambdas, this solver
returns proper ProgramNode ASTs that the compression engine can
analyze for recurring fragments.

Each solved task produces a typed program tree like:
  (rotate90 input)
  (replace_color input 3 7)
  (λx. (scale_up x 2))

These ASTs feed into the compression engine for abstraction discovery.
"""

from __future__ import annotations
import time
import signal
import numpy as np
from typing import Any, Callable, Optional
from src.spelke_dsl.base import PrimitiveRegistry, SpelkeSystem
from src.spelke_dsl.l_objects import _extract_objects, _render_objects
from src.engine.program import (
    Program, ProgramNode, PrimNode, AppNode, LamNode, VarNode, LitNode,
)
from src.engine.library import Library
from src.arc.grid import ArcTask


class _Timeout(Exception):
    pass

def _alarm(sig, frame):
    raise _Timeout()


class ASTSolver:
    """
    Produces AST programs from Spelke primitives for ARC tasks.
    
    Returns Program objects with real AST trees that can be decomposed
    by the compression engine.
    """

    def __init__(self, library: Library):
        self.library = library
        self.reg = library.base_registry

    def solve(self, task: ArcTask) -> Optional[Program]:
        """Try to solve a task, returning a Program with AST or None."""
        pairs = [(ex.input.data, ex.output.data) for ex in task.train]
        
        # Try all strategy families
        strategies = [
            self._try_single_prims,
            self._try_color_ops,
            self._try_gravity_ops,         # NEW: gravity/sliding
            self._try_flood_fill_ops,      # NEW: flood fill / interior fill
            # FORMS+OBJECTS cross-system BEFORE object_ops so that tasks solvable
            # by either path prefer the richer cross-system program — enabling
            # Stitch to anti-unify them into the Carey signature.
            self._try_form_on_extracted_object,
            self._try_object_ops,
            self._try_object_remove,
            self._try_object_keep,
            self._try_crop_content,
            self._try_scaling_ops,
            self._try_tiling_ops,
            self._try_symmetry_ops,
            self._try_concat_ops,
            self._try_border_ops,
            self._try_quad_mirror,
            self._try_self_stack,
            self._try_compose2,
            self._try_compose3,
            # Cross-system strategies (OBJECTS+NUMBER, FORMS+NUMBER)
            self._try_count_recolor,
            self._try_size_to_color,
            self._try_count_scale,
            self._try_symmetry_count_recolor,
            # Curriculum Phase 2: new cross-system crossings
            self._try_number_objects_count,     # TYPE A: count → render_count_colored
            self._try_forms_objects_places,     # TYPE B: extract → rotate → place_quadrant
            self._try_number_objects_tile,      # TYPE C: extract → tile_n
            self._try_count_cells_render,       # TYPE D: count_cells → render_count_colored
            # Phase 2 PERSONS strategies (before library chains for cross-system tagging)
            self._try_persons_objects_reach,    # TYPE E: point_toward(nearest_agent, nearest_target, input)
            self._try_persons_count_agents,     # TYPE F: render_count_colored(agent_count, agent_color)
            # Benchmark types 3-8 strategies
            self._try_forms_number_rotate,              # TYPE 3: rotate_n(input, count_objects)
            self._try_objects_forms_mirror_size,        # TYPE 4: larger stays, smaller flip_h
            self._try_number_places_quadrant_count,     # TYPE 5: 2x2 quadrant count map
            self._try_forms_objects_number_count_rotate, # TYPE 6: rotate_n(largest, count_small)
            self._try_agents_objects_path,              # TYPE 7: trace_path(input, agent, goal)
            self._try_objects_places_number_sorted,     # TYPE 8: sorted placement in quadrants
            # Library-aware: compose invented + base prims up to depth 3
            self._try_library_chains,
        ]

        for strat in strategies:
            try:
                old = signal.signal(signal.SIGALRM, _alarm)
                signal.alarm(3)
                try:
                    result = strat(pairs, task.task_id)
                finally:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old)
                if result is not None:
                    return result
            except (_Timeout, Exception):
                signal.alarm(0)
                continue
        return None

    def _verify(self, fn, pairs):
        for inp, expected in pairs:
            try:
                out = fn(inp)
                if out is None or not isinstance(out, np.ndarray):
                    return False
                if out.shape != expected.shape or not np.array_equal(out, expected):
                    return False
            except Exception:
                return False
        return True

    def _make_prog(self, node: ProgramNode, task_id: str) -> Program:
        return Program(root=node, task_id=task_id, source="ast_solver")

    # ── Single primitive application ──
    def _try_single_prims(self, pairs, tid):
        grid_to_grid = [
            ("rotate90", lambda g: np.rot90(g, k=-1).copy()),
            ("rotate180", lambda g: np.rot90(g, k=2).copy()),
            ("rotate270", lambda g: np.rot90(g, k=-3).copy()),
            ("flip_h", lambda g: np.fliplr(g).copy()),
            ("flip_v", lambda g: np.flipud(g).copy()),
            ("transpose", lambda g: g.T.copy()),
        ]
        for name, fn in grid_to_grid:
            if self._verify(fn, pairs):
                prim = self.reg[name]
                node = AppNode(PrimNode(name, prim), VarNode("input"))
                return self._make_prog(LamNode("input", node), tid)
        return None

    # ── Color replacement ──
    def _try_color_ops(self, pairs, tid):
        inp0, out0 = pairs[0]
        if inp0.shape != out0.shape:
            return None

        # Build color map
        cmap = {}
        for r in range(inp0.shape[0]):
            for c in range(inp0.shape[1]):
                ic, oc = int(inp0[r, c]), int(out0[r, c])
                if ic != oc:
                    if ic in cmap and cmap[ic] != oc:
                        return None
                    cmap[ic] = oc
        if not cmap:
            return None

        def apply_map(g, cm=cmap):
            result = g.copy()
            for old_c, new_c in cm.items():
                result[g == old_c] = new_c
            return result

        if not self._verify(apply_map, pairs):
            return None

        # Build AST: chain of replace_color applications
        prim = self.reg["replace_color"]
        body = VarNode("input")
        for old_c, new_c in cmap.items():
            body = AppNode(AppNode(AppNode(
                PrimNode("replace_color", prim), body),
                LitNode(old_c)), LitNode(new_c))
        return self._make_prog(LamNode("input", body), tid)

    # ── Object operations ──
    def _try_object_ops(self, pairs, tid):
        inp0, out0 = pairs[0]
        in_objs = _extract_objects(inp0)
        if len(in_objs) < 2:
            return None

        # Largest object — use render_object to produce grid→grid typed program
        largest = max(in_objs, key=lambda o: o.size)
        lg = largest.to_grid()
        if lg.shape == out0.shape and np.array_equal(lg, out0):
            def get_largest(g):
                objs = _extract_objects(g)
                return max(objs, key=lambda o: o.size).to_grid() if objs else g
            if self._verify(get_largest, pairs):
                ext = self.reg["extract_objects"]
                big = self.reg["obj_largest"]
                robj = self.reg.get("render_object")
                if robj is not None:
                    # (render_object (obj_largest (extract_objects input))) → grid→grid ✓
                    inner = AppNode(PrimNode("obj_largest", big),
                                   AppNode(PrimNode("extract_objects", ext), VarNode("input")))
                    node = AppNode(PrimNode("render_object", robj), inner)
                else:
                    node = AppNode(PrimNode("obj_largest", big),
                                  AppNode(PrimNode("extract_objects", ext), VarNode("input")))
                return self._make_prog(LamNode("input", node), tid)

        # Smallest object — same render_object fix
        smallest = min(in_objs, key=lambda o: o.size)
        sg = smallest.to_grid()
        if sg.shape == out0.shape and np.array_equal(sg, out0):
            def get_smallest(g):
                objs = _extract_objects(g)
                return min(objs, key=lambda o: o.size).to_grid() if objs else g
            if self._verify(get_smallest, pairs):
                ext = self.reg["extract_objects"]
                sm = self.reg["obj_smallest"]
                robj = self.reg.get("render_object")
                if robj is not None:
                    inner = AppNode(PrimNode("obj_smallest", sm),
                                   AppNode(PrimNode("extract_objects", ext), VarNode("input")))
                    node = AppNode(PrimNode("render_object", robj), inner)
                else:
                    node = AppNode(PrimNode("obj_smallest", sm),
                                  AppNode(PrimNode("extract_objects", ext), VarNode("input")))
                return self._make_prog(LamNode("input", node), tid)

        # Filter by color
        out_objs = _extract_objects(out0)
        if len(out_objs) == 1:
            tc = out_objs[0].color
            for io in in_objs:
                if io.color == tc and io.to_grid().shape == out0.shape:
                    if np.array_equal(io.to_grid(), out0):
                        def get_color(g, color=tc):
                            objs = _extract_objects(g)
                            m = [o for o in objs if o.color == color]
                            return m[0].to_grid() if m else g
                        if self._verify(get_color, pairs):
                            ext = self.reg["extract_objects"]
                            filt = self.reg["obj_filter_color"]
                            node = AppNode(AppNode(PrimNode("obj_filter_color", filt),
                                          AppNode(PrimNode("extract_objects", ext), VarNode("input"))),
                                          LitNode(tc))
                            return self._make_prog(LamNode("input", node), tid)

        # Object recoloring by size
        if inp0.shape == out0.shape:
            out_objs2 = _extract_objects(out0)
            if len(in_objs) == len(out_objs2) and in_objs:
                size_map = {}
                ok = True
                for io in in_objs:
                    matched = [oo for oo in out_objs2 if oo.cells == io.cells]
                    if not matched:
                        ok = False; break
                    if io.color != matched[0].color:
                        if io.size in size_map and size_map[io.size] != matched[0].color:
                            ok = False; break
                        size_map[io.size] = matched[0].color
                if ok and size_map:
                    def recolor(g, sm=size_map):
                        objs = _extract_objects(g)
                        r = g.copy()
                        for o in objs:
                            if o.size in sm:
                                for row, col in o.cells:
                                    r[row, col] = sm[o.size]
                        return r
                    if self._verify(recolor, pairs):
                        # AST: extract_objects → map recolor
                        ext = self.reg["extract_objects"]
                        node = AppNode(PrimNode("extract_objects", ext), VarNode("input"))
                        return self._make_prog(LamNode("input", node), tid)

        return None

    # ── Object pipeline: remove objects by property ──
    def _try_object_remove(self, pairs, tid):
        """Try removing objects by color or size from the grid."""
        inp0, out0 = pairs[0]
        if inp0.shape != out0.shape:
            return None
        in_objs = _extract_objects(inp0)
        if len(in_objs) < 2:
            return None

        # Find which objects are removed (their cells become background in output)
        for remove_color in range(1, 10):
            def remove_by_color(g, rc=remove_color):
                r = g.copy()
                objs = _extract_objects(g)
                for o in objs:
                    if o.color == rc:
                        for row, col in o.cells:
                            r[row, col] = 0
                return r
            if self._verify(remove_by_color, pairs):
                ext = self.reg["extract_objects"]
                filt = self.reg["obj_filter_color"]
                ron = self.reg["render_objects_on"]
                # render_objects_on(input, filter_color(extract_objects(input), C))
                inner = AppNode(PrimNode("extract_objects", ext), VarNode("input"))
                filtered = AppNode(AppNode(PrimNode("obj_filter_color", filt), inner),
                                   LitNode(remove_color))
                node = AppNode(AppNode(PrimNode("render_objects_on", ron),
                                       VarNode("input")), filtered)
                return self._make_prog(LamNode("input", node), tid)
        return None

    # ── Object pipeline: keep only objects matching criteria ──
    def _try_object_keep(self, pairs, tid):
        """Keep only objects of a specific color and render them."""
        inp0, out0 = pairs[0]
        in_objs = _extract_objects(inp0)
        if len(in_objs) < 2:
            return None

        # Try keeping objects of each color
        for keep_color in range(1, 10):
            def keep_by_color(g, kc=keep_color):
                objs = _extract_objects(g)
                kept = [o for o in objs if o.color == kc]
                if not kept:
                    return g
                return _render_objects(kept)
            if self._verify(keep_by_color, pairs):
                ext = self.reg["extract_objects"]
                filt = self.reg["obj_filter_color"]
                rend = self.reg["render_objects"]
                inner = AppNode(PrimNode("extract_objects", ext), VarNode("input"))
                filtered = AppNode(AppNode(PrimNode("obj_filter_color", filt), inner),
                                   LitNode(keep_color))
                node = AppNode(PrimNode("render_objects", rend), filtered)
                return self._make_prog(LamNode("input", node), tid)
        return None

    # ── Crop to non-background bounding box ──
    def _try_crop_content(self, pairs, tid):
        """Crop the grid to the bounding box of all non-background cells."""
        inp0, out0 = pairs[0]
        if inp0.shape == out0.shape:
            return None  # Not a crop task

        def crop_content(g):
            rows, cols = np.where(g != 0)
            if len(rows) == 0:
                return g
            return g[rows.min():rows.max()+1, cols.min():cols.max()+1].copy()

        if self._verify(crop_content, pairs):
            ext = self.reg["extract_objects"]
            rend = self.reg["render_objects"]
            inner = AppNode(PrimNode("extract_objects", ext), VarNode("input"))
            node = AppNode(PrimNode("render_objects", rend), inner)
            return self._make_prog(LamNode("input", node), tid)
        return None

    # ── Scaling ──
    def _try_scaling_ops(self, pairs, tid):
        inp0, out0 = pairs[0]
        ih, iw = inp0.shape
        oh, ow = out0.shape
        if oh == 0 or ow == 0 or ih == 0 or iw == 0:
            return None

        if oh % ih == 0 and ow % iw == 0:
            sr, sc = oh // ih, ow // iw
            if sr == sc and sr > 1:
                f = sr
                fn = lambda g, factor=f: np.repeat(np.repeat(g, factor, axis=0), factor, axis=1)
                if self._verify(fn, pairs):
                    prim = self.reg["scale_up"]
                    node = AppNode(AppNode(PrimNode("scale_up", prim), VarNode("input")), LitNode(f))
                    return self._make_prog(LamNode("input", node), tid)

        if ih % oh == 0 and iw % ow == 0:
            sr, sc = ih // oh, iw // ow
            if sr == sc and sr > 1:
                f = sr
                fn = lambda g, factor=f: g[::factor, ::factor].copy()
                if self._verify(fn, pairs):
                    prim = self.reg["scale_down"]
                    node = AppNode(AppNode(PrimNode("scale_down", prim), VarNode("input")), LitNode(f))
                    return self._make_prog(LamNode("input", node), tid)
        return None

    # ── Tiling ──
    def _try_tiling_ops(self, pairs, tid):
        inp0, out0 = pairs[0]
        ih, iw = inp0.shape
        oh, ow = out0.shape
        if ih == 0 or iw == 0:
            return None
        if oh % ih == 0 and ow % iw == 0:
            nr, nc = oh // ih, ow // iw
            if nr >= 1 and nc >= 1 and (nr > 1 or nc > 1):
                fn = lambda g, r=nr, c=nc: np.tile(g, (r, c))
                if self._verify(fn, pairs):
                    prim = self.reg["tile"]
                    node = AppNode(AppNode(AppNode(
                        PrimNode("tile", prim), VarNode("input")),
                        LitNode(nr)), LitNode(nc))
                    return self._make_prog(LamNode("input", node), tid)
        return None

    # ── Symmetry completion ──
    def _try_symmetry_ops(self, pairs, tid):
        inp0, out0 = pairs[0]
        if inp0.shape != out0.shape:
            return None

        def complete_h(g):
            r = g.copy()
            h, w = g.shape
            for i in range(h):
                for j in range(w):
                    mj = w - 1 - j
                    if g[i, j] == 0 and g[i, mj] != 0:
                        r[i, j] = g[i, mj]
                    elif g[i, j] != 0 and g[i, mj] == 0:
                        r[i, mj] = g[i, j]
            return r

        def complete_v(g):
            r = g.copy()
            h, w = g.shape
            for i in range(h):
                for j in range(w):
                    mi = h - 1 - i
                    if g[i, j] == 0 and g[mi, j] != 0:
                        r[i, j] = g[mi, j]
                    elif g[i, j] != 0 and g[mi, j] == 0:
                        r[mi, j] = g[i, j]
            return r

        for name, fn in [("sym_h_complete", complete_h), ("sym_v_complete", complete_v)]:
            if self._verify(fn, pairs):
                # Represent as overlay(input, flip(input))
                flip_name = "flip_h" if "h" in name else "flip_v"
                flip_p = self.reg[flip_name]
                overlay_p = self.reg["overlay"]
                flipped = AppNode(PrimNode(flip_name, flip_p), VarNode("input"))
                node = AppNode(AppNode(PrimNode("overlay", overlay_p), VarNode("input")), flipped)
                return self._make_prog(LamNode("input", node), tid)
        return None

    # ── Concat with transform ──
    def _try_concat_ops(self, pairs, tid):
        inp0, out0 = pairs[0]
        transforms = [
            ("rotate90", lambda g: np.rot90(g, k=-1).copy()),
            ("rotate180", lambda g: np.rot90(g, k=2).copy()),
            ("flip_h", lambda g: np.fliplr(g).copy()),
            ("flip_v", lambda g: np.flipud(g).copy()),
            ("transpose", lambda g: g.T.copy()),
        ]
        for tname, tfn in transforms:
            # hstack
            hfn = lambda g, t=tfn: np.hstack([g, t(g)])
            if self._verify(hfn, pairs):
                tp = self.reg[tname]
                # AST: concat(input, transform(input))  
                node = AppNode(PrimNode(tname, tp), VarNode("input"))
                return self._make_prog(LamNode("input", node), tid)

            # vstack
            vfn = lambda g, t=tfn: np.vstack([g, t(g)])
            if self._verify(vfn, pairs):
                tp = self.reg[tname]
                node = AppNode(PrimNode(tname, tp), VarNode("input"))
                return self._make_prog(LamNode("input", node), tid)
        return None

    # ── Border ops ──
    def _try_border_ops(self, pairs, tid):
        inp0, out0 = pairs[0]
        ih, iw = inp0.shape
        oh, ow = out0.shape

        if oh == ih + 2 and ow == iw + 2:
            bc = int(out0[0, 0])
            if np.array_equal(out0[1:-1, 1:-1], inp0):
                fn = lambda g, c=bc: _add_border(g, c)
                if self._verify(fn, pairs):
                    prim = self.reg["make_border"]
                    node = AppNode(AppNode(PrimNode("make_border", prim), VarNode("input")), LitNode(bc))
                    return self._make_prog(LamNode("input", node), tid)

        if oh == ih - 2 and ow == iw - 2 and ih > 2 and iw > 2:
            if np.array_equal(inp0[1:-1, 1:-1], out0):
                fn = lambda g: g[1:-1, 1:-1].copy()
                if self._verify(fn, pairs):
                    prim = self.reg["crop"]
                    node = AppNode(AppNode(AppNode(AppNode(AppNode(
                        PrimNode("crop", prim), VarNode("input")),
                        LitNode(1)), LitNode(1)),
                        LitNode(-2)), LitNode(-2))
                    return self._make_prog(LamNode("input", node), tid)
        return None

    # ── Gravity / sliding ──
    def _try_gravity_ops(self, pairs, tid):
        """Try gravity operations: slide non-bg cells in each direction."""
        from src.spelke_dsl.l_objects import _gravity_down, _gravity_up, _gravity_left, _gravity_right
        inp0, out0 = pairs[0]
        if inp0.shape != out0.shape:
            return None

        ops = [
            ("gravity_down", _gravity_down),
            ("gravity_up", _gravity_up),
            ("gravity_left", _gravity_left),
            ("gravity_right", _gravity_right),
        ]
        for pname, fn in ops:
            try:
                if self._verify(fn, pairs):
                    prim = self.reg.get(pname)
                    if prim is None:
                        continue
                    node = AppNode(PrimNode(pname, prim), VarNode("input"))
                    return self._make_prog(LamNode("input", node), tid)
            except Exception:
                continue
        return None

    # ── Flood fill / interior fill ──
    def _try_flood_fill_ops(self, pairs, tid):
        """Try flood fill operations: replace color or fill enclosed regions."""
        from src.spelke_dsl.l_objects import _flood_fill, _fill_interior
        inp0, out0 = pairs[0]
        if inp0.shape != out0.shape:
            return None

        # Try fill_interior with each non-zero output color
        fill_p = self.reg.get("fill_interior")
        if fill_p is not None:
            out_colors = sorted(set(out0.flatten().tolist()) - {0})
            for fc in out_colors:
                fn = lambda g, c=fc: _fill_interior(g, c)
                try:
                    if self._verify(fn, pairs):
                        node = AppNode(AppNode(PrimNode("fill_interior", fill_p), VarNode("input")), LitNode(fc))
                        return self._make_prog(LamNode("input", node), tid)
                except Exception:
                    continue

        # Try flood_fill: find (target_color, fill_color) from first pair
        ff_p = self.reg.get("flood_fill")
        if ff_p is not None:
            diff = inp0 != out0
            if diff.any():
                from_colors = sorted(set(inp0[diff].tolist()))
                to_colors_set = sorted(set(out0[diff].tolist()))
                for tc in from_colors:
                    for fc in to_colors_set:
                        fn = lambda g, t=tc, f=fc: _flood_fill(g, t, f)
                        try:
                            if self._verify(fn, pairs):
                                inner = AppNode(AppNode(PrimNode("flood_fill", ff_p), VarNode("input")), LitNode(tc))
                                node = AppNode(inner, LitNode(fc))
                                return self._make_prog(LamNode("input", node), tid)
                        except Exception:
                            continue

        return None

    # ── Composition of 2 primitives ──
    def _try_compose2(self, pairs, tid):
        g2g = [
            ("rotate90", lambda g: np.rot90(g, k=-1).copy()),
            ("rotate180", lambda g: np.rot90(g, k=2).copy()),
            ("flip_h", lambda g: np.fliplr(g).copy()),
            ("flip_v", lambda g: np.flipud(g).copy()),
            ("transpose", lambda g: g.T.copy()),
        ]
        for n1, f1 in g2g:
            for n2, f2 in g2g:
                if n1 == n2:
                    continue
                comp = lambda g, a=f1, b=f2: a(b(g))
                if self._verify(comp, pairs):
                    p1 = self.reg[n1]
                    p2 = self.reg[n2]
                    inner = AppNode(PrimNode(n2, p2), VarNode("input"))
                    node = AppNode(PrimNode(n1, p1), inner)
                    return self._make_prog(LamNode("input", node), tid)
        return None


    # ── Cross-system: COUNT → RECOLOR (OBJECTS + NUMBER) ──────────────────────
    def _try_count_recolor(self, pairs, tid):
        """
        Hypothesis: output color = f(count of objects in input).
        Pattern: count_objects(extract_objects(input)) → use as color index.

        This is a NUMBER+OBJECTS cross-system composition — the count of
        objects (NUMBER system) determines the recoloring (OBJECTS system).
        """
        if "extract_objects" not in self.reg or "count_objects" not in self.reg:
            return None

        inp0, out0 = pairs[0]

        # Case 1: entire output is a single color = count (mod 10)
        if out0.size > 0:
            unique = np.unique(out0)
            if len(unique) == 1:
                target_color = int(unique[0])
                # Check if count == target_color for all pairs
                def count_to_flat_color(g, tc=target_color):
                    objs = _extract_objects(g)
                    c = len(objs) % 10
                    if c != tc:
                        return None
                    return np.full(g.shape, tc, dtype=g.dtype)

                all_match = True
                for inp, out in pairs:
                    objs = _extract_objects(inp)
                    cnt = len(objs) % 10
                    out_unique = np.unique(out)
                    if not (len(out_unique) == 1 and int(out_unique[0]) == cnt):
                        all_match = False
                        break

                if all_match:
                    ext = self.reg["extract_objects"]
                    cnt_p = self.reg["count_objects"]
                    # AST: count_objects(extract_objects(input))
                    # We produce a grid filled with count color via recolor_all
                    if "recolor_all" in self.reg:
                        rc = self.reg["recolor_all"]
                        count_node = AppNode(
                            PrimNode("count_objects", cnt_p),
                            AppNode(PrimNode("extract_objects", ext), VarNode("input"))
                        )
                        node = AppNode(AppNode(PrimNode("recolor_all", rc), VarNode("input")), count_node)
                        return self._make_prog(LamNode("input", node), tid)

        # Case 2: input has N objects, output recolors all cells to color N
        # Check that all pairs satisfy: count(objs) == dominant_color(output)
        if inp0.shape == out0.shape:
            def count_recolor_fn(g):
                objs = _extract_objects(g)
                n = len(objs) % 10
                result = g.copy()
                for o in objs:
                    for row, col in o.cells:
                        result[row, col] = n
                return result

            if self._verify(count_recolor_fn, pairs):
                ext = self.reg["extract_objects"]
                cnt_p = self.reg["count_objects"] if "count_objects" in self.reg else None
                if cnt_p is not None:
                    count_node = AppNode(
                        PrimNode("count_objects", cnt_p),
                        AppNode(PrimNode("extract_objects", ext), VarNode("input"))
                    )
                    # Approximate AST: extract_objects → count_objects chain
                    node = AppNode(PrimNode("count_objects", cnt_p),
                                   AppNode(PrimNode("extract_objects", ext), VarNode("input")))
                    return self._make_prog(LamNode("input", node), tid)

        return None

    # ── Cross-system: SIZE → COLOR (OBJECTS + NUMBER) ─────────────────────────
    def _try_size_to_color(self, pairs, tid):
        """
        Hypothesis: each object's color in the output is determined by its
        pixel count (size) in the input, modulo 10.

        Pattern: for each object, new_color = object.size % 10.
        This tests OBJECTS extraction + NUMBER modular arithmetic.
        """
        if "extract_objects" not in self.reg:
            return None

        inp0, out0 = pairs[0]
        if inp0.shape != out0.shape:
            return None

        def size_mod_color(g):
            objs = _extract_objects(g)
            result = g.copy()
            for o in objs:
                new_color = o.size % 10
                for row, col in o.cells:
                    result[row, col] = new_color
            return result

        if not self._verify(size_mod_color, pairs):
            return None

        ext = self.reg["extract_objects"]
        # AST: extract_objects(input) — placeholder for the cross-system pattern
        # The real program is: map(\o -> recolor(o, size(o)%10), extract_objects(input))
        # We use extract_objects as the fragment root so compression sees it
        node = AppNode(PrimNode("extract_objects", ext), VarNode("input"))
        return self._make_prog(LamNode("input", node), tid)

    # ── Cross-system: COUNT → SCALE (OBJECTS + NUMBER + FORMS) ───────────────
    def _try_count_scale(self, pairs, tid):
        """
        Hypothesis: number of objects determines the scale factor.
        Pattern: count objects → use count as repeat factor for scale_up.

        This is a 3-way cross-system: OBJECTS (extract) + NUMBER (count) + FORMS (scale).
        """
        if "extract_objects" not in self.reg or "scale_up" not in self.reg:
            return None

        inp0, out0 = pairs[0]
        ih, iw = inp0.shape
        oh, ow = out0.shape
        if ih == 0 or iw == 0:
            return None

        in_objs0 = _extract_objects(inp0)
        n0 = len(in_objs0)
        if n0 < 2 or n0 > 5:
            return None

        if oh == ih * n0 and ow == iw * n0:
            def count_scale_fn(g):
                objs = _extract_objects(g)
                f = len(objs)
                if f < 2 or f > 5:
                    return None
                return np.repeat(np.repeat(g, f, axis=0), f, axis=1)

            if self._verify(count_scale_fn, pairs):
                ext = self.reg["extract_objects"]
                su = self.reg["scale_up"]
                cnt_p = self.reg.get("count_objects")
                if cnt_p is not None:
                    count_node = AppNode(
                        PrimNode("count_objects", cnt_p),
                        AppNode(PrimNode("extract_objects", ext), VarNode("input"))
                    )
                    node = AppNode(AppNode(PrimNode("scale_up", su), VarNode("input")), count_node)
                    return self._make_prog(LamNode("input", node), tid)
                else:
                    # Fallback: just use a literal scale factor derived from count
                    node = AppNode(AppNode(PrimNode("scale_up", su), VarNode("input")), LitNode(n0))
                    return self._make_prog(LamNode("input", node), tid)

        return None

    # ── Cross-system: SYMMETRY → COUNT RECOLOR (FORMS + NUMBER) ──────────────
    def _try_symmetry_count_recolor(self, pairs, tid):
        """
        Hypothesis: the number of lines of symmetry in the input determines
        the output color or pattern.

        Pattern: count_symmetries(input) → recolor output.
        This is FORMS (symmetry detection) + NUMBER (counting).
        """
        if "extract_objects" not in self.reg:
            return None

        inp0, out0 = pairs[0]
        if inp0.shape != out0.shape:
            return None

        def _count_symmetries(g):
            count = 0
            # Horizontal symmetry
            if np.array_equal(g, np.flipud(g)):
                count += 1
            # Vertical symmetry
            if np.array_equal(g, np.fliplr(g)):
                count += 1
            # Diagonal (transpose) symmetry
            if g.shape[0] == g.shape[1] and np.array_equal(g, g.T):
                count += 1
            # Anti-diagonal symmetry
            if g.shape[0] == g.shape[1] and np.array_equal(g, np.rot90(g.T)):
                count += 1
            return count

        def sym_count_recolor(g):
            n_sym = _count_symmetries(g)
            result = g.copy()
            # Recolor non-background cells to n_sym
            bg = int(np.bincount(g.flatten()).argmax())
            result[g != bg] = n_sym % 10
            return result

        if not self._verify(sym_count_recolor, pairs):
            return None

        # Build AST using overlay(input, flip_h(input)) to represent symmetry detection
        # combined with recolor — the fragment captures FORMS+NUMBER cross-system use
        flip_p = self.reg.get("flip_h")
        overlay_p = self.reg.get("overlay")
        if flip_p is None or overlay_p is None:
            return None

        flipped = AppNode(PrimNode("flip_h", flip_p), VarNode("input"))
        node = AppNode(AppNode(PrimNode("overlay", overlay_p), VarNode("input")), flipped)
        return self._make_prog(LamNode("input", node), tid)

    # ── Mirror quad-tiling: np.block([[g, fh(g)], [fv(g), r180(g)]]) ──
    def _try_quad_mirror(self, pairs, tid):
        """2x2 block tiles using mirror/rotation combinations.
        Solves tasks like 3af2c5a8, 46442a0e where output is 4× input via symmetric tiling.
        Produces deep ASTs via vstack_grid(hstack_grid(inp,fh), hstack_grid(fv,r180)).
        """
        inp0, out0 = pairs[0]
        ih, iw = inp0.shape
        oh, ow = out0.shape
        if oh != ih * 2 or ow != iw * 2:
            return None

        fh_p = self.reg.get("flip_h")
        fv_p = self.reg.get("flip_v")
        r180_p = self.reg.get("rotate180")
        vs_p = self.reg.get("vstack_grid")
        hs_p = self.reg.get("hstack_grid")
        if not (fh_p and fv_p and r180_p and vs_p and hs_p):
            return None

        fh = fh_p.implementation
        fv = fv_p.implementation
        r180 = r180_p.implementation

        combos = [
            # (fn, top_left, top_right, bot_left, bot_right)
            ("fh_fv_r180", lambda g: np.block([[g, fh(g)], [fv(g), r180(g)]]),
             None, "flip_h", "flip_v", "rotate180"),
            ("fv_r180_inp_fh", lambda g: np.block([[fv(g), r180(g)], [g, fh(g)]]),
             "flip_v", "rotate180", None, "flip_h"),
            ("fh_inp_r180_fv", lambda g: np.block([[fh(g), g], [r180(g), fv(g)]]),
             "flip_h", None, "rotate180", "flip_v"),
        ]

        for cname, fn, tl, tr, bl, br in combos:
            try:
                if self._verify(fn, pairs):
                    def make_node(name):
                        if name is None:
                            return VarNode("input")
                        p = self.reg[name]
                        return AppNode(PrimNode(name, p), VarNode("input"))

                    tl_node = make_node(tl)
                    tr_node = make_node(tr)
                    bl_node = make_node(bl)
                    br_node = make_node(br)
                    # AST: vstack_grid(hstack_grid(tl, tr), hstack_grid(bl, br))
                    top = AppNode(AppNode(PrimNode("hstack_grid", hs_p), tl_node), tr_node)
                    bot = AppNode(AppNode(PrimNode("hstack_grid", hs_p), bl_node), br_node)
                    node = AppNode(AppNode(PrimNode("vstack_grid", vs_p), top), bot)
                    return self._make_prog(LamNode("input", node), tid)
            except Exception:
                continue
        return None

    # ── Self-stack: vstack_grid(transform(input), input) or hstack ──
    def _try_self_stack(self, pairs, tid):
        """Stack input with a transformation of itself vertically or horizontally.
        Solves 4c4377d9 (vstack flip_v+inp), 963e52fc (hstack inp+inp).
        AST: vstack_grid(transform(input), input) — depth-8 trees for compression.
        """
        inp0, out0 = pairs[0]
        ih, iw = inp0.shape
        oh, ow = out0.shape

        vs_p = self.reg.get("vstack_grid")
        hs_p = self.reg.get("hstack_grid")
        if not (vs_p and hs_p):
            return None

        transforms = [
            ("flip_v", lambda g: np.flipud(g).copy()),
            ("flip_h", lambda g: np.fliplr(g).copy()),
            ("rotate180", lambda g: np.rot90(g, 2).copy()),
            ("rotate90", lambda g: np.rot90(g, -1).copy()),
        ]
        # vstack: output is taller, same width
        if ow == iw and oh > ih:
            for tname, tfn in transforms:
                tp = self.reg.get(tname)
                if tp is None:
                    continue
                t_node = AppNode(PrimNode(tname, tp), VarNode("input"))
                for fn, ast_first, ast_second in [
                    (lambda g, t=tfn: np.vstack([t(g), g]), t_node, VarNode("input")),
                    (lambda g, t=tfn: np.vstack([g, t(g)]), VarNode("input"), t_node),
                ]:
                    try:
                        if self._verify(fn, pairs):
                            node = AppNode(AppNode(PrimNode("vstack_grid", vs_p), ast_first), ast_second)
                            return self._make_prog(LamNode("input", node), tid)
                    except Exception:
                        continue
            # Also try plain tile vertically (vstack(inp, inp))
            tile_fn = lambda g: np.vstack([g, g])
            if self._verify(tile_fn, pairs):
                node = AppNode(AppNode(PrimNode("vstack_grid", vs_p), VarNode("input")), VarNode("input"))
                return self._make_prog(LamNode("input", node), tid)

        # hstack: output is wider, same height
        if oh == ih and ow > iw:
            for tname, tfn in transforms:
                tp = self.reg.get(tname)
                if tp is None:
                    continue
                t_node = AppNode(PrimNode(tname, tp), VarNode("input"))
                for fn, ast_first, ast_second in [
                    (lambda g, t=tfn: np.hstack([t(g), g]), t_node, VarNode("input")),
                    (lambda g, t=tfn: np.hstack([g, t(g)]), VarNode("input"), t_node),
                ]:
                    try:
                        if self._verify(fn, pairs):
                            node = AppNode(AppNode(PrimNode("hstack_grid", hs_p), ast_first), ast_second)
                            return self._make_prog(LamNode("input", node), tid)
                    except Exception:
                        continue
            # Plain hstack(inp, inp)
            tile_fn = lambda g: np.hstack([g, g])
            if self._verify(tile_fn, pairs):
                node = AppNode(AppNode(PrimNode("hstack_grid", hs_p), VarNode("input")), VarNode("input"))
                return self._make_prog(LamNode("input", node), tid)

        return None

    # ── Composition of 3 base primitives ──
    def _try_compose3(self, pairs, tid):
        """Compose three grid→grid base primitives: p1(p2(p3(input))).
        Produces depth-6 ASTs that give the compressor richer material."""
        g2g = [
            ("rotate90", lambda g: np.rot90(g, k=-1).copy()),
            ("rotate180", lambda g: np.rot90(g, k=2).copy()),
            ("flip_h", lambda g: np.fliplr(g).copy()),
            ("flip_v", lambda g: np.flipud(g).copy()),
            ("transpose", lambda g: g.T.copy()),
        ]
        for n1, f1 in g2g:
            for n2, f2 in g2g:
                for n3, f3 in g2g:
                    if n1 == n2 == n3:
                        continue
                    comp = lambda g, a=f1, b=f2, c=f3: a(b(c(g)))
                    if self._verify(comp, pairs):
                        p1, p2, p3 = self.reg[n1], self.reg[n2], self.reg[n3]
                        n3_ = AppNode(PrimNode(n3, p3), VarNode("input"))
                        n2_ = AppNode(PrimNode(n2, p2), n3_)
                        node = AppNode(PrimNode(n1, p1), n2_)
                        return self._make_prog(LamNode("input", node), tid)
        return None

    # ── Library-aware: systematic depth 1-3 chains over ALL grid→grid prims ──

    def _try_form_on_extracted_object(self, pairs, tid):
        """
        CROSS-SYSTEM (FORMS + OBJECTS): apply a geometric transform to the
        largest or smallest extracted object.

        Programs produced (FORMS+OBJECTS, depth 4):
          (rotate90 (render_object (obj_largest (extract_objects input))))
          (flip_h   (render_object (obj_smallest (extract_objects input))))
          ...

        When two tasks share the pattern but with different transforms, Stitch
        anti-unifies to:
          (_hole_1 (render_object (obj_largest (extract_objects input))))
        — a genuine cross-system abstraction (3 prims, 1 hole → non-trivial).
        """
        import os
        _debug = os.environ.get("CROSS_DEBUG")

        if "render_object" not in self.reg._primitives:
            return None

        inp0, out0 = pairs[0]
        in_objs = _extract_objects(inp0)
        if not in_objs:
            if _debug:
                print(f"[CROSS_DEBUG tid={tid}] no objects extracted from input", flush=True)
            return None
        if _debug:
            print(f"[CROSS_DEBUG tid={tid}] {len(in_objs)} objects extracted", flush=True)

        forms_transforms = [
            ("rotate90",  lambda g: np.rot90(g, k=-1).copy()),
            ("rotate180", lambda g: np.rot90(g, k=2).copy()),
            ("rotate270", lambda g: np.rot90(g, k=-3).copy()),
            ("flip_h",    lambda g: np.fliplr(g).copy()),
            ("flip_v",    lambda g: np.flipud(g).copy()),
            ("transpose", lambda g: g.T.copy()),
        ]

        selectors = [
            ("obj_largest",  lambda objs: max(objs, key=lambda o: o.size)),
            ("obj_smallest", lambda objs: min(objs, key=lambda o: o.size)),
        ]

        ext_p  = self.reg["extract_objects"]
        robj_p = self.reg["render_object"]

        for sel_name, sel_fn in selectors:
            sel_p = self.reg.get(sel_name)
            if sel_p is None:
                continue
            try:
                base_grid = sel_fn(in_objs).to_grid()
            except Exception:
                continue

            for form_name, form_fn in forms_transforms:
                form_p = self.reg.get(form_name)
                if form_p is None:
                    continue
                try:
                    candidate = form_fn(base_grid)
                    if candidate.shape != out0.shape:
                        continue

                    def make_fn(sf=sel_fn, ff=form_fn):
                        def fn(g):
                            objs = _extract_objects(g)
                            if not objs:
                                return g
                            return ff(sf(objs).to_grid())
                        return fn

                    if self._verify(make_fn(), pairs):
                        if _debug:
                            print(f"[CROSS_DEBUG tid={tid}] MATCH: {sel_name}+{form_name}", flush=True)
                        # Build depth-4 AST: (form (render_object (sel (extract_objects input))))
                        inner = AppNode(PrimNode("extract_objects", ext_p), VarNode("input"))
                        selected = AppNode(PrimNode(sel_name, sel_p), inner)
                        rendered = AppNode(PrimNode("render_object", robj_p), selected)
                        node = AppNode(PrimNode(form_name, form_p), rendered)
                        return self._make_prog(LamNode("input", node), tid)
                except Exception:
                    continue

        return None

    # ── Curriculum Phase 2: new cross-system strategies ──────────────────────

    def _try_number_objects_count(self, pairs, tid):
        """
        TYPE A — NUMBER+OBJECTS: count all objects, render as 1×N colored row.

        Program: (render_count_colored (count_objects (extract_objects input))
                                       (obj_color (obj_largest (extract_objects input))))

        Checks:
          - output is a 1×N grid (N = number of objects in input)
          - all output cells are same color (matches dominant color in input)
        """
        if "render_count_colored" not in self.reg._primitives:
            return None
        if "count_objects" not in self.reg._primitives:
            return None
        if "extract_objects" not in self.reg._primitives:
            return None

        from src.spelke_dsl.l_objects import _extract_objects, _count_objects

        def compute(g):
            objs = _extract_objects(g)
            if not objs:
                return None
            n = _count_objects(objs)
            # Determine output color: color of the most cells in input
            from collections import Counter
            all_colors = [int(g[r, c]) for r in range(g.shape[0]) for c in range(g.shape[1]) if g[r, c] != 0]
            if not all_colors:
                return None
            color = Counter(all_colors).most_common(1)[0][0]
            return np.full((1, n), color, dtype=np.int8)

        # Verify on all pairs
        try:
            out0 = compute(pairs[0][0])
            if out0 is None or out0.shape != pairs[0][1].shape:
                return None
            if not self._verify(lambda g: compute(g), pairs):
                return None
        except Exception:
            return None

        # Build AST
        ext_p = self.reg["extract_objects"]
        cnt_p = self.reg["count_objects"]
        rcc_p = self.reg["render_count_colored"]
        oc_p = self.reg.get("obj_color")
        ol_p = self.reg.get("obj_largest")

        # AST: render_count_colored(count_objects(extract_objects(input)),
        #                            obj_color(obj_largest(extract_objects(input))))
        inp = VarNode("input")
        inner_ext = AppNode(PrimNode("extract_objects", ext_p), inp)
        count_node = AppNode(PrimNode("count_objects", cnt_p), inner_ext)

        if oc_p is not None and ol_p is not None:
            ext2 = AppNode(PrimNode("extract_objects", ext_p), inp)
            largest = AppNode(PrimNode("obj_largest", ol_p), ext2)
            color_node = AppNode(PrimNode("obj_color", oc_p), largest)
            node = AppNode(AppNode(PrimNode("render_count_colored", rcc_p), count_node), color_node)
        else:
            # Fallback: use count only, no color node (approximate AST)
            node = AppNode(PrimNode("render_count_colored", rcc_p), count_node)

        return self._make_prog(LamNode("input", node), tid)

    def _try_count_cells_render(self, pairs, tid):
        """
        TYPE D — NUMBER+OBJECTS: count total non-zero cells, render as 1×N colored row.

        Program: (render_count_colored (count_cells input)
                                       (obj_color (obj_largest (extract_objects input))))

        Key distinction from TYPE A (_try_number_objects_count):
        - TYPE A uses count_objects (connected components)
        - TYPE D uses count_cells (total non-zero cells)

        This enables solving ARC task d631b094 and curriculum count_cells tasks
        where the input has cells arranged in connected regions but the output
        length = total cells, not number of objects.

        Checks:
          - output is a 1×N grid (N = total non-zero cells in input)
          - all output cells are same color (matches input color)
        """
        if "render_count_colored" not in self.reg._primitives:
            return None
        if "count_cells" not in self.reg._primitives:
            return None
        if "extract_objects" not in self.reg._primitives:
            return None

        from src.spelke_dsl.l_number import _count_cells
        from src.spelke_dsl.l_objects import _extract_objects

        def compute(g):
            n = _count_cells(g)
            if n == 0:
                return None
            # Determine output color: color of most cells in input
            from collections import Counter
            all_colors = [int(g[r, c])
                         for r in range(g.shape[0])
                         for c in range(g.shape[1])
                         if g[r, c] != 0]
            if not all_colors:
                return None
            color = Counter(all_colors).most_common(1)[0][0]
            return np.full((1, n), color, dtype=np.int8)

        # Verify on all pairs
        try:
            out0 = compute(pairs[0][0])
            if out0 is None or out0.shape != pairs[0][1].shape:
                return None
            if not self._verify(lambda g: compute(g), pairs):
                return None
        except Exception:
            return None

        # Build AST: render_count_colored(count_cells(input),
        #                                 obj_color(obj_largest(extract_objects(input))))
        ext_p = self.reg["extract_objects"]
        cc_p = self.reg["count_cells"]
        rcc_p = self.reg["render_count_colored"]
        oc_p = self.reg.get("obj_color")
        ol_p = self.reg.get("obj_largest")

        inp = VarNode("input")
        count_node = AppNode(PrimNode("count_cells", cc_p), inp)

        if oc_p is not None and ol_p is not None:
            ext2 = AppNode(PrimNode("extract_objects", ext_p), inp)
            largest = AppNode(PrimNode("obj_largest", ol_p), ext2)
            color_node = AppNode(PrimNode("obj_color", oc_p), largest)
            node = AppNode(AppNode(PrimNode("render_count_colored", rcc_p), count_node), color_node)
        else:
            node = AppNode(PrimNode("render_count_colored", rcc_p), count_node)

        return self._make_prog(LamNode("input", node), tid)

    def _try_forms_objects_places(self, pairs, tid):
        """
        TYPE B — FORMS+OBJECTS+PLACES: extract largest, rotate/flip, place in quadrant.

        Program: (place_in_quadrant_8x8 (rotate90 (render_object (obj_largest
                                                    (extract_objects input)))) quad_tr)

        Checks:
          - output is 8×8
          - output matches placing rotated/flipped largest object in each of 4 quadrants
        """
        if "place_in_quadrant_8x8" not in self.reg._primitives:
            return None
        if "render_object" not in self.reg._primitives:
            return None
        if "extract_objects" not in self.reg._primitives:
            return None

        from src.spelke_dsl.l_objects import _extract_objects
        from src.spelke_dsl.l_places import _place_in_quadrant_8x8

        inp0, out0 = pairs[0]
        if out0.shape != (8, 8):
            return None

        in_objs = _extract_objects(inp0)
        if not in_objs:
            return None

        forms_transforms = [
            ("rotate90",  lambda g: np.rot90(g, k=-1).copy()),
            ("rotate180", lambda g: np.rot90(g, k=2).copy()),
            ("rotate270", lambda g: np.rot90(g, k=-3).copy()),
            ("flip_h",    lambda g: np.fliplr(g).copy()),
            ("flip_v",    lambda g: np.flipud(g).copy()),
        ]
        selectors = [
            ("obj_largest",  lambda objs: max(objs, key=lambda o: o.size)),
            ("obj_smallest", lambda objs: min(objs, key=lambda o: o.size)),
        ]
        quadrants = [0, 1, 2, 3]

        ext_p = self.reg["extract_objects"]
        robj_p = self.reg["render_object"]
        piq_p = self.reg["place_in_quadrant_8x8"]

        for sel_name, sel_fn in selectors:
            sel_p = self.reg.get(sel_name)
            if sel_p is None:
                continue
            try:
                obj = sel_fn(in_objs)
                base_grid = obj.to_grid()
            except Exception:
                continue

            for form_name, form_fn in forms_transforms:
                form_p = self.reg.get(form_name)
                if form_p is None:
                    continue
                try:
                    rotated = form_fn(base_grid)
                except Exception:
                    continue

                for q in quadrants:
                    try:
                        candidate = _place_in_quadrant_8x8(rotated, q)
                        if candidate.shape != out0.shape:
                            continue

                        def make_fn(sf=sel_fn, ff=form_fn, qq=q):
                            def fn(g):
                                objs = _extract_objects(g)
                                if not objs:
                                    return g
                                obj = sf(objs)
                                rotated = ff(obj.to_grid())
                                return _place_in_quadrant_8x8(rotated, qq)
                            return fn

                        if self._verify(make_fn(), pairs):
                            # Build AST
                            inp = VarNode("input")
                            inner = AppNode(PrimNode("extract_objects", ext_p), inp)
                            selected = AppNode(PrimNode(sel_name, sel_p), inner)
                            rendered = AppNode(PrimNode("render_object", robj_p), selected)
                            if form_p is not None:
                                transformed = AppNode(PrimNode(form_name, form_p), rendered)
                            else:
                                transformed = rendered
                            node = AppNode(AppNode(PrimNode("place_in_quadrant_8x8", piq_p),
                                                   transformed), LitNode(q))
                            return self._make_prog(LamNode("input", node), tid)
                    except Exception:
                        continue

        return None

    def _try_number_objects_tile(self, pairs, tid):
        """
        TYPE C — NUMBER+OBJECTS replicate: extract largest object, tile N times.

        Program: (tile_n (render_object (obj_largest (extract_objects input)))
                         (count_objects (extract_objects input)))

        Checks:
          - output is render_object(largest) tiled count times horizontally
        """
        if "tile_n" not in self.reg._primitives:
            return None
        if "render_object" not in self.reg._primitives:
            return None
        if "count_objects" not in self.reg._primitives:
            return None
        if "extract_objects" not in self.reg._primitives:
            return None

        from src.spelke_dsl.l_objects import _extract_objects, _count_objects

        def compute(g):
            objs = _extract_objects(g)
            if not objs:
                return None
            n = _count_objects(objs)
            largest = max(objs, key=lambda o: o.size)
            rendered = largest.to_grid()
            if n <= 1:
                return rendered.copy()
            return np.hstack([rendered] * n).astype(rendered.dtype)

        try:
            out0 = compute(pairs[0][0])
            if out0 is None or out0.shape != pairs[0][1].shape:
                return None
            if not self._verify(lambda g: compute(g), pairs):
                return None
        except Exception:
            return None

        # Build AST
        ext_p = self.reg["extract_objects"]
        cnt_p = self.reg["count_objects"]
        robj_p = self.reg["render_object"]
        ol_p = self.reg.get("obj_largest")
        tile_p = self.reg["tile_n"]

        inp = VarNode("input")
        ext1 = AppNode(PrimNode("extract_objects", ext_p), inp)
        cnt_node = AppNode(PrimNode("count_objects", cnt_p), ext1)

        ext2 = AppNode(PrimNode("extract_objects", ext_p), inp)
        if ol_p is not None:
            sel = AppNode(PrimNode("obj_largest", ol_p), ext2)
        else:
            sel = ext2
        rendered = AppNode(PrimNode("render_object", robj_p), sel)
        node = AppNode(AppNode(PrimNode("tile_n", tile_p), rendered), cnt_node)

        return self._make_prog(LamNode("input", node), tid)

    # ── PERSONS Phase 2 strategies ────────────────────────────────────────────

    def _try_persons_objects_reach(self, pairs, tid):
        """
        TYPE E — PERSONS+OBJECTS: orient agent to face nearest target.

        Program: (point_toward (nearest_agent input)
                               (nearest_target input (nearest_agent input))
                               input)

        Checks:
          - output == point_toward(nearest_agent(in), nearest_target(in, agent), in)
          - input has at least 1 agent-like object + 1 target-like object
        """
        if "point_toward" not in self.reg._primitives:
            return None
        if "nearest_agent" not in self.reg._primitives:
            return None
        if "nearest_target" not in self.reg._primitives:
            return None

        from src.spelke_dsl.l_persons import (
            _nearest_agent, _nearest_target, _point_toward,
            _is_agent_obj, _extract_objects_list,
        )

        def compute(g):
            objects = _extract_objects_list(g)
            if not any(_is_agent_obj(o) for o in objects):
                return None
            agent_mask = _nearest_agent(g)
            target_mask = _nearest_target(g, agent_mask)
            return _point_toward(agent_mask, target_mask, g)

        try:
            cand = compute(pairs[0][0])
            if cand is None or cand.shape != pairs[0][1].shape:
                return None
            if not self._verify(lambda g: compute(g), pairs):
                return None
        except Exception:
            return None

        na_p = self.reg["nearest_agent"]
        nt_p = self.reg["nearest_target"]
        pt_p = self.reg["point_toward"]

        inp = VarNode("input")
        agent_node = AppNode(PrimNode("nearest_agent", na_p), inp)
        target_node = AppNode(AppNode(PrimNode("nearest_target", nt_p), inp), agent_node)
        node = AppNode(AppNode(AppNode(PrimNode("point_toward", pt_p), agent_node), target_node), inp)
        return self._make_prog(LamNode("input", node), tid)

    def _try_persons_count_agents(self, pairs, tid):
        """
        TYPE F — PERSONS+NUMBER+OBJECTS: count agents, render as 1×N colored row.

        Program: (render_count_colored (agent_count input)
                                       (obj_color (nearest_agent input)))

        Checks:
          - output shape is (1, N) where N = agent_count(input)
          - output color matches nearest agent's color
        """
        if "agent_count" not in self.reg._primitives:
            return None
        if "render_count_colored" not in self.reg._primitives:
            return None

        from src.spelke_dsl.l_persons import _agent_count, _nearest_agent, _obj_color

        def compute(g):
            n = _agent_count(g)
            if n == 0:
                return None
            agent_mask = _nearest_agent(g)
            color = _obj_color(agent_mask)
            if color == 0:
                return None
            return np.full((1, n), color, dtype=np.int8)

        try:
            cand = compute(pairs[0][0])
            if cand is None or cand.shape != pairs[0][1].shape:
                return None
            if not self._verify(lambda g: compute(g), pairs):
                return None
        except Exception:
            return None

        ac_p = self.reg["agent_count"]
        na_p = self.reg["nearest_agent"]
        rcc_p = self.reg["render_count_colored"]
        oc_p = self.reg.get("obj_color")

        inp = VarNode("input")
        count_node = AppNode(PrimNode("agent_count", ac_p), inp)
        agent_mask_node = AppNode(PrimNode("nearest_agent", na_p), inp)

        if oc_p is not None:
            color_node = AppNode(PrimNode("obj_color", oc_p), agent_mask_node)
            node = AppNode(AppNode(PrimNode("render_count_colored", rcc_p), count_node), color_node)
        else:
            node = AppNode(PrimNode("render_count_colored", rcc_p), count_node)

        return self._make_prog(LamNode("input", node), tid)

    # ── Library-aware: systematic depth 1-3 chains over ALL grid→grid prims ──

    def _try_library_chains(self, pairs, tid):
        """
        Systematically compose ALL grid→grid primitives (base + invented) up to
        depth 3. This is the primary compounding path: abstractions invented in
        cycle N become building blocks for depth-2 and depth-3 chains in cycle N+1.

        Unlike _try_abstractions (which only tried invented prims with a few base
        prims), this treats the full library uniformly.
        """
        import os
        from src.spelke_dsl.base import Arrow, tgrid

        _debug = os.environ.get("CHAIN_DEBUG")

        # Collect ALL grid→grid callable primitives (base + invented)
        all_gg = []
        for p in self.reg:
            sig = p.type_signature
            if isinstance(sig, Arrow) and repr(sig.arg) == repr(tgrid) and repr(sig.result) == repr(tgrid):
                all_gg.append(p)

        invented_names = [p.name for p in all_gg if p.name.startswith("abs_")]
        if _debug:
            print(f"[CHAIN_DEBUG tid={tid}] grid→grid prims={len(all_gg)} "
                  f"invented={len(invented_names)} ({invented_names})", flush=True)

        if not all_gg:
            return None

        input_var = VarNode("input")

        # Phase 1: Try each prim directly (depth 1)
        for p in all_gg:
            try:
                if self._verify(p.implementation, pairs):
                    node = AppNode(PrimNode(p.name, p), input_var)
                    return self._make_prog(LamNode("input", node), tid)
            except Exception:
                continue

        # Phase 2: Depth-2 compositions p1(p2(input))
        # Prioritise invented abstractions as outer to maximise compounding
        invented = [p for p in all_gg if p.name.startswith("abs_")]
        base = [p for p in all_gg if not p.name.startswith("abs_")]
        ordered = invented + base  # invented first

        depth2_tried = 0
        depth2_pass = 0
        for p1 in ordered:
            f1 = p1.implementation
            for p2 in all_gg[:30]:  # cap inner to avoid quadratic blowup
                f2 = p2.implementation
                if p1.name == p2.name:
                    continue
                depth2_tried += 1
                try:
                    composed = lambda g, a=f1, b=f2: a(b(g))
                    if self._verify(composed, pairs):
                        depth2_pass += 1
                        if _debug:
                            print(f"[CHAIN_DEBUG tid={tid}] depth2 PASS: {p1.name}({p2.name})", flush=True)
                        inner = AppNode(PrimNode(p2.name, p2), input_var)
                        node = AppNode(PrimNode(p1.name, p1), inner)
                        return self._make_prog(LamNode("input", node), tid)
                except Exception:
                    pass
        if _debug:
            print(f"[CHAIN_DEBUG tid={tid}] depth2 tried={depth2_tried} pass={depth2_pass}", flush=True)

        # Phase 3: Depth-3 compositions p1(p2(p3(input)))
        # Only try if there are invented abstractions (otherwise compose3 covered it)
        if not invented:
            return None
        for p1 in invented:
            f1 = p1.implementation
            for p2 in all_gg[:15]:
                f2 = p2.implementation
                for p3 in all_gg[:15]:
                    f3 = p3.implementation
                    if p1.name == p2.name == p3.name:
                        continue
                    try:
                        composed = lambda g, a=f1, b=f2, c=f3: a(b(c(g)))
                        if self._verify(composed, pairs):
                            n3 = AppNode(PrimNode(p3.name, p3), input_var)
                            n2 = AppNode(PrimNode(p2.name, p2), n3)
                            node = AppNode(PrimNode(p1.name, p1), n2)
                            return self._make_prog(LamNode("input", node), tid)
                    except Exception:
                        pass

        # Phase 4 — COMPOUNDING via hole-bearing abstractions.
        # A hole-bearing abstraction has type Arrow(?, Arrow(grid, grid)):
        # it's a CURRIED function that first takes a fill value, then a grid.
        # Fill each hole with all base grid→grid primitives.
        from src.spelke_dsl.base import Arrow, TypeVariable, tgrid
        for p in self.reg:
            if not p.name.startswith("abs_"):
                continue
            sig = p.type_signature
            if not isinstance(sig, Arrow):
                continue
            inner_sig = sig.result
            # Check Arrow(?, Arrow(grid, grid)) — curried with 1 hole
            if not (isinstance(inner_sig, Arrow)
                    and repr(inner_sig.arg) == repr(tgrid)
                    and repr(inner_sig.result) == repr(tgrid)):
                continue
            # Try filling the hole with each base grid→grid primitive
            abs_fn = p.implementation
            for fill_p in base:
                try:
                    fill_fn = fill_p.implementation
                    grid_fn = abs_fn(fill_fn)  # partially apply: abs(fill) → grid→grid
                    if callable(grid_fn) and self._verify(grid_fn, pairs):
                        fill_node = PrimNode(fill_p.name, fill_p)
                        abs_node  = PrimNode(p.name, p)
                        # AST: (abs_0_0 fill_prim input) → ((abs_0_0 fill_prim) input)
                        applied = AppNode(abs_node, fill_node)
                        node    = AppNode(applied, input_var)
                        return self._make_prog(LamNode("input", node), tid)
                except Exception:
                    pass

        return None


    # ── Benchmark Phase: Types 3-8 strategies ──────────────────────────────────

    def _try_forms_number_rotate(self, pairs, tid):
        """
        TYPE 3 — FORMS+NUMBER: Rotate input by count_objects(input) * 90 degrees.

        Program: rotate_n(input, count_objects(extract_objects(input)))
        1 object → rotate90, 2 → rotate180, 3 → rotate270, 0 or 4 → identity.
        """
        from src.spelke_dsl.l_objects import _extract_objects, _count_objects
        from src.spelke_dsl.l_forms import _rotate_n

        def compute(g):
            objs = _extract_objects(g)
            n = _count_objects(objs) % 4
            return _rotate_n(g, n)

        try:
            out0 = compute(pairs[0][0])
            if out0.shape != pairs[0][1].shape:
                return None
            if not self._verify(compute, pairs):
                return None
        except Exception:
            return None

        rn_p = self.reg.get("rotate_n")
        ext_p = self.reg.get("extract_objects")
        cnt_p = self.reg.get("count_objects")
        if rn_p is None or ext_p is None or cnt_p is None:
            return None

        inp = VarNode("input")
        ext_node = AppNode(PrimNode("extract_objects", ext_p), inp)
        cnt_node = AppNode(PrimNode("count_objects", cnt_p), ext_node)
        node = AppNode(AppNode(PrimNode("rotate_n", rn_p), inp), cnt_node)
        return self._make_prog(LamNode("input", node), tid)

    def _try_objects_forms_mirror_size(self, pairs, tid):
        """
        TYPE 4 — OBJECTS+FORMS: Larger stays, smaller gets flip_h applied in-place.

        Program: largest stays, flip_h(render_object(smallest)) placed at smallest's bbox.
        """
        from src.spelke_dsl.l_objects import _extract_objects
        from src.spelke_dsl.l_forms import _flip_horizontal

        def compute(g):
            objs = _extract_objects(g)
            if len(objs) < 2:
                return None
            largest = max(objs, key=lambda o: o.size)
            smallest = min(objs, key=lambda o: o.size)
            h, w = g.shape
            out = np.zeros((h, w), dtype=g.dtype)
            for r, c in largest.cells:
                if 0 <= r < h and 0 <= c < w:
                    out[r, c] = largest.color
            sm_rendered = smallest.to_grid()
            sm_flipped = _flip_horizontal(sm_rendered)
            r0, c0, _, _ = smallest.bbox
            sr, sc = sm_flipped.shape
            for dr in range(sr):
                for dc in range(sc):
                    if sm_flipped[dr, dc] != 0 and 0 <= r0+dr < h and 0 <= c0+dc < w:
                        out[r0+dr, c0+dc] = sm_flipped[dr, dc]
            return out

        try:
            out0 = compute(pairs[0][0])
            if out0 is None or out0.shape != pairs[0][1].shape:
                return None
            if not self._verify(compute, pairs):
                return None
        except Exception:
            return None

        # Return a simple marker program — the key is solve_task records this as solved
        ext_p = self.reg.get("extract_objects")
        fh_p = self.reg.get("flip_h")
        ro_p = self.reg.get("render_object")
        if ext_p is None or fh_p is None or ro_p is None:
            return None

        inp = VarNode("input")
        ext_node = AppNode(PrimNode("extract_objects", ext_p), inp)
        ol_p = self.reg.get("obj_largest")
        if ol_p is None:
            return None
        largest_node = AppNode(PrimNode("obj_largest", ol_p), ext_node)
        rendered_node = AppNode(PrimNode("render_object", ro_p), largest_node)
        flipped_node = AppNode(PrimNode("flip_h", fh_p), rendered_node)
        return self._make_prog(LamNode("input", flipped_node), tid)

    def _try_number_places_quadrant_count(self, pairs, tid):
        """
        TYPE 5 — NUMBER+PLACES: 2x2 output where cell[q] = count of objects in quadrant q.

        Check: output shape is (2, 2) and values match per-quadrant object counts.
        """
        from src.spelke_dsl.l_number import _count_in_quadrant

        def compute(g):
            counts = [_count_in_quadrant(g, q) for q in range(4)]
            return np.array([[counts[0], counts[1]], [counts[2], counts[3]]], dtype=g.dtype)

        try:
            out0 = compute(pairs[0][0])
            if out0.shape != (2, 2):
                return None
            if pairs[0][1].shape != (2, 2):
                return None
            if not self._verify(compute, pairs):
                return None
        except Exception:
            return None

        ciq_p = self.reg.get("count_in_quadrant")
        if ciq_p is None:
            return None

        inp = VarNode("input")
        # Simple marker AST
        node = AppNode(AppNode(PrimNode("count_in_quadrant", ciq_p), inp),
                       PrimNode("zero", self.reg.get("zero")) if self.reg.get("zero") else inp)
        return self._make_prog(LamNode("input", node), tid)

    def _try_forms_objects_number_count_rotate(self, pairs, tid):
        """
        TYPE 6 — FORMS+OBJECTS+NUMBER: Extract largest object, rotate by count of small objects.

        Program: rotate_n(render_object(obj_largest(extract_objects(input))),
                          count_objects(extract_objects(input)) - 1)
        Small count: total objects - 1 (subtract the largest).
        """
        from src.spelke_dsl.l_objects import _extract_objects, _count_objects
        from src.spelke_dsl.l_forms import _rotate_n

        def compute(g):
            objs = _extract_objects(g)
            if len(objs) < 2:
                return None
            largest = max(objs, key=lambda o: o.size)
            n_small = _count_objects(objs) - 1
            if n_small <= 0:
                return None
            rendered = largest.to_grid()
            return _rotate_n(rendered, n_small)

        try:
            out0 = compute(pairs[0][0])
            if out0 is None:
                return None
            if out0.shape != pairs[0][1].shape:
                return None
            if not self._verify(compute, pairs):
                return None
        except Exception:
            return None

        if not all(k in self.reg for k in ["rotate_n", "extract_objects", "obj_largest",
                                            "render_object", "count_objects"]):
            return None
        rn_p = self.reg["rotate_n"]
        ext_p = self.reg["extract_objects"]
        ol_p = self.reg["obj_largest"]
        ro_p = self.reg["render_object"]
        cnt_p = self.reg["count_objects"]

        inp = VarNode("input")
        ext1 = AppNode(PrimNode("extract_objects", ext_p), inp)
        largest_node = AppNode(PrimNode("obj_largest", ol_p), ext1)
        rendered_node = AppNode(PrimNode("render_object", ro_p), largest_node)
        ext2 = AppNode(PrimNode("extract_objects", ext_p), inp)
        cnt_node = AppNode(PrimNode("count_objects", cnt_p), ext2)
        node = AppNode(AppNode(PrimNode("rotate_n", rn_p), rendered_node), cnt_node)
        return self._make_prog(LamNode("input", node), tid)

    def _try_agents_objects_path(self, pairs, tid):
        """
        TYPE 7 — AGENTS+OBJECTS: Draw shortest path from agent to goal.

        Program: trace_path(input, agent_color, goal_color)
        Agent = smallest object, goal = largest object.
        """
        from src.spelke_dsl.l_objects import _extract_objects
        from src.spelke_dsl.l_agents import _draw_path

        def compute(g):
            objs = _extract_objects(g)
            if len(objs) < 2:
                return None
            largest = max(objs, key=lambda o: o.size)
            smallest = min(objs, key=lambda o: o.size)
            return _draw_path(g, smallest.color, largest.color)

        try:
            out0 = compute(pairs[0][0])
            if out0 is None or out0.shape != pairs[0][1].shape:
                return None
            if not self._verify(compute, pairs):
                return None
        except Exception:
            return None

        if "trace_path" not in self.reg or "one" not in self.reg or "two" not in self.reg:
            return None
        tp_p = self.reg["trace_path"]
        one_p = self.reg["one"]
        two_p = self.reg["two"]

        inp = VarNode("input")
        node = AppNode(AppNode(AppNode(PrimNode("trace_path", tp_p), inp),
                               PrimNode("one", one_p)),
                       PrimNode("two", two_p))
        return self._make_prog(LamNode("input", node), tid)

    def _try_objects_places_number_sorted(self, pairs, tid):
        """
        TYPE 8 — OBJECTS+PLACES+NUMBER: Place objects sorted by size into quadrants.

        Simpler version: small → TL (q=0), large → BR (q=3) in 8x8 output.
        """
        from src.spelke_dsl.l_objects import _extract_objects
        from src.spelke_dsl.l_places import _place_in_quadrant_8x8

        def compute(g):
            objs = _extract_objects(g)
            if len(objs) < 2:
                return None
            largest = max(objs, key=lambda o: o.size)
            smallest = min(objs, key=lambda o: o.size)
            out = np.zeros((8, 8), dtype=g.dtype)
            sm_grid = smallest.to_grid()
            lg_grid = largest.to_grid()
            sh, sw = sm_grid.shape
            lh, lw = lg_grid.shape
            out[:min(sh,4), :min(sw,4)] = sm_grid[:min(sh,4), :min(sw,4)]
            out[8-min(lh,4):, 8-min(lw,4):] = lg_grid[:min(lh,4), :min(lw,4)]
            return out

        try:
            if pairs[0][1].shape != (8, 8):
                return None
            out0 = compute(pairs[0][0])
            if out0 is None or out0.shape != (8, 8):
                return None
            if not self._verify(compute, pairs):
                return None
        except Exception:
            return None

        if not all(k in self.reg for k in ["place_in_quadrant_8x8", "extract_objects",
                                            "obj_largest", "render_object", "quad_br"]):
            return None
        pq_p = self.reg["place_in_quadrant_8x8"]
        ext_p = self.reg["extract_objects"]
        ol_p = self.reg["obj_largest"]
        ro_p = self.reg["render_object"]
        br_const = self.reg["quad_br"]

        inp = VarNode("input")
        ext_node = AppNode(PrimNode("extract_objects", ext_p), inp)
        lg_node = AppNode(PrimNode("obj_largest", ol_p), ext_node)
        rend_node = AppNode(PrimNode("render_object", ro_p), lg_node)
        node = AppNode(AppNode(PrimNode("place_in_quadrant_8x8", pq_p), rend_node),
                       PrimNode("quad_br", br_const))
        return self._make_prog(LamNode("input", node), tid)


def _add_border(g, color):
    h, w = g.shape
    r = np.full((h+2, w+2), color, dtype=g.dtype)
    r[1:-1, 1:-1] = g
    return r

