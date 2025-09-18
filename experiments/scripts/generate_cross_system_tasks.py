"""
generate_cross_system_tasks.py

Generates synthetic ARC-format tasks requiring cross-system programs:
  FORMS + OBJECTS: apply geometric transform to the largest extracted object.

Ground-truth programs (FORMS+OBJECTS, depth 4):
  (rotate90  (render_object (obj_largest (extract_objects input))))
  (flip_h    (render_object (obj_largest (extract_objects input))))
  (rotate180 (render_object (obj_largest (extract_objects input))))
  etc.

These tasks are ARC-format JSON files placed in a separate directory.
The system discovers the cross-system abstraction by compressing the corpus.
"""
from __future__ import annotations
import json, os, sys, random
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.spelke_dsl.l_objects import _extract_objects

SHAPES = {
    'L': [(0,0),(1,0),(2,0),(2,1),(2,2)],
    'T': [(0,0),(0,1),(0,2),(1,1)],
    'J': [(0,2),(1,2),(2,0),(2,1),(2,2)],
    'S': [(0,1),(0,2),(1,0),(1,1)],
    'Z': [(0,0),(0,1),(1,1),(1,2)],
    'C': [(0,0),(0,1),(0,2),(1,0),(2,0),(2,1),(2,2)],
    'U': [(0,0),(0,2),(1,0),(1,2),(2,0),(2,1),(2,2)],
    'Plus': [(0,1),(1,0),(1,1),(1,2),(2,1)],
}

TRANSFORMS = {
    'rotate90':  lambda g: np.rot90(g, k=-1).copy(),
    'rotate180': lambda g: np.rot90(g, k=2).copy(),
    'rotate270': lambda g: np.rot90(g, k=-3).copy(),
    'flip_h':    lambda g: np.fliplr(g).copy(),
    'flip_v':    lambda g: np.flipud(g).copy(),
    'transpose': lambda g: g.T.copy(),
}


def make_example(shape_name: str, obj_color: int, small_color: int,
                 transform_fn, rng: np.random.Generator) -> dict | None:
    """
    Create one (input, output) example for a given shape and transform.
    Returns None if extraction/transform fails.
    """
    shape_cells = SHAPES[shape_name]
    h, w = 8, 8

    # Random offset so shape lands away from edges
    max_r = max(r for r, c in shape_cells)
    max_c = max(c for r, c in shape_cells)
    off_r = int(rng.integers(1, h - max_r - 2))
    off_c = int(rng.integers(1, w - max_c - 2))

    grid = np.zeros((h, w), dtype=int)
    for r, c in shape_cells:
        grid[r + off_r, c + off_c] = obj_color

    # Small 1-cell marker (different color, different location)
    occupied = set((r + off_r, c + off_c) for r, c in shape_cells)
    attempts = 0
    while attempts < 50:
        sr = int(rng.integers(0, h))
        sc = int(rng.integers(0, w))
        if (sr, sc) not in occupied:
            grid[sr, sc] = small_color
            break
        attempts += 1
    else:
        return None

    # Verify extraction works
    objs = _extract_objects(grid)
    if len(objs) < 2:
        return None
    largest = max(objs, key=lambda o: o.size)
    if largest.color != obj_color:
        return None

    rendered = largest.to_grid()
    output = transform_fn(rendered)

    return {
        'input': grid.tolist(),
        'output': output.tolist(),
    }


def make_task(transform_name: str, n_train: int = 4, n_test: int = 1,
              seed: int = 0) -> dict | None:
    """
    Generate one synthetic ARC task for the given transform.
    Returns an ARC-format dict or None if not enough examples found.
    """
    rng = np.random.default_rng(seed)
    transform_fn = TRANSFORMS[transform_name]

    shape_names = list(SHAPES.keys())
    colors = list(range(1, 10))

    examples = []
    attempts = 0
    while len(examples) < n_train + n_test and attempts < 200:
        shape = rng.choice(shape_names)
        obj_color = int(rng.choice(colors))
        small_color = int(rng.choice([c for c in colors if c != obj_color]))
        ex = make_example(shape, obj_color, small_color, transform_fn, rng)
        if ex is not None:
            # Verify non-trivial (output != rendered, i.e. transform is not identity)
            inp_arr = np.array(ex['input'])
            out_arr = np.array(ex['output'])
            objs = _extract_objects(inp_arr)
            if objs:
                largest = max(objs, key=lambda o: o.size)
                rendered = largest.to_grid()
                if not np.array_equal(rendered, out_arr):
                    examples.append(ex)
        attempts += 1

    if len(examples) < n_train + 1:
        return None

    return {
        'train': examples[:n_train],
        'test': examples[n_train:n_train + n_test],
    }


def generate_all(out_dir: str, tasks_per_transform: int = 3, seed: int = 42):
    """
    Generate synthetic tasks for each FORMS transform, n per transform.
    Files named: synth_{transform}_{i}.json
    """
    os.makedirs(out_dir, exist_ok=True)

    # Transforms likely to produce non-trivial results on L/J/T shapes
    active_transforms = ['rotate90', 'rotate180', 'rotate270', 'flip_h', 'flip_v']
    generated = 0

    for tname in active_transforms:
        for i in range(tasks_per_transform):
            task = make_task(tname, n_train=4, n_test=1, seed=seed + generated * 7)
            if task is None:
                print(f'  WARNING: failed to generate {tname}_{i}')
                continue
            fname = f'synth_{tname}_{i}.json'
            with open(os.path.join(out_dir, fname), 'w') as f:
                json.dump(task, f)
            print(f'  Generated {fname}: {len(task["train"])} train examples')
            generated += 1

    print(f'\nTotal synthetic tasks generated: {generated}')
    return generated


def generate_compounding_tasks(out_dir: str, n_tasks: int = 5, seed: int = 100):
    """
    Generate tasks requiring abs_0_0(extract_tile, input).

    Pattern: extract the largest solid-color object, then apply extract_tile.
    For a solid-color MxN block, extract_tile returns [[C]] (1x1 tile).

    These tasks are UNSOLVABLE by _try_form_on_extracted_object (which never
    produces 1x1 output from larger input) but SOLVABLE in cycle 1 via Phase 4
    of _try_library_chains: abs_0_0(extract_tile, input).

    This demonstrates COMPOUNDING: cycle 1 uses abs_0_0 (discovered in cycle 0)
    to solve tasks that cycle 0 cannot solve.
    """
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    from src.spelke_dsl import build_spelke_library
    from src.spelke_dsl.l_objects import _extract_objects
    reg = build_spelke_library()
    extract_tile_fn = reg['extract_tile'].implementation

    generated = 0
    task_seed = seed
    while generated < n_tasks:
        task_seed += 1
        rng = np.random.default_rng(task_seed)
        examples = []

        for _ in range(8):  # try up to 8 examples per task
            if len(examples) >= 5:
                break
            grid_h, grid_w = 10, 10
            grid = np.zeros((grid_h, grid_w), dtype=int)

            # Solid-color block (the large object)
            obj_color = int(rng.integers(1, 9))
            block_h = int(rng.integers(2, 5))
            block_w = int(rng.integers(2, 5))
            off_r = int(rng.integers(1, grid_h - block_h - 1))
            off_c = int(rng.integers(1, grid_w - block_w - 1))
            grid[off_r:off_r+block_h, off_c:off_c+block_w] = obj_color

            # Small marker (different color, single cell)
            marker_color = int(rng.integers(1, 9))
            while marker_color == obj_color:
                marker_color = int(rng.integers(1, 9))
            occupied = set(
                (r, c) for r in range(off_r, off_r+block_h) for c in range(off_c, off_c+block_w)
            )
            for _ in range(50):
                mr, mc = int(rng.integers(0, grid_h)), int(rng.integers(0, grid_w))
                if (mr, mc) not in occupied:
                    grid[mr, mc] = marker_color
                    break
            else:
                continue

            # Compute output: extract_tile(render_object(obj_largest(extract_objects(input))))
            objs = _extract_objects(grid)
            if not objs:
                continue
            largest = max(objs, key=lambda o: o.size)
            if largest.color != obj_color:
                continue
            rendered = largest.to_grid()
            output = extract_tile_fn(rendered)
            if output.shape != (1, 1):
                continue  # Only keep tasks where output IS 1x1

            examples.append({'input': grid.tolist(), 'output': output.tolist()})

        if len(examples) >= 4:
            task = {'train': examples[:4], 'test': examples[4:5]}
            fname = f'synth_extract_tile_{generated}.json'
            with open(os.path.join(out_dir, fname), 'w') as f:
                json.dump(task, f)
            print(f'  Generated {fname}: {len(task["train"])} train examples')
            generated += 1

    print(f'\nTotal compounding tasks generated: {generated}')
    return generated


def generate_number_objects_tasks(out_dir: str, n_tasks: int = 30, seed: int = 200,
                                   canvas_sizes: list | None = None):
    """
    Generate TYPE A curriculum tasks: NUMBER + OBJECTS crossing.

    Pattern: N same-colored objects scattered in input → 1×N row of that color.
    Ground truth: (render_count_colored (count_objects (extract_objects input))
                                        (obj_color (obj_largest (extract_objects input))))

    Requirements:
      - N ranges from 1 to 5 (ARC-compatible, not trivial)
      - All objects are the same color (ensures unambiguous color extraction)
      - Each example in a task has a different N (so the task requires counting)
      - Objects are 1-cell scattered points for clear counting
      - Generic DSL (no extract_objects) solves 0 of these tasks
      - canvas_sizes: list of sizes to sample from (default: ARC distribution)

    Verified: AST solver _try_number_objects_count discovers these programs.
    """
    # ARC top grid sizes: 3×3, 8×8, 9×9, 10×10, 11×11, 15×15
    # Use sizes ≥5 so we can fit N=1..5 scattered objects
    if canvas_sizes is None:
        canvas_sizes = [8, 9, 10, 11, 15]

    os.makedirs(out_dir, exist_ok=True)

    from src.spelke_dsl.l_objects import _extract_objects

    def make_example_count(n_objects: int, color: int, rng: np.random.Generator,
                            h: int = 8, w: int = 8) -> dict | None:
        """Create one (input, output) pair with n_objects of given color."""
        grid = np.zeros((h, w), dtype=int)

        # Place n_objects distinct 1-cell objects at random positions
        positions = set()
        attempts = 0
        while len(positions) < n_objects and attempts < 200:
            r = int(rng.integers(0, h))
            c = int(rng.integers(0, w))
            if (r, c) not in positions:
                positions.add((r, c))
            attempts += 1

        if len(positions) < n_objects:
            return None

        for r, c in positions:
            grid[r, c] = color

        # Verify extraction gives exactly n_objects
        objs = _extract_objects(grid)
        if len(objs) != n_objects:
            return None

        # Output: 1×n_objects row filled with color
        output = np.full((1, n_objects), color, dtype=int)

        return {'input': grid.tolist(), 'output': output.tolist()}

    rng = np.random.default_rng(seed)
    colors = list(range(1, 10))
    generated = 0

    for task_idx in range(n_tasks):
        # Each task uses one fixed color, with examples varying the count N
        color = int(rng.choice(colors))
        n_values = sorted(rng.choice(range(1, 6), size=5, replace=False).tolist())
        # n_values: 5 different counts from 1-5

        # Sample canvas size from ARC distribution (one size per task, but varies across tasks)
        canvas_size = int(canvas_sizes[task_idx % len(canvas_sizes)])
        h, w = canvas_size, canvas_size

        examples = []
        for n in n_values:
            for attempt in range(10):
                ex = make_example_count(n, color, rng, h=h, w=w)
                if ex is not None:
                    examples.append(ex)
                    break

        if len(examples) < 4:
            print(f'  WARNING: task {task_idx} only got {len(examples)} examples, skipping')
            continue

        task = {'train': examples[:4], 'test': examples[4:5]}
        fname = f'synth_count_render_{task_idx:02d}.json'
        with open(os.path.join(out_dir, fname), 'w') as f:
            json.dump(task, f)
        print(f'  Generated {fname}: canvas={h}x{w}, color={color}, counts={n_values[:4]}')
        generated += 1

    print(f'\nTotal NUMBER+OBJECTS tasks generated: {generated}')
    return generated


def generate_forms_objects_places_tasks(out_dir: str, n_tasks: int = 30, seed: int = 300):
    """
    Generate TYPE B curriculum tasks: FORMS + OBJECTS + PLACES crossing.

    Pattern: Extract largest object → apply geometric transform → place in a
    specific quadrant of a blank 8×8 canvas.

    Ground truth: (place_in_quadrant_8x8 (rotate90 (render_object
                                           (obj_largest (extract_objects input)))) quad_tr)

    Requirements:
      - Input has 1 large object (≤4×4) + 1 small 1-cell marker of different color
      - Output is 8×8 with rotated large object in specified quadrant
      - The same transform and quadrant applies across all examples in a task
      - Generic DSL (no extract_objects) solves 0 of these tasks

    Verified: AST solver _try_forms_objects_places discovers these programs.
    """
    os.makedirs(out_dir, exist_ok=True)

    from src.spelke_dsl.l_objects import _extract_objects
    from src.spelke_dsl.l_places import _place_in_quadrant_8x8

    SMALL_SHAPES = {
        'L': [(0,0),(1,0),(2,0),(2,1)],
        'T': [(0,0),(0,1),(0,2),(1,1)],
        'S': [(0,1),(0,2),(1,0),(1,1)],
        'Z': [(0,0),(0,1),(1,1),(1,2)],
        'I3': [(0,0),(1,0),(2,0)],
        'I2': [(0,0),(1,0)],
        'Plus': [(0,1),(1,0),(1,1),(1,2),(2,1)],
    }

    TRANSFORMS = {
        'rotate90':  lambda g: np.rot90(g, k=-1).copy(),
        'rotate180': lambda g: np.rot90(g, k=2).copy(),
        'flip_h':    lambda g: np.fliplr(g).copy(),
        'flip_v':    lambda g: np.flipud(g).copy(),
    }

    rng = np.random.default_rng(seed)
    colors = list(range(1, 10))
    shape_names = list(SMALL_SHAPES.keys())
    transform_names = list(TRANSFORMS.keys())
    quadrants = [0, 1, 2, 3]  # TL, TR, BL, BR
    generated = 0

    def make_example_fop(shape_name: str, obj_color: int, marker_color: int,
                          transform_fn, quadrant: int,
                          rng: np.random.Generator) -> dict | None:
        """Create one (input, output) pair for FORMS+OBJECTS+PLACES task."""
        shape_cells = SMALL_SHAPES[shape_name]
        h, w = 8, 8

        max_r = max(r for r, c in shape_cells)
        max_c = max(c for r, c in shape_cells)
        # Place shape in left half of input (away from TR quadrant area)
        off_r = int(rng.integers(0, min(3, h - max_r - 1)))
        off_c = int(rng.integers(0, min(3, w - max_c - 1)))

        grid = np.zeros((h, w), dtype=int)
        placed = set()
        for r, c in shape_cells:
            grid[r + off_r, c + off_c] = obj_color
            placed.add((r + off_r, c + off_c))

        # Place 1-cell marker of different color, not overlapping
        for _ in range(50):
            mr = int(rng.integers(0, h))
            mc = int(rng.integers(0, w))
            if (mr, mc) not in placed:
                grid[mr, mc] = marker_color
                break
        else:
            return None

        # Verify extraction
        objs = _extract_objects(grid)
        if len(objs) < 2:
            return None
        largest = max(objs, key=lambda o: o.size)
        if largest.color != obj_color:
            return None
        if largest.size < 2:  # Must be the large object
            return None

        # Compute output
        rendered = largest.to_grid()
        transformed = transform_fn(rendered)

        # Verify fits in 4×4 quadrant
        if transformed.shape[0] > 4 or transformed.shape[1] > 4:
            return None

        output = _place_in_quadrant_8x8(transformed, quadrant)

        return {'input': grid.tolist(), 'output': output.tolist()}

    for task_idx in range(n_tasks):
        # Each task: same transform + same quadrant, varying shapes/colors
        transform_name = transform_names[task_idx % len(transform_names)]
        transform_fn = TRANSFORMS[transform_name]
        quadrant = quadrants[task_idx % len(quadrants)]

        examples = []
        attempts = 0
        while len(examples) < 5 and attempts < 100:
            shape = rng.choice(shape_names)
            obj_color = int(rng.choice(colors))
            marker_color = int(rng.choice([c for c in colors if c != obj_color]))
            ex = make_example_fop(shape, obj_color, marker_color, transform_fn, quadrant, rng)
            if ex is not None:
                examples.append(ex)
            attempts += 1

        if len(examples) < 4:
            print(f'  WARNING: task {task_idx} only got {len(examples)} examples, skipping')
            continue

        task = {'train': examples[:4], 'test': examples[4:5]}
        fname = f'synth_fop_{transform_name}_q{quadrant}_{task_idx:02d}.json'
        with open(os.path.join(out_dir, fname), 'w') as f:
            json.dump(task, f)
        print(f'  Generated {fname}: transform={transform_name}, quadrant={quadrant}')
        generated += 1

    print(f'\nTotal FORMS+OBJECTS+PLACES tasks generated: {generated}')
    return generated


def generate_number_objects_replicate_tasks(out_dir: str, n_tasks: int = 30, seed: int = 400,
                                             canvas_sizes: list | None = None):
    """
    Generate TYPE C curriculum tasks: NUMBER + OBJECTS replicate crossing.

    Pattern: Extract largest object, count all objects, tile the extracted
    object that many times horizontally.

    Ground truth: (tile_n (render_object (obj_largest (extract_objects input)))
                          (count_objects (extract_objects input)))

    Requirements:
      - Input has 1 large object (shape) + N-1 small 1-cell markers (different color)
      - Count N = total number of objects (large + small)
      - Output: extracted large object tiled N times horizontally
      - N ranges from 2 to 4 across examples in a task (so tiling varies)
      - Generic DSL (no extract_objects) solves 0 of these tasks

    Verified: AST solver _try_number_objects_tile discovers these programs.

    canvas_sizes: list of grid sizes to sample from (default: ARC distribution ≥8 for shape fits)
    """
    # Use ARC-matching sizes that are large enough for tiling shapes (min 8)
    if canvas_sizes is None:
        canvas_sizes = [8, 9, 10, 11, 15]

    os.makedirs(out_dir, exist_ok=True)

    from src.spelke_dsl.l_objects import _extract_objects

    TILE_SHAPES = {
        'Rect2x3': [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2)],
        'L3':      [(0,0),(1,0),(2,0),(2,1)],
        'T3':      [(0,0),(0,1),(0,2),(1,1)],
        'Plus':    [(0,1),(1,0),(1,1),(1,2),(2,1)],
        'Rect2x2': [(0,0),(0,1),(1,0),(1,1)],
        'I3':      [(0,0),(0,1),(0,2)],
    }

    rng = np.random.default_rng(seed)
    colors = list(range(1, 10))
    shape_names = list(TILE_SHAPES.keys())
    generated = 0

    def make_example_tile(shape_name: str, obj_color: int, marker_color: int,
                           n_markers: int, rng: np.random.Generator,
                           h: int = 8, w: int = 8) -> dict | None:
        """Create one (input, output) pair with large object + n_markers small dots."""
        shape_cells = TILE_SHAPES[shape_name]

        max_r = max(r for r, c in shape_cells)
        max_c = max(c for r, c in shape_cells)

        # Place large object in left portion of grid
        off_r = int(rng.integers(0, max(1, h - max_r - 2)))
        off_c = int(rng.integers(0, max(1, min(3, w - max_c - 2))))

        grid = np.zeros((h, w), dtype=int)
        occupied = set()
        for r, c in shape_cells:
            grid[r + off_r, c + off_c] = obj_color
            occupied.add((r + off_r, c + off_c))

        # Place n_markers 1-cell markers of marker_color (not touching large object)
        placed_markers = 0
        for _ in range(200):
            mr = int(rng.integers(0, h))
            mc = int(rng.integers(0, w))
            # Keep markers away from large object (at least 1 cell gap)
            too_close = any(abs(mr - pr) <= 1 and abs(mc - pc) <= 1
                           for pr, pc in occupied)
            if (mr, mc) not in occupied and not too_close:
                grid[mr, mc] = marker_color
                occupied.add((mr, mc))
                placed_markers += 1
                if placed_markers == n_markers:
                    break

        if placed_markers < n_markers:
            return None

        # Verify extraction: should have exactly 1 + n_markers objects
        objs = _extract_objects(grid)
        total_expected = 1 + n_markers
        if len(objs) != total_expected:
            return None

        # Largest should be the shape (not a marker)
        largest = max(objs, key=lambda o: o.size)
        if largest.color != obj_color or largest.size < 2:
            return None

        # Output: tile extracted object (total_expected) times horizontally
        rendered = largest.to_grid()
        n_tiles = total_expected  # count_objects total
        output = np.hstack([rendered] * n_tiles).astype(int)

        return {'input': grid.tolist(), 'output': output.tolist()}

    for task_idx in range(n_tasks):
        # Each task: same shape, colors vary; n_markers varies per example (1, 2, or 3)
        shape_name = shape_names[task_idx % len(shape_names)]
        obj_color = int(rng.choice(colors))
        marker_color = int(rng.choice([c for c in colors if c != obj_color]))

        # Sample canvas size from ARC distribution (one size per task)
        canvas_size = int(canvas_sizes[task_idx % len(canvas_sizes)])
        h, w = canvas_size, canvas_size

        # Examples: n_markers = 1 (total=2 tiles), 2 (total=3), 3 (total=4)
        examples = []
        for n_markers in [1, 2, 3, 1, 2]:  # try 5 counts for 5 examples
            for attempt in range(10):
                ex = make_example_tile(shape_name, obj_color, marker_color, n_markers, rng,
                                       h=h, w=w)
                if ex is not None:
                    examples.append(ex)
                    break

        if len(examples) < 4:
            print(f'  WARNING: task {task_idx} only got {len(examples)} examples, skipping')
            continue

        # Verify the examples have different output widths (proves counting is needed)
        output_widths = [len(ex['output'][0]) for ex in examples]
        if len(set(output_widths)) < 2:
            print(f'  WARNING: task {task_idx} has uniform widths {output_widths}, skipping')
            continue

        task = {'train': examples[:4], 'test': examples[4:5]}
        fname = f'synth_tile_n_{task_idx:02d}.json'
        with open(os.path.join(out_dir, fname), 'w') as f:
            json.dump(task, f)
        print(f'  Generated {fname}: canvas={h}x{w}, shape={shape_name}, widths={output_widths[:4]}')
        generated += 1

    print(f'\nTotal NUMBER+OBJECTS replicate tasks generated: {generated}')
    return generated


def generate_count_cells_tasks(out_dir: str, n_tasks: int = 30, seed: int = 500,
                               canvas_sizes: list | None = None):
    """
    Generate TYPE D curriculum tasks: NUMBER + OBJECTS crossing via count_cells.

    Pattern: N non-zero cells of one color (possibly adjacent/connected) →
             1×N row of that color.
    Ground truth: (render_count_colored (count_cells input)
                                        (obj_color (obj_largest (extract_objects input))))

    Key distinction from TYPE A (count_objects tasks):
    - TYPE A: objects are ALWAYS 1-cell and non-adjacent → count_objects = count_cells
    - TYPE D: objects are ADJACENT (forming multi-cell connected regions) →
              count_objects = 1 (one component) but count_cells = N (total cells)

    This teaches the system to use count_cells for "total non-zero cells" vs
    count_objects for "number of distinct objects". Stitch will discover
    abs = (lambda input. (render_count_colored (count_cells input) (obj_color ...)))

    Example: input has a 2×2 red square → output is [[r, r, r, r]] (4 cells)
    count_objects = 1 (one connected component), count_cells = 4 (four cells)

    Verified: Matches ARC task d631b094's exact pattern.

    canvas_sizes: default [3, 3, 3] — use 3×3 like d631b094
    """
    if canvas_sizes is None:
        canvas_sizes = [3, 3, 3, 4, 4]  # Mostly 3×3 like d631b094

    os.makedirs(out_dir, exist_ok=True)

    def make_example_cells(n_cells: int, color: int, rng: np.random.Generator,
                           h: int = 3, w: int = 3) -> dict | None:
        """Create one (input, output) pair with n_cells as a connected region."""
        if n_cells > h * w:
            return None

        # Build a connected region of exactly n_cells via BFS growth
        grid = np.zeros((h, w), dtype=int)
        start_r = int(rng.integers(0, h))
        start_c = int(rng.integers(0, w))
        grid[start_r, start_c] = color
        placed = [(start_r, start_c)]
        frontier = []

        # Add neighbors of start
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = start_r + dr, start_c + dc
            if 0 <= nr < h and 0 <= nc < w:
                frontier.append((nr, nc))

        rng.shuffle(frontier)

        while len(placed) < n_cells and frontier:
            r, c = frontier.pop(0)
            if grid[r, c] == 0:
                grid[r, c] = color
                placed.append((r, c))
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == 0:
                        frontier.append((nr, nc))

        if len(placed) < n_cells:
            return None

        # Verify cell count
        actual_cells = int(np.count_nonzero(grid))
        if actual_cells != n_cells:
            return None

        # Output: 1×n_cells row of color
        output = np.full((1, n_cells), color, dtype=int)
        return {'input': grid.tolist(), 'output': output.tolist()}

    rng = np.random.default_rng(seed)
    colors = list(range(1, 10))
    generated = 0

    for task_idx in range(n_tasks):
        color = int(rng.choice(colors))
        canvas_size = int(canvas_sizes[task_idx % len(canvas_sizes)])
        h, w = canvas_size, canvas_size
        max_n = h * w

        # Choose n_cells values: use 1..5 but cap at canvas area
        # Use at least some adjacent cells (n_cells ≥ 2) to distinguish from count_objects
        max_cells = min(5, max_n)
        if max_cells < 2:
            continue
        available = list(range(1, max_cells + 1))
        if len(available) < 4:
            continue
        n_values = sorted(rng.choice(available, size=min(5, len(available)), replace=False).tolist())

        examples = []
        for n in n_values:
            for attempt in range(20):
                ex = make_example_cells(n, color, rng, h=h, w=w)
                if ex is not None:
                    examples.append(ex)
                    break

        if len(examples) < 4:
            print(f'  WARNING: task {task_idx} only got {len(examples)} examples, skipping')
            continue

        task = {'train': examples[:4], 'test': examples[4:5]}
        fname = f'synth_count_cells_{task_idx:02d}.json'
        with open(os.path.join(out_dir, fname), 'w') as f:
            json.dump(task, f)
        print(f'  Generated {fname}: canvas={h}x{w}, color={color}, n_cells={n_values[:4]}')
        generated += 1

    print(f'\nTotal count_cells tasks generated: {generated}')
    return generated


def generate_persons_objects_tasks(out_dir: str, n_tasks: int = 30, seed: int = 600):
    """
    Generate 30 PERSONS+OBJECTS curriculum tasks in two sub-types.

    TYPE E (15 tasks) — "point_toward": orient agent to face target.
      Input:  wide agent (1×3, color A) + compact target (2×2, color B) in 10×10 grid
              where target is placed ABOVE/BELOW agent (vertical offset > horizontal)
              so agent needs to be rotated (is_tall mismatch in point_toward).
      Output: same grid with agent transposed (now 3×1 tall, facing target)
      Program: (point_toward (nearest_agent input)
                             (nearest_target input (nearest_agent input)) input)
      Why generic DSL fails: no generic primitive can rotate a specific sub-object
                             in place based on relative position of another object.

    TYPE F (15 tasks) — "count_agents": count asymmetric objects → render row.
      Input:  N asymmetric (1×3 wide, color A) + 1 compact (2×2, color B) in 10×10 grid
              N varies from 1 to 4 across examples within a task
      Output: 1×N row of color A (agent count only, ignores target)
      Program: (render_count_colored (agent_count input)
                                     (obj_color (nearest_agent input)))
      Why generic DSL fails: count_objects counts ALL objects (agents + target) = N+1,
                             but output width = N (agents only). Mismatch unless N=0.

    Verified: AST solver _try_persons_objects_reach and _try_persons_count_agents
              solve these programs. Generic DSL (no PERSONS) solves 0/30.
    """
    import sys as _sys
    _sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from src.spelke_dsl.l_persons import (
        _nearest_agent, _nearest_target, _point_toward,
        _is_agent_obj, _extract_objects_list, _obj_color, _agent_count,
    )

    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    colors = list(range(1, 10))
    generated_e = 0
    generated_f = 0

    # ── TYPE E: point_toward tasks ──────────────────────────────────────────
    task_idx = 0
    attempts_e = 0
    while generated_e < 15 and attempts_e < 500:
        attempts_e += 1
        agent_color = int(rng.choice(colors))
        target_color = int(rng.choice([c for c in colors if c != agent_color]))

        examples = []
        for _ in range(8):
            if len(examples) >= 5:
                break
            g = np.zeros((10, 10), dtype=int)

            # Place wide agent (1×3 horizontal) at a random row
            agent_row = int(rng.integers(1, 8))
            agent_col = int(rng.integers(1, 6))
            g[agent_row, agent_col] = agent_color
            g[agent_row, agent_col + 1] = agent_color
            g[agent_row, agent_col + 2] = agent_color

            # Place target (2×2 block) ABOVE or BELOW agent (vertical offset)
            # Vertical offset > horizontal offset → agent needs to rotate to face vertically
            direction = rng.choice(['above', 'below'])
            if direction == 'above':
                t_row = max(0, agent_row - int(rng.integers(3, 6)))
            else:
                t_row = min(7, agent_row + int(rng.integers(3, 6)))

            # Target centered horizontally close to agent center (small horizontal offset)
            agent_center_col = agent_col + 1
            t_col = max(0, min(7, agent_center_col + int(rng.integers(-1, 2))))

            # Ensure no overlap with agent
            agent_cells = {(agent_row, agent_col), (agent_row, agent_col+1),
                          (agent_row, agent_col+2)}
            target_cells = {(t_row, t_col), (t_row, t_col+1),
                           (t_row+1, t_col), (t_row+1, t_col+1)}

            if t_row + 1 > 9 or t_col + 1 > 9:
                continue
            if agent_cells & target_cells:
                continue
            if not target_cells.isdisjoint(agent_cells):
                continue

            g[t_row, t_col] = target_color
            g[t_row, t_col + 1] = target_color
            g[t_row + 1, t_col] = target_color
            g[t_row + 1, t_col + 1] = target_color

            # Compute output: point_toward(nearest_agent, nearest_target, grid)
            agent_mask = _nearest_agent(g)
            if not np.any(agent_mask == agent_color):
                continue
            target_mask = _nearest_target(g, agent_mask)
            if not np.any(target_mask == target_color):
                continue

            output = _point_toward(agent_mask, target_mask, g)

            # Ensure the output is DIFFERENT from input (rotation happened)
            if np.array_equal(output, g):
                continue

            # Sanity: output has the agent color somewhere
            if not np.any(output == agent_color):
                continue

            examples.append({'input': g.tolist(), 'output': output.tolist()})

        if len(examples) < 4:
            continue

        task = {
            'train': examples[:4],
            'test': examples[4:5],
            'task_id': f'synth_persons_e_{generated_e:02d}',
            'systems': ['PERSONS', 'OBJECTS'],
            'type': 'E_point_toward',
        }
        fname = f'synth_persons_e_{generated_e:02d}.json'
        with open(os.path.join(out_dir, fname), 'w') as f:
            json.dump(task, f)
        print(f'  Generated {fname}: agent_color={agent_color}, target_color={target_color}')
        generated_e += 1
        task_idx += 1

    # ── TYPE F: count_agents tasks ───────────────────────────────────────────
    attempts_f = 0
    while generated_f < 15 and attempts_f < 500:
        attempts_f += 1
        agent_color = int(rng.choice(colors))
        target_color = int(rng.choice([c for c in colors if c != agent_color]))

        examples = []
        for n_agents in [1, 2, 3, 4, 2]:  # 5 examples with varying agent counts
            placed = False
            for _ in range(20):
                g = np.zeros((10, 10), dtype=int)

                # Place n_agents wide (1×3) agents
                agent_positions = []
                ok = True
                occupied = set()
                for ai in range(n_agents):
                    placed_agent = False
                    for _ in range(50):
                        ar = int(rng.integers(1, 8))
                        ac = int(rng.integers(0, 7))
                        cells = {(ar, ac), (ar, ac+1), (ar, ac+2)}
                        # Need 1 cell buffer to avoid merging
                        buffer = set()
                        for r, c in cells:
                            for dr in [-1, 0, 1]:
                                for dc in [-1, 0, 1]:
                                    buffer.add((r+dr, c+dc))
                        if not cells.isdisjoint(occupied) or not buffer.isdisjoint(occupied):
                            continue
                        for r, c in cells:
                            g[r, c] = agent_color
                        occupied |= cells
                        agent_positions.append((ar, ac))
                        placed_agent = True
                        break
                    if not placed_agent:
                        ok = False
                        break

                if not ok:
                    continue

                # Place 1 compact target (2×2 block, different color)
                placed_target = False
                for _ in range(50):
                    tr = int(rng.integers(1, 8))
                    tc = int(rng.integers(1, 8))
                    target_cells = {(tr, tc), (tr, tc+1), (tr+1, tc), (tr+1, tc+1)}
                    if tr + 1 > 9 or tc + 1 > 9:
                        continue
                    # Buffer: avoid merging with agents
                    buffer = set()
                    for r, c in target_cells:
                        for dr in [-1, 0, 1]:
                            for dc in [-1, 0, 1]:
                                buffer.add((r+dr, c+dc))
                    if not target_cells.isdisjoint(occupied) or not buffer.isdisjoint(occupied):
                        continue
                    g[tr, tc] = target_color
                    g[tr, tc + 1] = target_color
                    g[tr + 1, tc] = target_color
                    g[tr + 1, tc + 1] = target_color
                    placed_target = True
                    break

                if not placed_target:
                    continue

                # Verify agent_count = n_agents
                actual_count = _agent_count(g)
                if actual_count != n_agents:
                    continue

                # Output: 1×n_agents row of agent_color
                output = np.full((1, n_agents), agent_color, dtype=int)

                examples.append({'input': g.tolist(), 'output': output.tolist()})
                placed = True
                break

            if not placed:
                pass  # Try with fewer examples

        if len(examples) < 4:
            continue

        # Verify output widths vary (proves agent counting is required)
        output_widths = [len(ex['output'][0]) for ex in examples[:4]]
        if len(set(output_widths)) < 2:
            continue

        task = {
            'train': examples[:4],
            'test': examples[4:5] if len(examples) >= 5 else examples[:1],
            'task_id': f'synth_persons_f_{generated_f:02d}',
            'systems': ['PERSONS', 'NUMBER', 'OBJECTS'],
            'type': 'F_count_agents',
        }
        fname = f'synth_persons_f_{generated_f:02d}.json'
        with open(os.path.join(out_dir, fname), 'w') as f:
            json.dump(task, f)
        print(f'  Generated {fname}: agent_color={agent_color}, counts={output_widths}')
        generated_f += 1

    total = generated_e + generated_f
    print(f'\nTotal PERSONS+OBJECTS tasks: {total} (TYPE E: {generated_e}, TYPE F: {generated_f})')
    return total


# ══════════════════════════════════════════════════════════════════════════
# BENCHMARK GENERATORS — Spelke Bootstrap Suite (1000 tasks, 8 crossing types)
# These generate into data/spelke_benchmark_staging/type{N}/
# T6 will assemble into data/spelke_benchmark/ with 80/20 train/test split.
# ══════════════════════════════════════════════════════════════════════════

def generate_benchmark_type1(out_dir: str, n_tasks: int = 95, seed: int = 1000,
                              canvas_sizes: list | None = None):
    """
    TYPE 1 — NUMBER+OBJECTS: Count and Render (benchmark variant).

    Program: render_count_colored(count_objects(extract_objects(input)),
                                  obj_color(obj_largest(extract_objects(input))))
    Extends the curriculum version with:
    - N from 1-8 (was 1-5)
    - Grid sizes up to 15 (was 8-15)
    - 5 different count values per task (not just 1-5)

    AST solver: _try_number_objects_count (already handles this program)
    """
    if canvas_sizes is None:
        canvas_sizes = [8, 9, 10, 11, 12, 15]

    os.makedirs(out_dir, exist_ok=True)
    from src.spelke_dsl.l_objects import _extract_objects

    def make_example(n_objects: int, color: int, rng: np.random.Generator,
                     h: int, w: int) -> dict | None:
        grid = np.zeros((h, w), dtype=int)
        positions = set()
        attempts = 0
        while len(positions) < n_objects and attempts < 300:
            r, c = int(rng.integers(0, h)), int(rng.integers(0, w))
            if (r, c) not in positions:
                positions.add((r, c))
            attempts += 1
        if len(positions) < n_objects:
            return None
        for r, c in positions:
            grid[r, c] = color
        objs = _extract_objects(grid)
        if len(objs) != n_objects:
            return None
        output = np.full((1, n_objects), color, dtype=int)
        return {'input': grid.tolist(), 'output': output.tolist()}

    rng = np.random.default_rng(seed)
    colors = list(range(1, 10))
    generated = 0

    for task_idx in range(n_tasks * 2):  # oversample to ensure n_tasks
        if generated >= n_tasks:
            break
        color = int(rng.choice(colors))
        canvas_size = canvas_sizes[task_idx % len(canvas_sizes)]
        h, w = canvas_size, canvas_size
        max_n = min(8, h * w // 3)  # ensure space
        if max_n < 3:
            continue
        available = list(range(1, max_n + 1))
        n_choose = min(5, len(available))
        n_values = sorted(rng.choice(available, size=n_choose, replace=False).tolist())

        examples = []
        for n in n_values:
            for _ in range(15):
                ex = make_example(n, color, rng, h, w)
                if ex is not None:
                    examples.append(ex)
                    break

        if len(examples) < 4:
            continue
        # Verify varied counts
        output_widths = set(len(ex['output'][0]) for ex in examples[:4])
        if len(output_widths) < 2:
            continue

        task = {'train': examples[:4], 'test': examples[4:5] if len(examples) >= 5 else examples[:1]}
        fname = f'type1_{generated:03d}.json'
        with open(os.path.join(out_dir, fname), 'w') as f:
            json.dump(task, f)
        print(f'  type1_{generated:03d}: canvas={h}x{w}, color={color}, counts={n_values[:4]}')
        generated += 1

    print(f'\nType 1 benchmark tasks: {generated}')
    return generated


def generate_benchmark_type2(out_dir: str, n_tasks: int = 95, seed: int = 2000,
                              canvas_sizes: list | None = None):
    """
    TYPE 2 — NUMBER+OBJECTS: Extract and Tile (benchmark variant).

    Program: tile_n(render_object(obj_largest(extract_objects(input))),
                    count_objects(extract_objects(input)))
    Extends curriculum replicate version with more shapes and N up to 5.

    AST solver: _try_number_objects_tile (already handles this program)
    """
    if canvas_sizes is None:
        canvas_sizes = [8, 9, 10, 11, 12, 15]

    os.makedirs(out_dir, exist_ok=True)
    from src.spelke_dsl.l_objects import _extract_objects

    BENCH_SHAPES = {
        'Rect2x3': [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2)],
        'L3':      [(0,0),(1,0),(2,0),(2,1)],
        'T3':      [(0,0),(0,1),(0,2),(1,1)],
        'Plus':    [(0,1),(1,0),(1,1),(1,2),(2,1)],
        'Rect2x2': [(0,0),(0,1),(1,0),(1,1)],
        'I3':      [(0,0),(0,1),(0,2)],
        'Z':       [(0,0),(0,1),(1,1),(1,2)],
        'S':       [(0,1),(0,2),(1,0),(1,1)],
    }

    def make_example(shape_name: str, obj_color: int, marker_color: int,
                     n_markers: int, rng: np.random.Generator,
                     h: int, w: int) -> dict | None:
        shape_cells = BENCH_SHAPES[shape_name]
        max_r = max(r for r, c in shape_cells)
        max_c = max(c for r, c in shape_cells)
        if h - max_r < 3 or w - max_c < 3:
            return None
        off_r = int(rng.integers(0, max(1, h - max_r - 2)))
        off_c = int(rng.integers(0, max(1, min(3, w - max_c - 2))))
        grid = np.zeros((h, w), dtype=int)
        occupied = set()
        for r, c in shape_cells:
            grid[r + off_r, c + off_c] = obj_color
            occupied.add((r + off_r, c + off_c))
        placed = 0
        for _ in range(300):
            mr, mc = int(rng.integers(0, h)), int(rng.integers(0, w))
            too_close = any(abs(mr - pr) <= 1 and abs(mc - pc) <= 1
                           for pr, pc in occupied)
            if (mr, mc) not in occupied and not too_close:
                grid[mr, mc] = marker_color
                occupied.add((mr, mc))
                placed += 1
                if placed == n_markers:
                    break
        if placed < n_markers:
            return None
        objs = _extract_objects(grid)
        if len(objs) != 1 + n_markers:
            return None
        largest = max(objs, key=lambda o: o.size)
        if largest.color != obj_color or largest.size < 2:
            return None
        rendered = largest.to_grid()
        n_tiles = 1 + n_markers
        output = np.hstack([rendered] * n_tiles).astype(int)
        return {'input': grid.tolist(), 'output': output.tolist()}

    rng = np.random.default_rng(seed)
    colors = list(range(1, 10))
    shape_names = list(BENCH_SHAPES.keys())
    generated = 0

    for task_idx in range(n_tasks * 2):
        if generated >= n_tasks:
            break
        shape_name = shape_names[task_idx % len(shape_names)]
        obj_color = int(rng.choice(colors))
        marker_color = int(rng.choice([c for c in colors if c != obj_color]))
        canvas_size = canvas_sizes[task_idx % len(canvas_sizes)]
        h, w = canvas_size, canvas_size

        examples = []
        for n_markers in [1, 2, 3, 1, 2]:
            for _ in range(15):
                ex = make_example(shape_name, obj_color, marker_color, n_markers,
                                  rng, h, w)
                if ex is not None:
                    examples.append(ex)
                    break

        if len(examples) < 4:
            continue
        output_widths = set(len(ex['output'][0]) for ex in examples[:4])
        if len(output_widths) < 2:
            continue

        task = {'train': examples[:4], 'test': examples[4:5] if len(examples) >= 5 else examples[:1]}
        fname = f'type2_{generated:03d}.json'
        with open(os.path.join(out_dir, fname), 'w') as f:
            json.dump(task, f)
        print(f'  type2_{generated:03d}: canvas={h}x{w}, shape={shape_name}')
        generated += 1

    print(f'\nType 2 benchmark tasks: {generated}')
    return generated


def generate_benchmark_type3(out_dir: str, n_tasks: int = 100, seed: int = 3000):
    """
    TYPE 3 — FORMS+NUMBER: Shape Rotation by Count.

    Program: rotate_n(input, count_objects(extract_objects(input)))
    1 object → rotate90, 2 → rotate180, 3 → rotate270, 0 or 4 → identity
    The marker objects drive the rotation; the background grid is rotated.

    AST solver: _try_forms_number_rotate (added in T3)
    """
    os.makedirs(out_dir, exist_ok=True)
    from src.spelke_dsl.l_objects import _extract_objects
    from src.spelke_dsl.l_forms import _rotate_n

    PATTERNS = {
        'checker': lambda h, w, c: _make_checker(h, w, c),
        'stripes_h': lambda h, w, c: _make_stripes_h(h, w, c),
        'stripes_v': lambda h, w, c: _make_stripes_v(h, w, c),
        'cross': lambda h, w, c: _make_cross(h, w, c),
        'border': lambda h, w, c: _make_border_grid(h, w, c),
    }

    def _make_checker(h, w, color):
        g = np.zeros((h, w), dtype=int)
        for r in range(h):
            for c in range(w):
                if (r + c) % 2 == 0:
                    g[r, c] = color
        return g

    def _make_stripes_h(h, w, color):
        g = np.zeros((h, w), dtype=int)
        for r in range(0, h, 2):
            g[r, :] = color
        return g

    def _make_stripes_v(h, w, color):
        g = np.zeros((h, w), dtype=int)
        for c in range(0, w, 2):
            g[:, c] = color
        return g

    def _make_cross(h, w, color):
        g = np.zeros((h, w), dtype=int)
        g[h // 2, :] = color
        g[:, w // 2] = color
        return g

    def _make_border_grid(h, w, color):
        g = np.zeros((h, w), dtype=int)
        g[0, :] = color
        g[-1, :] = color
        g[:, 0] = color
        g[:, -1] = color
        return g

    rng = np.random.default_rng(seed)
    colors = list(range(1, 10))
    pattern_names = list(PATTERNS.keys())
    generated = 0
    sizes = [5, 6, 7, 8, 9]

    def make_one_example(n_markers, marker_color, h, rng):
        """Make one example with n_markers isolated objects of marker_color."""
        grid = np.zeros((h, h), dtype=int)
        positions = set()
        for _ in range(500):
            if len(positions) >= n_markers:
                break
            r, c = int(rng.integers(0, h)), int(rng.integers(0, h))
            nbrs = {(r+dr, c+dc) for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]}
            if (r, c) not in positions and not nbrs & positions:
                positions.add((r, c))
        if len(positions) < n_markers:
            return None
        for r, c in positions:
            grid[r, c] = marker_color
        objs = _extract_objects(grid)
        if len(objs) != n_markers:
            return None
        output = _rotate_n(grid, n_markers)
        if np.array_equal(output, grid):
            return None
        return {'input': grid.tolist(), 'output': output.tolist()}

    for task_idx in range(n_tasks * 3):
        if generated >= n_tasks:
            break

        size = sizes[task_idx % len(sizes)]
        marker_color = int(rng.choice(colors))

        # KEY: vary n_markers WITHIN each task (1, 2, 3, 1) so generic solver can't
        # pick a fixed rotation — the count must be computed each time
        examples = []
        for n_markers in [1, 2, 3, 1, 2]:  # alternating counts within task
            for _ in range(20):
                ex = make_one_example(n_markers, marker_color, size, rng)
                if ex is not None:
                    examples.append(ex)
                    break

        if len(examples) < 4:
            continue

        # Verify count varies across train examples
        counts = [sum(1 for v in ex['input'] for vv in v if vv != 0)
                  for ex in examples[:4]]
        if len(set(counts)) < 2:
            continue

        task = {'train': examples[:4], 'test': examples[4:5] if len(examples) >= 5 else examples[:1]}
        fname = f'type3_{generated:03d}.json'
        with open(os.path.join(out_dir, fname), 'w') as f:
            json.dump(task, f)
        print(f'  type3_{generated:03d}: size={size}x{size}, marker={marker_color}, counts_per_ex={counts[:4]}')
        generated += 1

    print(f'\nType 3 benchmark tasks: {generated}')
    return generated


def generate_benchmark_type4(out_dir: str, n_tasks: int = 100, seed: int = 4000):
    """
    TYPE 4 — OBJECTS+FORMS: Mirror by Size Order.

    Simpler version: 2 objects in input. Larger stays, smaller gets flip_h.
    Program: compose(render_object(obj_largest(...)),
                     flip_h(render_object(obj_smallest(...))))
    Output: overlay of larger object + flip_h(smaller object) in same grid.

    AST solver: _try_objects_forms_mirror_size (added in T4)
    """
    os.makedirs(out_dir, exist_ok=True)
    from src.spelke_dsl.l_objects import _extract_objects
    from src.spelke_dsl.l_forms import _flip_horizontal

    LARGE_SHAPES = {
        'L4':      [(0,0),(1,0),(2,0),(3,0),(3,1),(3,2)],
        'T4':      [(0,0),(0,1),(0,2),(0,3),(1,1),(2,1)],
        'S4':      [(0,1),(0,2),(1,0),(1,1),(2,0)],
        'C':       [(0,0),(0,1),(0,2),(1,0),(2,0),(2,1),(2,2)],
        'Z4':      [(0,0),(0,1),(1,1),(1,2),(2,2)],
    }
    # Must be asymmetric so flip_h changes them
    SMALL_SHAPES = {
        'L2':   [(0,0),(1,0),(1,1)],       # L: asymmetric
        'J2':   [(0,1),(1,0),(1,1)],       # J: mirror of L
        'S3':   [(0,1),(0,2),(1,0),(1,1)], # S-tetromino: asymmetric
        'F3':   [(0,0),(0,1),(1,1),(1,2)], # Z-tetromino: asymmetric
    }

    def make_example(large_shape: str, large_color: int, small_shape: str, small_color: int,
                     rng: np.random.Generator, h: int = 12, w: int = 12) -> dict | None:
        # Place large object
        lshape = LARGE_SHAPES[large_shape]
        lr_max = max(r for r, c in lshape)
        lc_max = max(c for r, c in lshape)
        if h - lr_max < 2 or w - lc_max < 2:
            return None
        lr_off = int(rng.integers(0, h - lr_max - 1))
        lc_off = int(rng.integers(0, (w - lc_max - 1) // 2))  # left half

        grid = np.zeros((h, w), dtype=int)
        large_cells = set()
        for r, c in lshape:
            grid[r + lr_off, c + lc_off] = large_color
            large_cells.add((r + lr_off, c + lc_off))

        # Place small object in right half, no overlap
        sshape = SMALL_SHAPES[small_shape]
        sr_max = max(r for r, c in sshape)
        sc_max = max(c for r, c in sshape)
        placed = False
        for _ in range(100):
            sr_off = int(rng.integers(0, max(1, h - sr_max - 1)))
            sc_off = int(rng.integers(w // 2, max(w // 2 + 1, w - sc_max - 1)))
            small_cells = {(r + sr_off, c + sc_off) for r, c in sshape}
            if not small_cells & large_cells and all(0 <= r < h and 0 <= c < w
                                                     for r, c in small_cells):
                for r, c in small_cells:
                    grid[r, c] = small_color
                placed = True
                break
        if not placed:
            return None

        # Verify extraction
        objs = _extract_objects(grid)
        if len(objs) != 2:
            return None
        largest = max(objs, key=lambda o: o.size)
        smallest = min(objs, key=lambda o: o.size)
        if largest.color != large_color or smallest.color != small_color:
            return None

        # Output: overlay large (unchanged) + flip_h(small)
        out = np.zeros((h, w), dtype=int)
        # Place large in same position
        for r, c in largest.cells:
            out[r, c] = largest.color
        # flip_h of small object rendered as minimal grid, then re-place at same top-left
        small_rendered = smallest.to_grid()
        small_flipped = _flip_horizontal(small_rendered)
        r0, c0, _, _ = smallest.bbox
        sr, sc = small_flipped.shape
        for dr in range(sr):
            for dc in range(sc):
                if small_flipped[dr, dc] != 0 and 0 <= r0+dr < h and 0 <= c0+dc < w:
                    out[r0 + dr, c0 + dc] = small_flipped[dr, dc]

        # Verify output differs from input
        if np.array_equal(out, grid):
            return None

        return {'input': grid.tolist(), 'output': out.tolist()}

    rng = np.random.default_rng(seed)
    colors = list(range(1, 10))
    large_names = list(LARGE_SHAPES.keys())
    small_names = list(SMALL_SHAPES.keys())
    generated = 0

    for task_idx in range(n_tasks * 3):
        if generated >= n_tasks:
            break
        large_shape = large_names[task_idx % len(large_names)]
        small_shape = small_names[task_idx % len(small_names)]
        large_color = int(rng.choice(colors))
        small_color = int(rng.choice([c for c in colors if c != large_color]))

        examples = []
        for _ in range(20):
            if len(examples) >= 5:
                break
            ex = make_example(large_shape, large_color, small_shape, small_color, rng)
            if ex is not None:
                examples.append(ex)

        if len(examples) < 4:
            continue

        task = {'train': examples[:4], 'test': examples[4:5] if len(examples) >= 5 else examples[:1]}
        fname = f'type4_{generated:03d}.json'
        with open(os.path.join(out_dir, fname), 'w') as f:
            json.dump(task, f)
        print(f'  type4_{generated:03d}: large={large_shape}/{large_color}, small={small_shape}/{small_color}')
        generated += 1

    print(f'\nType 4 benchmark tasks: {generated}')
    return generated


def generate_benchmark_type5(out_dir: str, n_tasks: int = 100, seed: int = 5000):
    """
    TYPE 5 — NUMBER+PLACES: Quadrant Count Map.

    Program: count objects per quadrant → output 2×2 grid where cell[q] = count in quadrant q.
    Each cell of the 2x2 output is the number of objects in that quadrant (value = count, color = count).

    AST solver: _try_number_places_quadrant_count (added in T4)
    """
    os.makedirs(out_dir, exist_ok=True)
    from src.spelke_dsl.l_objects import _extract_objects

    def place_n_objects_in_quadrant(grid: np.ndarray, n: int, q: int,
                                    color: int, rng: np.random.Generator) -> bool:
        """Place n isolated 1-cell objects of color in quadrant q. Returns True if successful."""
        h, w = grid.shape
        mid_r, mid_c = h // 2, w // 2
        quad_bounds = {
            0: (0, 0, mid_r, mid_c),
            1: (0, mid_c, mid_r, w),
            2: (mid_r, 0, h, mid_c),
            3: (mid_r, mid_c, h, w),
        }
        r0, c0, r1, c1 = quad_bounds[q]
        existing = {(r, c) for r in range(r0, r1) for c in range(c0, c1) if grid[r, c] != 0}
        placed = 0
        for _ in range(500):
            if placed >= n:
                break
            r = int(rng.integers(r0, r1))
            c = int(rng.integers(c0, c1))
            neighbors = {(r+dr, c+dc) for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]}
            if (r, c) not in existing and not neighbors & existing:
                grid[r, c] = color
                existing.add((r, c))
                placed += 1
        return placed == n

    def make_example(counts: list[int], color: int, rng: np.random.Generator,
                     h: int = 12, w: int = 12) -> dict | None:
        """counts[q] = number of objects to place in quadrant q (0-3)."""
        grid = np.zeros((h, w), dtype=int)
        for q, n in enumerate(counts):
            if n == 0:
                continue
            if not place_n_objects_in_quadrant(grid, n, q, color, rng):
                return None
        # Verify total object count
        objs = _extract_objects(grid)
        if len(objs) != sum(counts):
            return None
        # Output: 2×2 grid where each cell = count for that quadrant
        # Values 0-4 (ARC colors 0-4 represent counts)
        output = np.array([[counts[0], counts[1]], [counts[2], counts[3]]], dtype=int)
        # Verify output is non-trivial (not all zeros or all same)
        if np.all(output == 0) or np.all(output == output[0, 0]):
            return None
        return {'input': grid.tolist(), 'output': output.tolist()}

    rng = np.random.default_rng(seed)
    colors = list(range(1, 10))
    generated = 0

    count_patterns = [
        [1, 2, 0, 1], [2, 0, 1, 3], [0, 1, 2, 0], [3, 1, 0, 2],
        [1, 0, 2, 1], [2, 1, 1, 0], [0, 2, 1, 3], [1, 1, 1, 0],
        [2, 0, 0, 1], [3, 0, 1, 0], [0, 3, 1, 1], [1, 2, 2, 0],
    ]

    for task_idx in range(n_tasks * 3):
        if generated >= n_tasks:
            break
        color = int(rng.choice(colors))

        # Choose a base count pattern and rotate for variety
        base_pattern = count_patterns[task_idx % len(count_patterns)]

        examples = []
        for ex_idx in range(8):
            if len(examples) >= 5:
                break
            # Vary counts slightly for each example while keeping pattern structure
            offset = ex_idx % 4
            counts = [(base_pattern[(q + offset) % 4]) for q in range(4)]
            ex = make_example(counts, color, rng)
            if ex is not None:
                examples.append(ex)

        if len(examples) < 4:
            continue

        task = {'train': examples[:4], 'test': examples[4:5] if len(examples) >= 5 else examples[:1]}
        fname = f'type5_{generated:03d}.json'
        with open(os.path.join(out_dir, fname), 'w') as f:
            json.dump(task, f)
        print(f'  type5_{generated:03d}: color={color}, pattern={base_pattern}')
        generated += 1

    print(f'\nType 5 benchmark tasks: {generated}')
    return generated


def generate_benchmark_type6(out_dir: str, n_tasks: int = 100, seed: int = 6000):
    """
    TYPE 6 — FORMS+OBJECTS+NUMBER: Count-Rotate-Replicate (three-way crossing).

    Program: rotate_n(render_object(obj_largest(extract_objects(input))),
                      count_objects(extract_objects(input)) - 1)
    The count of SMALL objects (not largest) drives the rotation.
    1 small obj → rotate90(largest), 2 → rotate180, 3 → rotate270.

    AST solver: _try_forms_objects_number_count_rotate (added in T5)
    """
    os.makedirs(out_dir, exist_ok=True)
    from src.spelke_dsl.l_objects import _extract_objects
    from src.spelke_dsl.l_forms import _rotate_n

    LARGE_SHAPES = {
        'L4':      [(0,0),(1,0),(2,0),(2,1),(2,2)],
        'T4':      [(0,0),(0,1),(0,2),(1,1),(2,1)],
        'S4':      [(0,1),(0,2),(1,0),(1,1),(2,0)],
        'Rect2x3': [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2)],
        'Plus':    [(0,1),(1,0),(1,1),(1,2),(2,1)],
        'Z4':      [(0,0),(0,1),(1,1),(1,2),(2,2)],
    }

    def make_example(large_shape: str, large_color: int, small_color: int,
                     n_small: int, rng: np.random.Generator,
                     h: int = 12, w: int = 12) -> dict | None:
        lshape = LARGE_SHAPES[large_shape]
        lr_max = max(r for r, c in lshape)
        lc_max = max(c for r, c in lshape)
        if h - lr_max < 3 or w - lc_max < 3:
            return None
        lr_off = int(rng.integers(1, h - lr_max - 1))
        lc_off = int(rng.integers(1, min(4, w - lc_max - 2)))
        grid = np.zeros((h, w), dtype=int)
        large_cells = set()
        for r, c in lshape:
            grid[r + lr_off, c + lc_off] = large_color
            large_cells.add((r + lr_off, c + lc_off))

        # Place n_small isolated markers in right half
        placed = 0
        occupied = set(large_cells)
        for _ in range(300):
            if placed >= n_small:
                break
            mr = int(rng.integers(0, h))
            mc = int(rng.integers(w // 2, w))
            neighbors = {(mr+dr, mc+dc) for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]}
            if (mr, mc) not in occupied and not neighbors & occupied:
                grid[mr, mc] = small_color
                occupied.add((mr, mc))
                placed += 1

        if placed < n_small:
            return None

        objs = _extract_objects(grid)
        if len(objs) != 1 + n_small:
            return None
        largest = max(objs, key=lambda o: o.size)
        if largest.color != large_color:
            return None

        rendered = largest.to_grid()
        output = _rotate_n(rendered, n_small)
        # Verify non-trivial
        if np.array_equal(output, rendered):
            return None
        return {'input': grid.tolist(), 'output': output.tolist()}

    rng = np.random.default_rng(seed)
    colors = list(range(1, 10))
    large_names = list(LARGE_SHAPES.keys())
    generated = 0

    for task_idx in range(n_tasks * 3):
        if generated >= n_tasks:
            break
        large_shape = large_names[task_idx % len(large_names)]
        large_color = int(rng.choice(colors))
        small_color = int(rng.choice([c for c in colors if c != large_color]))

        # KEY: vary n_small WITHIN each task so counting is required across examples
        examples = []
        for n_small in [1, 2, 3, 1, 2]:
            for _ in range(15):
                ex = make_example(large_shape, large_color, small_color, n_small, rng)
                if ex is not None:
                    examples.append(ex)
                    break

        if len(examples) < 4:
            continue

        # Verify outputs differ across examples (proves counting is needed)
        outputs = [np.array(ex['output']).tobytes() for ex in examples[:4]]
        if len(set(outputs)) < 2:
            continue

        task = {'train': examples[:4], 'test': examples[4:5] if len(examples) >= 5 else examples[:1]}
        fname = f'type6_{generated:03d}.json'
        with open(os.path.join(out_dir, fname), 'w') as f:
            json.dump(task, f)
        print(f'  type6_{generated:03d}: shape={large_shape}, n_small_pattern=[1,2,3,1]')
        generated += 1

    print(f'\nType 6 benchmark tasks: {generated}')
    return generated


def generate_benchmark_type7(out_dir: str, n_tasks: int = 100, seed: int = 7000):
    """
    TYPE 7 — AGENTS+OBJECTS: Path Trace.

    Program: trace_path(input, agent_color, goal_color)
    Agent (small 1-cell object, agent_color) traces straight-line path to goal
    (large multi-cell block, goal_color). Output has path filled with agent_color.

    AST solver: _try_agents_objects_path (added in T5)
    """
    os.makedirs(out_dir, exist_ok=True)
    from src.spelke_dsl.l_agents import _draw_path
    from src.spelke_dsl.l_objects import _extract_objects

    def make_example(agent_color: int, goal_color: int, rng: np.random.Generator,
                     h: int = 10, w: int = 10) -> dict | None:
        grid = np.zeros((h, w), dtype=int)

        # Place 3x3 goal block in one area
        gr = int(rng.integers(1, h - 4))
        gc = int(rng.integers(1, w - 4))
        goal_cells = set()
        for dr in range(3):
            for dc in range(3):
                grid[gr + dr, gc + dc] = goal_color
                goal_cells.add((gr + dr, gc + dc))

        # Place agent in a different quadrant (far from goal)
        for _ in range(100):
            ar = int(rng.integers(0, h))
            ac = int(rng.integers(0, w))
            if (ar, ac) not in goal_cells:
                dist = abs(ar - gr) + abs(ac - gc)
                if dist > 4:  # ensure some path distance
                    grid[ar, ac] = agent_color
                    break
        else:
            return None

        # Verify extraction
        objs = _extract_objects(grid)
        if len(objs) != 2:
            return None
        largest = max(objs, key=lambda o: o.size)
        smallest = min(objs, key=lambda o: o.size)
        if largest.color != goal_color or smallest.color != agent_color:
            return None

        # Compute output using trace_path
        output = _draw_path(grid, agent_color, goal_color)
        if np.array_equal(output, grid):
            return None

        return {'input': grid.tolist(), 'output': output.tolist()}

    rng = np.random.default_rng(seed)
    colors = list(range(1, 10))
    generated = 0

    for task_idx in range(n_tasks * 3):
        if generated >= n_tasks:
            break
        agent_color = int(rng.choice(colors))
        goal_color = int(rng.choice([c for c in colors if c != agent_color]))

        examples = []
        for _ in range(20):
            if len(examples) >= 5:
                break
            ex = make_example(agent_color, goal_color, rng)
            if ex is not None:
                examples.append(ex)

        if len(examples) < 4:
            continue

        task = {'train': examples[:4], 'test': examples[4:5] if len(examples) >= 5 else examples[:1]}
        fname = f'type7_{generated:03d}.json'
        with open(os.path.join(out_dir, fname), 'w') as f:
            json.dump(task, f)
        print(f'  type7_{generated:03d}: agent={agent_color}, goal={goal_color}')
        generated += 1

    print(f'\nType 7 benchmark tasks: {generated}')
    return generated


def generate_benchmark_type8(out_dir: str, n_tasks: int = 100, seed: int = 8000):
    """
    TYPE 8 — OBJECTS+PLACES+NUMBER: Sorted Placement.

    Simpler version: 2 objects sorted by size → place_in_quadrant_8x8.
    Small object → TL quadrant (0), Large object → BR quadrant (3).
    Program: place small in quad_tl, large in quad_br.

    AST solver: _try_objects_places_number_sorted (added in T5)
    """
    os.makedirs(out_dir, exist_ok=True)
    from src.spelke_dsl.l_objects import _extract_objects
    from src.spelke_dsl.l_places import _place_in_quadrant_8x8

    SMALL_SHAPES = {
        'Dot': [(0, 0)],
        'I2':  [(0,0),(0,1)],
        'Vert2': [(0,0),(1,0)],
    }
    LARGE_SHAPES = {
        'Rect2x3': [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2)],
        'L3':      [(0,0),(1,0),(2,0),(2,1)],
        'T3':      [(0,0),(0,1),(0,2),(1,1)],
        'Plus':    [(0,1),(1,0),(1,1),(1,2),(2,1)],
    }

    def make_example(small_shape: str, small_color: int, large_shape: str, large_color: int,
                     rng: np.random.Generator) -> dict | None:
        h, w = 8, 8
        grid = np.zeros((h, w), dtype=int)

        # Place small object in right half of input (not in final output position)
        sshape = SMALL_SHAPES[small_shape]
        sr_max = max(r for r, c in sshape)
        sc_max = max(c for r, c in sshape)
        for _ in range(50):
            sr = int(rng.integers(0, h - sr_max - 1))
            sc = int(rng.integers(w // 2, w - sc_max - 1))
            small_cells = {(sr + r, sc + c) for r, c in sshape}
            if all(0 <= r < h and 0 <= c < w for r, c in small_cells):
                for r, c in small_cells:
                    grid[r, c] = small_color
                break
        else:
            return None

        # Place large object in left half
        lshape = LARGE_SHAPES[large_shape]
        lr_max = max(r for r, c in lshape)
        lc_max = max(c for r, c in lshape)
        occupied = set(small_cells)
        for _ in range(50):
            lr = int(rng.integers(0, h - lr_max - 1))
            lc = int(rng.integers(0, min(3, w // 2 - lc_max - 1)))
            large_cells = {(lr + r, lc + c) for r, c in lshape}
            buffer = {(r + dr, c + dc) for r, c in large_cells
                      for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]}
            if not large_cells & occupied and not buffer & occupied:
                for r, c in large_cells:
                    grid[r, c] = large_color
                break
        else:
            return None

        # Verify extraction
        objs = _extract_objects(grid)
        if len(objs) != 2:
            return None
        largest = max(objs, key=lambda o: o.size)
        smallest = min(objs, key=lambda o: o.size)
        if largest.color != large_color or smallest.color != small_color:
            return None

        # Output: place objects sorted by size
        # Small → TL quadrant (0), Large → BR quadrant (3)
        output = np.zeros((8, 8), dtype=int)
        small_grid = smallest.to_grid()
        large_grid = largest.to_grid()
        # Place small in TL
        sh, sw = small_grid.shape
        output[:min(sh,4), :min(sw,4)] = small_grid[:min(sh,4), :min(sw,4)]
        # Place large in BR
        lh, lw = large_grid.shape
        output[8-min(lh,4):, 8-min(lw,4):] = large_grid[:min(lh,4), :min(lw,4)]

        if np.array_equal(output, grid):
            return None

        return {'input': grid.tolist(), 'output': output.tolist()}

    rng = np.random.default_rng(seed)
    colors = list(range(1, 10))
    small_names = list(SMALL_SHAPES.keys())
    large_names = list(LARGE_SHAPES.keys())
    generated = 0

    for task_idx in range(n_tasks * 3):
        if generated >= n_tasks:
            break
        small_shape = small_names[task_idx % len(small_names)]
        large_shape = large_names[task_idx % len(large_names)]
        small_color = int(rng.choice(colors))
        large_color = int(rng.choice([c for c in colors if c != small_color]))

        examples = []
        for _ in range(20):
            if len(examples) >= 5:
                break
            ex = make_example(small_shape, small_color, large_shape, large_color, rng)
            if ex is not None:
                examples.append(ex)

        if len(examples) < 4:
            continue

        task = {'train': examples[:4], 'test': examples[4:5] if len(examples) >= 5 else examples[:1]}
        fname = f'type8_{generated:03d}.json'
        with open(os.path.join(out_dir, fname), 'w') as f:
            json.dump(task, f)
        print(f'  type8_{generated:03d}: small={small_shape}/{small_color}, large={large_shape}/{large_color}')
        generated += 1

    print(f'\nType 8 benchmark tasks: {generated}')
    return generated


def generate_benchmark_tier3A(out_dir: str, n_tasks: int = 25, seed: int = 10000):
    """
    Tier 3-A: COUNT-RENDER-ROTATE90 (compounding task, requires abs_0_3).

    Program: rotate90(render_count_colored(count_objects(extract_objects(input)),
                                           obj_color(obj_largest(extract_objects(input)))))

    Input:  1 LARGE shape (COLOR_A) + N-1 small 1-cell markers (COLOR_B)
            Total N = 1 + (N-1) objects; N varies 2..5 per example
    Output: N×1 column of COLOR_A (rotate90 of the 1×N row abs_0_3 produces)

    Why two colors (large + markers)?
      - count_objects = N (counts ALL objects)
      - obj_largest = the large shape → COLOR_A
      - abs_0_3(input) = 1×N row of COLOR_A
      - rotate90(abs_0_3(input)) = N×1 column of COLOR_A

    UNSOLVABLE in cycle 0:
      - count_output heuristic sees 2 colors (A_cells + N-1 B cells);
        its bar-graph would need N_colors=2 columns but output has 1 → MISMATCH
      - _try_number_objects_count produces 1×N row, not N×1 column
      - No AST strategy composes count-render + rotate90
    SOLVABLE in cycle 1:
      - abs_0_3 (grid→grid, discovered in cycle 0) = count-render pattern
      - _try_library_chains Phase 2 finds rotate90(abs_0_3(input))

    Crossing: NUMBER+OBJECTS+FORMS
    """
    os.makedirs(out_dir, exist_ok=True)
    from src.spelke_dsl.l_objects import _extract_objects

    LARGE_SHAPES = {
        'L3':      [(0,0),(1,0),(2,0),(2,1)],
        'T3':      [(0,0),(0,1),(0,2),(1,1)],
        'Plus':    [(0,1),(1,0),(1,1),(1,2),(2,1)],
        'Rect2x2': [(0,0),(0,1),(1,0),(1,1)],
        'I3':      [(0,0),(0,1),(0,2)],
        'S':       [(0,1),(0,2),(1,0),(1,1)],
    }

    def make_example(shape_name: str, large_color: int, marker_color: int,
                     n_markers: int, rng: np.random.Generator,
                     h: int = 10, w: int = 10) -> dict | None:
        """1 large shape + n_markers small dots → total N = 1 + n_markers objects."""
        shape_cells = LARGE_SHAPES[shape_name]
        max_r = max(r for r, c in shape_cells)
        max_c = max(c for r, c in shape_cells)
        if h - max_r < 3 or w - max_c < 3:
            return None

        off_r = int(rng.integers(0, max(1, h - max_r - 2)))
        off_c = int(rng.integers(0, max(1, min(3, w - max_c - 2))))
        grid = np.zeros((h, w), dtype=int)
        occupied = set()
        for r, c in shape_cells:
            grid[r + off_r, c + off_c] = large_color
            occupied.add((r + off_r, c + off_c))

        # Place n_markers isolated 1-cell markers (isolated = no adjacency to shape)
        placed = 0
        for _ in range(300):
            mr, mc = int(rng.integers(0, h)), int(rng.integers(0, w))
            too_close = any(abs(mr - pr) <= 1 and abs(mc - pc) <= 1 for pr, pc in occupied)
            if (mr, mc) not in occupied and not too_close:
                grid[mr, mc] = marker_color
                occupied.add((mr, mc))
                placed += 1
                if placed == n_markers:
                    break
        if placed < n_markers:
            return None

        # Verify: exactly 1 + n_markers objects, largest = shape
        objs = _extract_objects(grid)
        n_total = 1 + n_markers
        if len(objs) != n_total:
            return None
        largest = max(objs, key=lambda o: o.size)
        if largest.color != large_color or largest.size < 2:
            return None

        # abs_0_3(input) = render_count_colored(N_total, large_color) = 1×N_total row
        # rotate90(1×N_total row) = N_total×1 column of large_color
        output = np.full((n_total, 1), large_color, dtype=int)
        return {'input': grid.tolist(), 'output': output.tolist()}

    rng = np.random.default_rng(seed)
    colors = list(range(1, 10))
    shape_names = list(LARGE_SHAPES.keys())
    generated = 0

    for task_idx in range(n_tasks * 3):
        if generated >= n_tasks:
            break
        shape_name = shape_names[task_idx % len(shape_names)]
        large_color = int(rng.choice(colors))
        marker_color = int(rng.choice([c for c in colors if c != large_color]))

        # n_markers varies 1..4 → total N varies 2..5
        examples = []
        for n_markers in [1, 2, 3, 4, 1]:
            for _ in range(15):
                ex = make_example(shape_name, large_color, marker_color, n_markers, rng)
                if ex is not None:
                    examples.append(ex)
                    break

        if len(examples) < 4:
            continue
        # Verify varied output heights
        output_heights = set(len(ex['output']) for ex in examples[:4])
        if len(output_heights) < 2:
            continue

        task = {'train': examples[:4], 'test': examples[4:5] if len(examples) >= 5 else examples[:1]}
        fname = f'tier3a_{generated:03d}.json'
        with open(os.path.join(out_dir, fname), 'w') as f:
            json.dump(task, f)
        generated += 1

    print(f'Tier 3-A (count-render-rotate90): {generated}/{n_tasks} tasks')
    return generated


def generate_benchmark_tier3B(out_dir: str, n_tasks: int = 25, seed: int = 11000):
    """
    Tier 3-B: TILE-ROTATE90 (compounding task, requires abs_0_4).

    Program: rotate90(tile_n(render_object(obj_largest(extract_objects(input))),
                             count_objects(extract_objects(input))))

    Input:  1 large shape object + N-1 small 1-cell markers (total N objects)
    Output: rotate90 of the horizontally-tiled large object (N copies rotated)

    UNSOLVABLE in cycle 0:
      - _try_number_objects_tile produces horizontal tile, not rotated tile
      - No strategy composes tile-n + rotate90
    SOLVABLE in cycle 1:
      - abs_0_4 (grid→grid, discovered in cycle 0) = tile-n pattern
      - _try_library_chains Phase 2 finds rotate90(abs_0_4(input))

    Crossing: NUMBER+OBJECTS+FORMS
    """
    os.makedirs(out_dir, exist_ok=True)
    from src.spelke_dsl.l_objects import _extract_objects

    SHAPES = {
        'L3':      [(0,0),(1,0),(2,0),(2,1)],
        'T3':      [(0,0),(0,1),(0,2),(1,1)],
        'Plus':    [(0,1),(1,0),(1,1),(1,2),(2,1)],
        'Rect2x2': [(0,0),(0,1),(1,0),(1,1)],
        'I3':      [(0,0),(0,1),(0,2)],
        'S':       [(0,1),(0,2),(1,0),(1,1)],
    }

    def make_example(shape_name: str, obj_color: int, marker_color: int,
                     n_markers: int, rng: np.random.Generator,
                     h: int = 10, w: int = 10) -> dict | None:
        shape_cells = SHAPES[shape_name]
        max_r = max(r for r, c in shape_cells)
        max_c = max(c for r, c in shape_cells)
        if h - max_r < 3 or w - max_c < 3:
            return None

        off_r = int(rng.integers(0, max(1, h - max_r - 2)))
        off_c = int(rng.integers(0, max(1, min(3, w - max_c - 2))))
        grid = np.zeros((h, w), dtype=int)
        occupied = set()
        for r, c in shape_cells:
            grid[r + off_r, c + off_c] = obj_color
            occupied.add((r + off_r, c + off_c))

        placed = 0
        for _ in range(300):
            mr, mc = int(rng.integers(0, h)), int(rng.integers(0, w))
            too_close = any(abs(mr - pr) <= 1 and abs(mc - pc) <= 1 for pr, pc in occupied)
            if (mr, mc) not in occupied and not too_close:
                grid[mr, mc] = marker_color
                occupied.add((mr, mc))
                placed += 1
                if placed == n_markers:
                    break
        if placed < n_markers:
            return None

        objs = _extract_objects(grid)
        if len(objs) != 1 + n_markers:
            return None
        largest = max(objs, key=lambda o: o.size)
        if largest.color != obj_color or largest.size < 2:
            return None

        # Compute tile_n output then rotate90
        rendered = largest.to_grid()
        n_tiles = 1 + n_markers
        tiled = np.hstack([rendered] * n_tiles).astype(int)
        output = np.rot90(tiled, k=-1).copy()  # rotate90 clockwise

        return {'input': grid.tolist(), 'output': output.tolist()}

    rng = np.random.default_rng(seed)
    colors = list(range(1, 10))
    shape_names = list(SHAPES.keys())
    generated = 0

    for task_idx in range(n_tasks * 3):
        if generated >= n_tasks:
            break
        shape_name = shape_names[task_idx % len(shape_names)]
        obj_color = int(rng.choice(colors))
        marker_color = int(rng.choice([c for c in colors if c != obj_color]))
        h, w = 10, 10

        examples = []
        for n_markers in [1, 2, 3, 1, 2]:
            for _ in range(15):
                ex = make_example(shape_name, obj_color, marker_color, n_markers, rng, h, w)
                if ex is not None:
                    examples.append(ex)
                    break

        if len(examples) < 4:
            continue
        # Verify varied outputs (different shapes due to different tile counts)
        output_shapes = set((len(ex['output']), len(ex['output'][0])) for ex in examples[:4])
        if len(output_shapes) < 2:
            continue

        task = {'train': examples[:4], 'test': examples[4:5] if len(examples) >= 5 else examples[:1]}
        fname = f'tier3b_{generated:03d}.json'
        with open(os.path.join(out_dir, fname), 'w') as f:
            json.dump(task, f)
        generated += 1

    print(f'Tier 3-B (tile-rotate90): {generated}/{n_tasks} tasks')
    return generated


def generate_benchmark_tier3C(out_dir: str, n_tasks: int = 25, seed: int = 12000):
    """
    Tier 3-C: RENDER-FLIP-ROTATE90 (compounding task, requires abs_0_5).

    Program: rotate90(flip_h(render_object(obj_largest(extract_objects(input)))))

    Input:  1 large NON-SQUARE asymmetric object + 1 small 1-cell marker (different color)
    Output: rotate90(flip_h(largest_object_rendered)) — double-transformed shape

    UNSOLVABLE in cycle 0:
      - _try_form_on_extracted_object tries SINGLE transforms only; any single
        transform applied to an L/J/I shape does not match the double-transform output
      - Shapes are chosen non-square so rotate90 changes dimensions, avoiding
        accidental match with 2D symmetric transform equivalences
    SOLVABLE in cycle 1:
      - abs_0_5 (grid→grid, FORMS+OBJECTS, discovered in cycle 0)
        = flip_h(render_object(obj_largest(extract_objects(input))))
      - _try_library_chains Phase 2 finds rotate90(abs_0_5(input))

    Crossing: FORMS+OBJECTS (deeper than Type 4 — requires discovered abstraction)
    """
    os.makedirs(out_dir, exist_ok=True)
    from src.spelke_dsl.l_objects import _extract_objects

    # Use ONLY non-square shapes so bounding box H ≠ W.
    # This ensures rotate90(output) always changes shape dimensions,
    # making it impossible for any single flip transform to match.
    NON_SQUARE_SHAPES = {
        'L5':    [(0,0),(1,0),(2,0),(3,0),(4,0),(4,1)],   # 5×2 bounding box
        'J5':    [(0,1),(1,1),(2,1),(3,1),(4,0),(4,1)],   # 5×2
        'I4':    [(0,0),(0,1),(0,2),(0,3)],               # 1×4
        'I5':    [(0,0),(0,1),(0,2),(0,3),(0,4)],         # 1×5
        'T4H':   [(0,0),(0,1),(0,2),(0,3),(1,1)],         # 2×4
        'L4':    [(0,0),(1,0),(2,0),(3,0),(3,1)],         # 4×2
        'J4':    [(0,1),(1,1),(2,1),(3,0),(3,1)],         # 4×2
        'Rect3x2': [(0,0),(0,1),(1,0),(1,1),(2,0),(2,1)], # 3×2
    }

    def make_example(shape_name: str, obj_color: int, marker_color: int,
                     rng: np.random.Generator, h: int = 12, w: int = 12) -> dict | None:
        shape_cells = NON_SQUARE_SHAPES[shape_name]
        max_r = max(r for r, c in shape_cells)
        max_c = max(c for r, c in shape_cells)
        if h - max_r < 3 or w - max_c < 3:
            return None

        off_r = int(rng.integers(1, max(2, h - max_r - 1)))
        off_c = int(rng.integers(1, max(2, w - max_c - 1)))
        grid = np.zeros((h, w), dtype=int)
        placed = set()
        for r, c in shape_cells:
            grid[r + off_r, c + off_c] = obj_color
            placed.add((r + off_r, c + off_c))

        # Place small marker (1-cell) far from the shape
        for _ in range(100):
            mr, mc = int(rng.integers(0, h)), int(rng.integers(0, w))
            if (mr, mc) not in placed:
                grid[mr, mc] = marker_color
                break
        else:
            return None

        # Verify extraction
        objs = _extract_objects(grid)
        if len(objs) < 2:
            return None
        largest = max(objs, key=lambda o: o.size)
        if largest.color != obj_color or largest.size < 3:
            return None

        # Rendered shape must be non-square
        rendered = largest.to_grid()
        rh, rw = rendered.shape
        if rh == rw:
            return None  # skip square renders (can cause symmetric confusion)

        # Output: rotate90(flip_h(render_object(obj_largest(...))))
        flipped = np.fliplr(rendered).copy()
        output = np.rot90(flipped, k=-1).copy()  # clockwise

        # Verify output differs from ALL 6 single-transform variants
        all_single = [
            np.rot90(rendered, k=-1),    # rotate90
            np.rot90(rendered, k=2),     # rotate180
            np.rot90(rendered, k=-3),    # rotate270
            np.fliplr(rendered),         # flip_h
            np.flipud(rendered),         # flip_v
            rendered.T,                  # transpose
        ]
        for s in all_single:
            if s.shape == output.shape and np.array_equal(output, s):
                return None  # degenerate — skip

        return {'input': grid.tolist(), 'output': output.tolist()}

    rng = np.random.default_rng(seed)
    colors = list(range(1, 10))
    shape_names = list(NON_SQUARE_SHAPES.keys())
    generated = 0

    for task_idx in range(n_tasks * 4):
        if generated >= n_tasks:
            break
        shape_name = shape_names[task_idx % len(shape_names)]
        obj_color = int(rng.choice(colors))
        marker_color = int(rng.choice([c for c in colors if c != obj_color]))

        examples = []
        for _ in range(40):
            if len(examples) >= 5:
                break
            ex = make_example(shape_name, obj_color, marker_color, rng)
            if ex is not None:
                examples.append(ex)

        if len(examples) < 4:
            continue

        task = {'train': examples[:4], 'test': examples[4:5] if len(examples) >= 5 else examples[:1]}
        fname = f'tier3c_{generated:03d}.json'
        with open(os.path.join(out_dir, fname), 'w') as f:
            json.dump(task, f)
        generated += 1

    print(f'Tier 3-C (render-flip-rotate90): {generated}/{n_tasks} tasks')
    return generated


def generate_spelke_benchmark(base_dir: str, seed: int = 9000):
    """
    Generate all 1000 tasks for the Spelke Bootstrap Suite benchmark.
    Writes to base_dir/spelke_benchmark_staging/type{N}/ directories.
    T6 will assemble into base_dir/spelke_benchmark/ with 80/20 split.
    """
    staging = os.path.join(base_dir, 'spelke_benchmark_staging')
    print(f'\n=== Spelke Bootstrap Suite — 1000 Task Generation ===')
    print(f'Staging dir: {staging}\n')

    type1_dir = os.path.join(staging, 'type1')
    type2_dir = os.path.join(staging, 'type2')
    type3_dir = os.path.join(staging, 'type3')
    type4_dir = os.path.join(staging, 'type4')
    type5_dir = os.path.join(staging, 'type5')
    type6_dir = os.path.join(staging, 'type6')
    type7_dir = os.path.join(staging, 'type7')
    type8_dir = os.path.join(staging, 'type8')

    print('--- Type 1: NUMBER+OBJECTS count-render (95 new + 30 existing = 125 total) ---')
    n1 = generate_benchmark_type1(type1_dir, n_tasks=95, seed=seed + 1000)

    print('\n--- Type 2: NUMBER+OBJECTS tile (95 new + 30 existing = 125 total) ---')
    n2 = generate_benchmark_type2(type2_dir, n_tasks=95, seed=seed + 2000)

    print('\n--- Type 3: FORMS+NUMBER rotate-by-count (100 tasks) ---')
    n3 = generate_benchmark_type3(type3_dir, n_tasks=100, seed=seed + 3000)

    print('\n--- Type 4: OBJECTS+FORMS mirror-by-size (100 tasks) ---')
    n4 = generate_benchmark_type4(type4_dir, n_tasks=100, seed=seed + 4000)

    print('\n--- Type 5: NUMBER+PLACES quadrant-count (100 tasks) ---')
    n5 = generate_benchmark_type5(type5_dir, n_tasks=100, seed=seed + 5000)

    print('\n--- Type 6: FORMS+OBJECTS+NUMBER three-way (100 tasks) ---')
    n6 = generate_benchmark_type6(type6_dir, n_tasks=100, seed=seed + 6000)

    print('\n--- Type 7: AGENTS+OBJECTS path-trace (100 tasks) ---')
    n7 = generate_benchmark_type7(type7_dir, n_tasks=100, seed=seed + 7000)

    print('\n--- Type 8: OBJECTS+PLACES+NUMBER sorted-placement (100 tasks) ---')
    n8 = generate_benchmark_type8(type8_dir, n_tasks=100, seed=seed + 8000)

    total = n1 + n2 + n3 + n4 + n5 + n6 + n7 + n8
    print(f'\n=== Generation complete: {total} new tasks ===')
    print(f'Type 1: {n1}/95, Type 2: {n2}/95, Type 3: {n3}/100, Type 4: {n4}/100')
    print(f'Type 5: {n5}/100, Type 6: {n6}/100, Type 7: {n7}/100, Type 8: {n8}/100')
    return total


if __name__ == '__main__':
    out_dir_synth = os.path.join(
        os.path.dirname(__file__), '..', '..', 'data', 'synthetic_cross_system'
    )
    generate_all(out_dir_synth, tasks_per_transform=3, seed=42)

    out_dir_compound = os.path.join(
        os.path.dirname(__file__), '..', '..', 'data', 'synthetic_compounding'
    )
    print('\n--- Generating compounding tasks ---')
    generate_compounding_tasks(out_dir_compound, n_tasks=5, seed=100)
    print(f'\nCompounding tasks written to: {os.path.abspath(out_dir_compound)}')

    # Curriculum Phase 2 task generation
    print('\n--- Generating NUMBER+OBJECTS tasks (TYPE A) ---')
    out_dir_no = os.path.join(
        os.path.dirname(__file__), '..', '..', 'data', 'synthetic_number_objects'
    )
    generate_number_objects_tasks(out_dir_no, n_tasks=30, seed=200)
    print(f'\nNUMBER+OBJECTS tasks written to: {os.path.abspath(out_dir_no)}')

    print('\n--- Generating FORMS+OBJECTS+PLACES tasks (TYPE B) ---')
    out_dir_fop = os.path.join(
        os.path.dirname(__file__), '..', '..', 'data', 'synthetic_forms_objects_places'
    )
    generate_forms_objects_places_tasks(out_dir_fop, n_tasks=30, seed=300)
    print(f'\nFORMS+OBJECTS+PLACES tasks written to: {os.path.abspath(out_dir_fop)}')

    print('\n--- Generating NUMBER+OBJECTS replicate tasks (TYPE C) ---')
    out_dir_nor = os.path.join(
        os.path.dirname(__file__), '..', '..', 'data', 'synthetic_number_objects_replicate'
    )
    generate_number_objects_replicate_tasks(out_dir_nor, n_tasks=30, seed=400)
    print(f'\nNUMBER+OBJECTS replicate tasks written to: {os.path.abspath(out_dir_nor)}')

    print('\n--- Generating count_cells tasks (TYPE D) ---')
    out_dir_cells = os.path.join(
        os.path.dirname(__file__), '..', '..', 'data', 'synthetic_count_cells'
    )
    generate_count_cells_tasks(out_dir_cells, n_tasks=30, seed=500)
    print(f'\ncount_cells tasks written to: {os.path.abspath(out_dir_cells)}')

    print('\n--- Generating PERSONS+OBJECTS tasks (TYPE E+F) ---')
    out_dir_persons = os.path.join(
        os.path.dirname(__file__), '..', '..', 'data', 'synthetic_persons_objects'
    )
    generate_persons_objects_tasks(out_dir_persons, n_tasks=30, seed=600)
    print(f'\nPERSONS+OBJECTS tasks written to: {os.path.abspath(out_dir_persons)}')
