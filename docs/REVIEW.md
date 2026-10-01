# Code review and modernization

Reviewed upstream commit `f40b338`. The replacement keeps the original boards and move semantics.
The original sources remain accessible with `git show f40b338:BlockStar.py` and
`git show f40b338:AgentFormer.py`.

| Upstream finding | Consequence | Resolution |
| --- | --- | --- |
| 1,000 attempts at module scope, appending to `results.txt`. | Imports cause long work and mutate training data. | Import-safe modules; explicit CLI; atomic JSON artifacts. |
| Bare exceptions discard failures; runs saved without full goal equality. | Results need independent validation. | Separate cell referee; error codes; count every failed attempt. |
| A* costs unit translations; heuristic sums displacement over every piece cell. | Heuristic overestimates edge cost for multi-cell pieces; objective differs from slide count. | BFS over every legal whole slide, with unit edge cost. |
| DFS called topological sort accepts dependency cycles. | Derived order lacks topological guarantees. | Dynamic geometry, parking search, frozen-goal reachability checks. |
| Fitness subtracts the distance it adds; state fitness calls have wrong arguments. | Unreliable or unusable objectives. | Explicit slide-count scores and tunable named geometric features. |
| Duplicate empty-space method, unfinished Population, unused GA constants. | Dead code obscures actual behavior. | Remove scaffolding; implement evolutionary parameter search with logged fitness. |
| Recursive retries can omit bound increments; a retry counter resets inside its loop. | Attempts may hang or recurse indefinitely. | Every successful stage fixes a new piece; bounded attempts report failure. |
| Deep copies and full heuristic reconstruction after every move. | High overhead. | Immutable anchor tuples and precomputed integer footprint masks. |
| Invalid moves silently ignored; nonpositive distance accepted by parts of the API. | Intended and executed actions differ. | Explicit positive integral moves, swept-cell validation, transactional failure. |
| AgentFormer uses an absolute path, unconditional MPS, import-time training, unordered vocabularies. | Nonportable and irreproducible. | Explicit train/sample CLI, stable action space, seeds, device selection. |
| Unconditioned character generation; decoding mutates lists during iteration. | Malformed or illegal moves. | Current/target coordinate inputs; whole-slide outputs; legality masking and replay. |
| Attention scales with embedding width, not head width. | Incorrect attention scaling. | PyTorch scaled dot-product attention; zero evaluation dropout. |
| Character-level data split and no tracked checkpoints. | Episode leakage and poor reproducibility. | Deduplicate and split episodes, save config/seed/data digest, safely load weights. |
| No packaging, tests, CI, or declared dependencies. | Fragile installation and undetected regressions. | pyproject, lockfile, zero-dependency core, optional training, tests, pinned CI actions. |

## Evidence

- `benchmarks/upstream-audit.json`: all 883 historical records inspected; 882 valid, minimum 85.
  Record 183 attempts to move W through C at move 7. Thirteen valid records are duplicates.
- `benchmarks/benchmark.json`: 15 method/seed runs, 300 rollouts each. Every adaptive-and-refined
  search reached 40 slides. `benchmarks/solutions/` retains solutions and per-rollout logs.
- `benchmarks/extended-discovery.json`: fresh search, seed 2026, 1,500 rollouts. Trial 227
  discovers a 48-slide raw run and refines it to 37. Stored trajectories are not inputs.
- `benchmarks/champion.json`: complete independently replayed result; unit distance is recorded
  separately and is not the optimization objective.
- `benchmarks/champion-polish.json`: 100 further restarts without improvement. A plateau is
  evidence about this search, not a global optimality proof.
- Core tested on Python 3.14.6; optional training, loading, sampling, and the full suite
  exercised on Python 3.13.15 / PyTorch 2.14.0 on CPU.

## Limits and next research steps

The global planner is incomplete: it freezes finished pieces, considers limited relaxed
corridors, parks blockers one at a time, and greedily accepts successful stages. Exact
one-piece rescheduling is only exact with the other pieces' move order fixed. Exploring
equal-cost schedules helps but does not prove a global minimum. The current certified lower
bound is 30 and best upper bound is 37.

The strongest next experiment is bounded multi-piece rescheduling or constraint-based search
on that seven-slide gap, using the same independent referee. Externally mandated placement
orders must be modeled before applying these results to such a site. Dynamic obstacles,
lifting, rotations, simultaneous moves, crane reach, and physical movement costs are outside
this puzzle's rules.

The imitation model has a tested training/inference pipeline, but its short smoke test does
not demonstrate a solved rollout. The historical data contains highly variable, often
inefficient solutions. Future studies should train on newly discovered short trajectories,
compare legal-policy success with geometric search on unseen puzzles, and report failures.

## Implementation references

Packaging follows the [PyPA pyproject guidance](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/).
Attention follows the [PyTorch SDPA API](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html),
including the evaluation dropout setting. CI revisions were resolved from official
[checkout](https://github.com/actions/checkout) and [setup-python](https://github.com/actions/setup-python)
v6 tags on 2026-09-05.
