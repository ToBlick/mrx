---
name: general
description: Working knowledge for the MRX code base (matrix-free FEEC/B-spline 3D MHD equilibrium solver in JAX, no nested-surface assumption). Use this skill for ANY task in the mrx repository or its worktrees - reading or changing mrx/ (derham_sequence, geometry, operators, mass, metric_lumping, slab_mass, multigrid, solvers, precision, double_single, nullspace, symmetry, paths, equilibria readers, relaxation loop/physics/newton/seeding, diagnostics poincare, optimization shape_ad), running scripts/relax.py, the tutorials, poincare_trace/poincare_plot, slurm jobs or the test suite, writing docs, docstrings or tests, and interpreting relax.json, checkpoints, iteration counts, residual floors, island widths or Poincare sections - even if the user does not say "MRX". Consult it before editing code, launching a GPU job or adding a test.
---

# MRX: working in the code base

MRX finds 3D MHD equilibria by relaxing a magnetic field toward minimum energy along
divergence-free, wall-tangent flows, without assuming nested flux surfaces. It discretises the
de Rham complex `V0 -grad-> V1 -curl-> V2 -div-> V3` with tensor-product B-splines on the logical
cube `(r, theta, zeta)`, mapped by a polar spline map `Phi`. Everything is matrix-free, JAX/equinox,
and differentiable. Preprint arXiv:2510.26986.

This skill is orientation plus the project's rules, for any contributor, human or agent. The rules
(solver strategy, code, tests, docs, outputs) are in `references/rules.md`. The skill is not the source of
truth. The source of truth is the Sphinx tree **`docs/source/`** (guides, `concepts/`, `api/`) and,
for the measurements behind design decisions, `docs/research/` (terse, dated). Read the
relevant page before changing code it describes. If the skill and the docs or code disagree, the
code and docs win: say so.

## Reference files (read the one you need, they are short)

| file | read when |
|---|---|
| `references/code-map.md` | changing or reading library code: spaces, operators, build order, data model, parity views, solvers, precision, jit rules |
| `references/running.md` | running anything: slurm, sizing, worktrees, the test suite, relax.py flags and output, Poincare, healthy-run checks, data files |
| `references/rules.md` | before writing code, tests or docs: solver strategy, code hygiene, tests, docs, outputs |
| `references/pitfalls.md` | a result looks off, or before interpreting numbers (sharp bits and measured lessons) |
| `references/history.md` | before proposing a method or option: what was settled, what was removed and why |

Paths are relative to the checkout or to the data root `$MRX_DATA`, one directory outside the repository
(for example `<root>/mrx_data` beside `<root>/mrx`), set in `~/.config/mrx/site.env`. Module and partition
names differ per cluster.

## The ten things most often gotten wrong

1. **No runs on a login node.** On a cluster, anything that imports `mrx` and builds a sequence,
   solves or traces is a GPU job through `slurm/run.sh`. Only matplotlib re-renders from archives (e.g.
   `scripts/poincare_plot.py`) and pure-numpy checks run inline.
2. **Size GPU jobs** `CPUS=4 MEM_GB=16` on small meshes, the `run.sh` default memory on larger ones,
   `TIMEOUT_MIN` at least twice the estimate (`references/running.md`).
3. **Worktree jobs must import the worktree.** `python scripts/x.py` resolves `mrx` to the main
   checkout's editable install unless `PYTHONPATH=$WT`. `slurm/run.sh` sets it from `MRX_ROOT`.
   Check the log's first line `mrx from: <path>` before reading a result.
4. **Precision is fixed at import.** `MRX_DTYPE` is `float64` (default) or `float32`, read at the first
   `import mrx`. float32 is for devices without float64: every solve then keeps its solution and residual
   in double-single. AD, shape optimisation and finite differences need float64.
5. **Signed iteration counts.** `+k` converged, `-k` did not. Counts sum over refinement passes, so
   they can exceed maxiter. A negative count is the first thing to look for.
6. **`compute_nullspaces(seq)` is a separate call** after `build_sequence`. Without it the harmonic
   forms are zero and every deflation removes nothing. Rerun it after every `set_geometry_from_map`.
   The preconditioners are rebuilt too, unless the call keeps the old ones (`keep_preconditioners=True`),
   which stay valid and are good after a small change of the map.
7. **Fields live on their parity view.** On the default stellarator-symmetric (half-period)
   sequence, `B, A, J, E` are odd (`seq.odd`), velocity, force, pressures even (`seq.even`). Norms
   and solves on the wrong view give wrong numbers silently.
8. **Angles are right-handed.** VMEC files are read with `theta -> -theta`, so their iota reads
   negative (`theta_reversed`). Never reintroduce a map sign. MRX `zeta` spans one field period:
   transform per MRX turn is `iota/nfp`. Seeds count `n` in field periods.
9. **Judge relaxation by energy and floor, not by `||F||` or end state.** The force residual is not
   monotone. Trajectories diverge from round-off (a decade per ~10 steps). Validate kernel changes
   per-apply and by energy.
10. **Production stays clean.** No aliases or back-compat shims, no defensive `abs()`/clamps/
    fallbacks over guaranteed quantities, no inner Krylov under MINRES/CG, no hidden metric factors
    (`references/rules.md`), no expensive or parametrised test without the maintainers' agreement.
    Study and paper code goes to branch `paper`, not `scripts/`.

## Git and GitHub in one paragraph

`main` is the public release and protected on GitHub: changes reach it only through pull requests. Work on a
branch (or a worktree under `.claude/worktrees/`). CI (`.github/workflows/ci.yml`) runs on every pull request
to `main`: ruff, the test suite in both precisions on a CPU, a `--help` start of every script, and the Sphinx
build with warnings as errors.

## Where things live

- Library `mrx/`, tests `test/`, shipped scripts `scripts/relax.py`, `scripts/poincare_trace.py`,
  `scripts/poincare_plot.py`, `scripts/plot_mesh.py`, `scripts/tutorials/1..6_*.py`.
- Cluster wrappers `slurm/run.sh`, `slurm/suite.sh`, `slurm/waitjob.sh`. Site settings in
  `~/.config/mrx/site.env` (or `$MRX_SITE_ENV`), outside the repo and shared by every checkout and worktree.
  Cloud TPU jobs: `gke/`, `terraform/`, `Makefile`.
- Data: the tracked files in `data/`, every other input in `$MRX_DATA/data`. `mrx.paths.data_file` finds a
  file there by name when it is not at the given path, so a worktree needs no symlink.
- Runs: the runs root `$MRX_DATA/runs/mrx` (`mrx.paths.runs_root()`) is the default of `run.sh`, `relax.py`
  and the tutorials. Each line of work (a study, a branch) gets its own `<line>/` below it. A checkout's
  `outputs/` is only the fallback without `MRX_DATA`. Write from any worktree straight there.
- Papers are not in the repository. Paper tooling is on branch `paper`.

## How to approach a task

Anchor the change to the nearest controlling code and doc page, make the smallest grounded change,
validate with the cheapest focused check (one test file, one short GPU job). Update the docs page in
the same change when behaviour changes.

## Keeping this skill current

This skill lives in this repo, `.claude/skills/general`. When a contribution changes a default or settles
a lasting rule of MRX, edit the skill in the same branch. Notes about one machine, one cluster or one
person's way of working do not belong here.
