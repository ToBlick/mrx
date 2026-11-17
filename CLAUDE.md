# MRX

Matrix-free FEEC 3D MHD equilibrium solver in JAX. Docs in `docs/source/`, measurements behind
design decisions in `docs/research/`.

## Skills

The MRX working knowledge is the skill `general` in `.claude/skills/` (code map, running, rules,
pitfalls, settled decisions, the project's rules in `references/rules.md`). Load it before any task
here. A contributor's personal preferences belong in their own user-level skills, not in this repo.

## Layout

Nothing but code lives in the checkout. Next to it is the data root, named by `MRX_DATA`
(`docs/source/cluster.md`):

```
<root>/mrx/                   this repository, worktrees in .claude/worktrees/<name>
<root>/mrx_data/              $MRX_DATA
  data/                       input files not tracked in git, found by name
  runs/mrx/<line>/            run directories and job logs, one <line> per line of work
  cache/xla/                  test-suite compilation cache
```

- Site settings (account, partition, `MRX_DATA`) are in `~/.config/mrx/site.env`, never in the repo.
  Source it in any shell that runs MRX by hand. On a new cluster, create it first.
- Write runs, figures, records and scratch files under `$MRX_DATA/runs/mrx/<line>/`, never into the
  checkout or a worktree.

## Rules that bite

- On a cluster, nothing that imports `mrx` runs on a login node. Jobs go through `slurm/run.sh`.
  From a worktree set `MRX_ROOT` and check the log's `mrx from:` line.
- `main` is protected on GitHub. Work on a branch, changes reach `main` by pull request.
