# Paper scripts

Every run, figure and table of the JCP paper (MHD equilibria without nested flux surfaces), built on the
command-line tools of `scripts/` (`relax.py`, `poincare_trace.py`, `poincare_plot.py`, `plot_mesh.py`) and on
their run records: `relax.json` and `checkpoints/` per run, continuations in `cont/`, a trace archive `trace.npz`
next to the run it traces. The vacuum study scripts (`analytic_vacuum.py`, `qa_vacuum_sweep.py`,
`plot_vacuum_qa.py`) live here too: only the paper runs them. Everything follows the API of this branch and
needs updating when it changes.

## How to run

1. GPU runs. From the repository root, `bash scripts/paper_scripts/runs/<experiment>.sh`, one slurm job per run
   through `slurm/run.sh` (site settings as there), each launcher chaining its restarts and traces by slurm
   dependencies. The arms that use options `scripts/relax.py` no longer exposes (the velocity smoothing
   order and scale, the helicity correction, the potential velocity, the auxiliary B field: `gradient.sh`
   and `newton_convergence.sh`'s gradient arm) run `relax_paper.py`, the same run on an extended
   configuration; every other arm runs `scripts/relax.py`. `RECORDS=<root>` puts the records under `<root>/<experiment>/<arm>` [outputs]. `vacuum`,
   `newton_convergence`, `gradient`, `newton_sweeps` and the `ad_*` launchers are independent; `seeding` and
   `reconnection` start from `newton_convergence/newton_32`; `ad_trace` needs `ad_constrained`'s draw 0.
2. Traces: part of the launchers, a `poincare_trace.py` job after the run it traces.
3. Plotting on the login node, matplotlib only, a TeX Live `bin` directory on `PATH` for the PGF (xelatex for
   `figures.py` and `ad_figures.py`, pdflatex for the Poincaré pages and the landscape):

       python scripts/paper_scripts/figures.py --records <root> --out scripts/paper_scripts/build
       python scripts/paper_scripts/poincare_pages.py --records <root> --out scripts/paper_scripts/build
       python scripts/paper_scripts/energy_landscape.py --out scripts/paper_scripts/build
       python scripts/paper_scripts/ad_figures.py --records <root> --out scripts/paper_scripts/build
       python scripts/paper_scripts/ad_poincare_page.py --records <root> --out scripts/paper_scripts/build
       python scripts/paper_scripts/logical_section.py <root>/seeding/relax32/trace.npz --plane 0.5 ...

   `build/` (gitignored) then holds the paper's layout, `tables/<table>.tex` and `figs/pgf/<figure>/`, to copy
   into the paper; `figs/vacuum_qa_Bmag.pdf` is `<root>/vacuum_vmec/bmag/vacuum_qa_Bmag.pdf`. `figures.py` also
   prints the numbers the text quotes from each item; `--only` picks items.

## Paper items (numbering of mrx_jcp.tex, 2026-09-24; labels in parentheses)

| item | launcher | records under `<root>` | generator |
|---|---|---|---|
| Fig. 1 `figure1_standalone.pdf` (`fig:ncsx_sections`) | `seeding.sh` | `seeding/relax32` (`relax.json`, `trace.npz`) | `figure1_mesh3d.py` (GPU: the run's map) |
| Tab. 1 `mhs_codes_table` (`tab:codes`) | -- | -- | hand-written |
| Fig. 2 `mesh_2d.pdf` (`fig:mesh_refinement`) | -- | -- | `plot_mesh_paper.py` (several meshes side by side, with sections and optimized boundaries; command not recorded) |
| Fig. 3 `vacuum_qa_Bmag.pdf` (`fig:vacuum_qa_bmag`) | `vacuum.sh` (`plot_vacuum_qa.py`) | `vacuum_vmec/bmag` | the job itself |
| Fig. 4 `vacuum_convergence_analytic` (`fig:analytic_vacuum`) | `vacuum.sh` (`analytic_vacuum.py`) | `vacuum_analytic/p*` | `figures.py vacuum_convergence` |
| Fig. 5 `vacuum_convergence_vmec` (`fig:vmec_vacuum`) | `vacuum.sh` (`qa_vacuum_sweep.py`) | `vacuum_vmec/{lowres,highres}/rung_*` | `figures.py vacuum_convergence` |
| Tab. 2 `symmetry_vacuum_table_jcp` (`tab:symmetry_vacuum`) | `vacuum.sh` | `vacuum_symmetry/{torus,period,half}_*` | `figures.py symmetry_vacuum` (the n = 48 rows; Tobias 2026-09-23) |
| Figs. 6, 7 `qa_paper_qa`, `qa_paper_shape` (`fig:shape_optimization`, `fig:shape_errors`) | `ad_constrained.sh`, `ad_baseline.sh`; `ad_baselines.py` (numpy, login node) | `shape_optimization/qa_{constrained,recover,guard}_*`, `qa_baseline_*`, `qa_boundary_baselines.json` | `ad_figures.py` |
| Fig. 8 `qa_poincare_optimized` (`fig:vacuum_qa_poincare`) | `ad_trace.sh` | `shape_optimization/qa_trace_M10s0_remesh.npz` | `ad_poincare_page.py` |
| Fig. 9 `energy_landscape` (`fig:landscape`) | -- | -- | `energy_landscape.py` |
| Fig. 10 `lbfgs_smoothing` (`fig:gamma_sweep`) | `gradient.sh`, `newton_convergence.sh` | `gradient/gamma0`, `newton_convergence/gradient_16` | `figures.py smoothing` |
| Sec. 5.3, the smoothing constant | `gradient.sh` | `gradient/smoothing_c*`, `newton_convergence/gradient_16` | `figures.py smoothing_constant` |
| Figs. 11, 12 `newton_vs_lbfgs`, `newton_resolution`, Tab. 3 `newton_convergence_table_jcp` (`fig:newton`, `fig:newton_h_sweep`, `tab:newton_convergence`) | `newton_convergence.sh` | `newton_convergence/{gradient_16,newton_*}` | `figures.py newton_convergence` |
| Fig. 13 `islands_equilibrium48` (`fig:islands_equilibrium`) | `newton_convergence.sh` | `newton_convergence/newton_48` | `poincare_pages.py` |
| Tab. 4 `seed_selection_table`, Fig. 14 `seed_trace` (`tab:seed_selection`, `fig:seed_trace`) | `seeding.sh` (`seed.py`) | `seeding/{seeded.json,relax32}`, `newton_convergence/newton_32` | `figures.py seeding` |
| Fig. 15 `islands_seeded32` (`fig:islands_seeded32`) | `seeding.sh` | `seeding/relax32` | `poincare_pages.py` |
| Fig. 16 `islands_seeded32_logical` (`fig:islands_seeded32_logical`) | `seeding.sh` | `seeding/relax32/trace.npz` | `logical_section.py` |
| Tab. 5 `reconnection_demo_table` (`tab:reconnection_demo`) | `reconnection.sh` | `reconnection/{unseeded,s51q,s61}/{ideal,resistive,ideal_after}`, `newton_convergence/newton_32` | `figures.py reconnection` |
| Figs. 17, 18 `reconnection_*_{before,after}` (`fig:reconnection_before`, `fig:reconnection_after`) | `reconnection.sh`, `newton_convergence.sh` | as Tab. 5 | `poincare_pages.py` |
| App. A `notation_table` | -- | -- | hand-written |
| App. `landreman_convergence_table` (`tab:landreman_convergence`) | `landreman_chain.py` / `landreman_verify.py` part A | `landreman_verify_*/partA_sheared/rung_*/result.json` | hand-assembled from the records (header of the table) |
| App. `helicity_table` (`tab:helicity_preservation`) | `gradient.sh` | `gradient/helicity_*` | `figures.py helicity` |
| App. `velocity_table` (`tab:velocity_choices`) | `gradient.sh`, `newton_convergence.sh` | `gradient/leray`, `newton_convergence/gradient_16` | `figures.py velocity` |
| App. `newton_precision`, `newton_degree` (`fig:newton_tol_sweep`, `fig:newton_p_sweep`) | `newton_sweeps.sh`, `newton_convergence.sh` | `newton_sweeps/{p*,float*,tol*}`, `newton_convergence/newton_16` | `figures.py newton_order_precision` |
| App. `newton_penalty_table`, `newton_inner_table` (`tab:newton_penalty`, `tab:newton_inner`) | `newton_sweeps.sh` | `newton_sweeps/{kappa*,minres*}` | `figures.py newton_sweeps` |
| App. `newton_poincare_n{16,24,32}` (`fig:newton_sections`) | `newton_convergence.sh` | `newton_convergence/newton_{16,24,32}/cont` | `poincare_pages.py` |
| App. `appendix_run_parameters_jcp` (`tab:runs`), `hyperparameters_table` (`tab:hyperparameters`) | -- | the `relax.json` of every run above | hand-written, checked against the records 2026-09-22 |

Fig. 1 is `figure1_mesh3d.py` (2026-09-24): the run's 3-D mesh cut open at two traced planes with the seeded
32^3 sections on the cut faces, every kept line, the pressure on the Poincaré pages' scales
(`--pressure-factor` as `poincare_pages.py`); the colour bars of `figure1_standalone.tex` frame it;
`figs/graphical_abstract.pdf` is a copy of the earlier `mesh_3d.pdf`. The command behind Fig. 2 is not
recorded. The Landreman appendix scripts (`landreman_*.py`) are the verification behind
`tab:landreman_convergence` and the Landreman island runs; `landreman_chain.py` runs `<script> :: relax.py ... ::
poincare_trace.py ...` groups as one job, resolving siblings here and the tools in `scripts/`.

## Notes

- `seed.py` and `landreman_seed.py` are the scripts here on mrx internals: no command-line tool seeds a
  checkpoint. `seed.py` writes the seeded checkpoint and, beside it, the table of every resonance that Tab. 4
  reads. `vacuum_run.py` wraps the stored vacuum field of a `qa_vacuum_sweep.py` rung as a run of zero steps,
  because `poincare_trace.py` traces runs only.
- The first step after a restart spikes (a poorer Newton direction) and is gone the step after; the plotted trace
  of a continued run carries the step before in its place (Tobias 2026-09-19).
- The DOF counts of Tab. 3 are one stellarator parity class of the Dirichlet 2-forms, in closed form for p = 2;
  Tab. 2 halves the field period's count for the half-period model.
- Changes from the runs as first made: `gradient_16` passes its smoothing scale, the default of its day
  (0.02 / n_r^2; today 0.075 <g_rr> h_r^2); Fig. 4 runs every (p, n) once on today's code and plots all of them;
  Fig. 5's two floors are the mean over p at the finest mesh; Tab. 4's widths use the pendulum estimate as the
  paper states it, sqrt(8/pi) where `seed_final.py` had 1.6; the three pages of `fig:newton_sections` share the
  iota and pressure scale of the three states, and their page is `poincare_zeta0.5` (one field per archive).
- Fig. 8 regenerated with today's `mrx/plotting.py` differs slightly in layout from the committed page (colour-bar
  gap, r ticks every 0.25): the committed one used the plotter of the day, like the other Poincaré pages.
- The largest vacuum rungs of Figs. 4 and 5 ran with `MRX_MAP_BATCH_SIZE_INNER` set, which the vacuum scripts no
  longer read; they may run out of GPU memory. `data/wout_LandremanPaul2021_QA_highres.nc` is not in the
  repository. The `vacuum_qa_poincare` page of `poincare_pages.py` (the unoptimized QA vacuum field) is no longer
  in the paper; Fig. 8 replaced it.
