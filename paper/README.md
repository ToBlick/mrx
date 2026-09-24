# Paper scripts

The runs, figures and tables of the JCP paper (MHD equilibria without nested flux surfaces), built on the
command-line tools (`scripts/relax.py`, `scripts/poincare_trace.py`, `scripts/poincare_plot.py` and the vacuum
scripts `analytic_vacuum.py`, `qa_vacuum_sweep.py`, `plot_vacuum_qa.py`) and on their run records: `relax.json` and
`checkpoints/` per run, continuations in `cont/`, a trace archive `trace.npz` next to the run it traces.
They follow the API of this branch and will need updating when it changes.

## How to run

1. GPU runs. From the repository root, `bash paper/runs/<experiment>.sh`, one slurm job per run through
   `slurm/run.sh` (site settings as there), each launcher chaining its restarts and traces by slurm dependencies.
   `RECORDS=<root>` puts the records under `<root>/<experiment>/<arm>` [outputs]. `vacuum`, `newton_convergence`,
   `gradient` and `newton_sweeps` are independent; `seeding` and `reconnection` start from
   `newton_convergence/newton_32`.
2. Traces: part of the launchers, a `poincare_trace.py` job after the run it traces.
3. Plotting on the login node, matplotlib only, a TeX Live `bin` directory on `PATH` for the PGF (xelatex for
   `figures.py`, pdflatex for the other two):

       python paper/figures.py --records <root> --out paper/build
       python paper/poincare_pages.py --records <root> --out paper/build
       python paper/energy_landscape.py --out paper/build

   `paper/build` then holds the paper's layout, `tables/<table>.tex` and `figs/pgf/<figure>/`, to copy into the
   paper; `figs/vacuum_qa_Bmag.pdf` is `<root>/vacuum_vmec/bmag/vacuum_qa_Bmag.pdf`. `figures.py` also prints the
   numbers the text quotes from each item; `--only` picks items.

## Paper items

| item | launcher | records under `<root>` | generator |
|---|---|---|---|
| Tab. 1 codes | -- | -- | hand-written |
| Tab. 2 `symmetry_vacuum_table_jcp` | `vacuum.sh` | `vacuum_symmetry/{torus,period,half}_{26,34}` | `figures.py symmetry_vacuum` |
| Fig. 3 `vacuum_qa_Bmag.pdf` | `vacuum.sh` (`plot_vacuum_qa.py`) | `vacuum_vmec/bmag` | the job itself |
| Fig. 4 `vacuum_convergence_analytic` | `vacuum.sh` | `vacuum_analytic/p*` | `figures.py vacuum_convergence` |
| Fig. 5 `vacuum_convergence_vmec` | `vacuum.sh` | `vacuum_vmec/{lowres,highres}/rung_*` | `figures.py vacuum_convergence` |
| Fig. 6 `vacuum_qa_poincare` | `vacuum.sh` | `vacuum_vmec/highres/rung_32x64x32_p3/run` | `poincare_pages.py` |
| Fig. 7 `energy_landscape` | -- | -- | `energy_landscape.py` |
| Fig. 8 `lbfgs_smoothing` | `gradient.sh`, `newton_convergence.sh` | `gradient/gamma0`, `newton_convergence/gradient_16` | `figures.py smoothing` |
| Sec. 5.3, the smoothing constant | `gradient.sh` | `gradient/smoothing_c*`, `newton_convergence/gradient_16` | `figures.py smoothing_constant` |
| Figs. 9, 10, Tab. 3 `newton_convergence_table_jcp` | `newton_convergence.sh` | `newton_convergence/{gradient_16,newton_*}` | `figures.py newton_convergence` |
| Fig. 11 `islands_equilibrium48` | `newton_convergence.sh` | `newton_convergence/newton_48` | `poincare_pages.py` |
| Fig. 12 `islands_seeded32` | `seeding.sh` | `seeding/relax32` | `poincare_pages.py` |
| Tab. 4 `seed_selection_table`, Fig. 13 `seed_trace` | `seeding.sh` | `seeding/{seeded.json,relax32}`, `newton_convergence/newton_32` | `figures.py seeding` |
| Tab. 5 `reconnection_demo_table` | `reconnection.sh` | `reconnection/{unseeded,s51q,s61}/{ideal,resistive,ideal_after}`, `newton_convergence/newton_32` | `figures.py reconnection` |
| Figs. 14, 15 `reconnection_*_{before,after}` | `reconnection.sh`, `newton_convergence.sh` | as Tab. 5 | `poincare_pages.py` |
| Tab. C.6 `helicity_table` | `gradient.sh` | `gradient/helicity_*` | `figures.py helicity` |
| Tab. C.7 `velocity_table` | `gradient.sh`, `newton_convergence.sh` | `gradient/leray`, `newton_convergence/gradient_16` | `figures.py velocity` |
| Figs. C.16 `newton_precision`, C.17 `newton_degree` | `newton_sweeps.sh`, `newton_convergence.sh` | `newton_sweeps/{p*,float*,tol*}`, `newton_convergence/newton_16` | `figures.py newton_order_precision` |
| Tabs. C.8 `newton_penalty_table`, C.9 `newton_inner_table` | `newton_sweeps.sh` | `newton_sweeps/{kappa*,minres*}` | `figures.py newton_sweeps` |
| Fig. C.18 `newton_poincare_n{16,24,32}` | `newton_convergence.sh` | `newton_convergence/newton_{16,24,32}/cont` | `poincare_pages.py` |
| Tabs. D.1, D.2 | -- | -- | hand-written |

Figs. 1 and 2 and the graphical abstract are illustrations this directory does not rebuild. Fig. 1 (`mesh_3d.pdf`,
`figure1_standalone.pdf`) came from `outputs/figures_2026-09/mesh_sections_3d.py` on the sections of an earlier
reconnection run, with the colour bars of `figure1_standalone.tex`; the command behind Fig. 2 (`mesh_2d.pdf`) is not
recorded; `figs/graphical_abstract.pdf` is a copy of `mesh_3d.pdf`. The automatic-differentiation demonstration
lives on the branch `ad-shape-optimization`.

## Notes

- `seed.py` is the one script here on mrx internals: no command-line tool seeds a checkpoint. It writes the seeded
  checkpoint and, beside it, the table of every resonance that Tab. 4 reads. `vacuum_run.py` wraps the stored vacuum
  field of a `qa_vacuum_sweep.py` rung as a run of zero steps, because `poincare_trace.py` traces runs only.
- The first step after a restart spikes (a poorer Newton direction) and is gone the step after; the plotted trace of
  a continued run carries the step before in its place (Tobias 2026-09-19).
- The DOF counts of Tab. 3 are one stellarator parity class of the Dirichlet 2-forms, in closed form for p = 2;
  Tab. 2 halves the field period's count for the half-period model.
- Changes from the runs as first made: `gradient_16` passes its smoothing scale, the default of its day
  (0.02 / n_r^2; today 0.075 <g_rr> h_r^2); Fig. 4 runs every (p, n) once on today's code and plots all of them;
  Fig. 5's two floors are the mean over p at the finest mesh; Fig. 6's trace goes through `vacuum_run.py`
  (`poincare_trace.py --field-npz` went in a2f3c6b); Tab. 4's widths use the pendulum estimate as the paper states
  it, sqrt(8/pi) where `seed_final.py` had 1.6; the three pages of Fig. C.18 share the iota and pressure scale of the
  three states, and their page is `poincare_zeta0.5` (one field per archive).
- The largest vacuum rungs of Figs. 4 and 5 ran with `MRX_MAP_BATCH_SIZE_INNER` set, which the vacuum scripts no
  longer read; they may run out of GPU memory. `data/wout_LandremanPaul2021_QA_highres.nc` is not in the repository.
