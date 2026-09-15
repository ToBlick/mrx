# DESC → MRX interface

Status: complete on `desc-interface-v2`; `mrx/desc.py` reads a DESC `.h5`
with `h5py` alone and the validation layers below all pass.
Read it for: the measured DESC-vs-VMEC differences on li383, the GVEC
control on the same wout, the converged comparison, the sweep over DESC's
own 3-D equilibria, and two DESC-side conventions that cost time (the
poloidal flip, the lambda half-mesh bug).
Do not read it for: how the interface works — that is
`docs/source/concepts/external_interfaces.md`, which this record backs.

Written 2026-09-13 and revised 2026-09-15. Branch `desc-interface-v2` off
`static-dynamic-refactor`, worktree `/Users/aak572/mrx-desc`. Numbers
below are float64. The reader needs only `h5py`, `numpy`, `scipy`. DESC
0.17.2 was used once, to generate the fixtures; pyGVEC 1.4.1 converts the
wouts for the GVEC column.

## 1. What was built

`mrx/desc.py`, modelled on `mrx/vmec.py`: `read_desc`, `read_nfp`,
`profile_spline`. The poloidal-orientation pair
`flip_poloidal_angle` / `match_orientation` lives on the shared block
dict in `mrx/gvec.py` and is re-exported from `mrx.desc`, because a GVEC
`convert-wout` needs the same pair. The conversion is product-to-sum:
each DESC mode becomes the MRX pair `(m, +n nfp)` and `(m, -n nfp)` at
half weight. Radially, each mode's Zernike sum is sampled and refit
through `mrx.vmec._fit_block`; `read_desc` records the midpoint error as
`refit_error` and refuses an under-resolved `n_rho` above `REFIT_TOL`.

Wiring is six dispatch extensions plus `--geometry` help on
`scripts/relax.py`, `scripts/plot_mesh.py` and
`scripts/poincare_trace.py`. Figures and the VMEC/DESC/GVEC comparisons
are one script, `scripts/desc_figures.py`. The sweep over DESC's own
equilibria is `scripts/desc_example_sweep.py`.

## 2. DESC against VMEC on a common grid (`scripts/desc_figures.py --figure grid`)

Both states from the *same* wout (`data/wout_li383_low_res_reference.nc`,
nfp=3, finite beta), DESC's side via `VMECIO.load(..., profile="iota")` at
four fit resolutions. This isolates DESC's fit error plus our conversion
from every other difference. Grid `(40, 64, 32)`, relative sup norms:

| fit `L,M,N` | R | Z | lambda | lambda, `rho>0.3` | iota | `B` |
|---|---|---|---|---|---|---|
| 4,4,3   | 2.07e-3 | 8.12e-3 | 1.32e-1 | 1.30e-1 | 2.00e-3 | 1.42e-2 |
| 6,6,3   | 6.22e-4 | 2.73e-3 | 2.08e-1 | 1.44e-1 | 2.00e-3 | 9.49e-3 |
| 8,8,3   | 2.54e-4 | 1.86e-3 | 2.33e-1 | 7.84e-2 | 2.00e-3 | 7.67e-3 |
| 10,10,3 | 1.15e-4 | 9.96e-4 | 2.64e-1 | 5.59e-2 | 2.00e-3 | 7.74e-3 |

(`grid_convergence.png`).

R and Z converge cleanly toward the VMEC reference, a factor 18 and 8 over
the sweep — that is the headline, and it is what says the conversion is
right rather than merely self-consistent. iota sits at 2e-3 flat: it is a
profile, resolved at every rung, and 2e-3 is `VMECIO`'s own spline fit of
`iotaf`, independent of `L,M,N` as it should be.

The initial 2-form from each is sound on both sides: `div` at 1.4e-15,
`wall_discarded` below 2.6e-8, toroidal flux matching to 2.3e-5 relative,
force residuals 0.0564 (DESC) against 0.0553 (VMEC) at the coarsest rung
they share. `B` differs by 7.7e-3, and does not fall below that — which is
lambda's doing.

**Lambda does not converge, and the cause is in DESC.** Two mechanisms,
checked against DESC source (`desc/vmec.py:123` and
`desc/vmec_utils.py:303`, identical in checkouts `2df753d17` and
`cc87f931e`):

1. `lmns = file.variables["lmns"][:].filled()` keeps VMEC's dummy first
   row. There is no `[1:]`.
2. `fourier_to_zernike` then fits at `rho = sqrt(linspace(0, 1, surfs))`,
   the **full** mesh. `surfs` is `ns`. There is no half-mesh branch and no
   argument by which a caller could supply one.

So every `lmns` row, whose value belongs at `s_{j-1/2}`, is fit at `s_j`
(a half-cell outward shift, worst near the axis), and row 0 is fit at
`rho = 0`, pulling lambda toward zero where the `m = 0` component is not
zero. Raising `L,M,N` tracks the wrong nodes more faithfully, so the sup
error *grows* (1.3e-1 to 2.6e-1) while everything else falls. Restricting
to `rho > 0.3` recovers convergence (1.30e-1 to 5.59e-2). MRX's own wout
reader handles the half mesh correctly (`mrx.vmec._lambda_nodes`). Not
worked around; the scripts report `lambda_outer` alongside `lambda` so
the artefact cannot be mistaken for ours.

## 2b. The GVEC control (`scripts/desc_figures.py --figure gvec`)

pyGVEC's `convert-wout` (`whichInitEquilibrium=1`, `init_LA=True`) refits
the same wout interior into B-splines × Fourier. Two converter
conventions have to be undone before the labels agree: it writes Fourier
`n` with the opposite sign from VMEC's `xn` (so `R` matches at `zeta = 0`
and diverges elsewhere until every block's `n` is negated), and it sets
`phiedge = -phiedge`, which flips `iota`. Neither is a poloidal flip —
`flip_poloidal_angle` extra-negates `Z` and makes it worse.

On li383, after those two alignments, against the wout on grid
`(40, 64, 32)`:

| | R | Z | lambda | lambda, `rho>0.3` | iota |
|---|---|---|---|---|---|
| DESC `L=M=8` | 2.54e-4 | 1.86e-3 | 2.33e-1 | 7.84e-2 | 2.00e-3 |
| GVEC 10 els | 2.47e-4 | 6.64e-4 | **2.63e-2** | **6.07e-3** | 2.78e-4 |
| GVEC 40 els | 2.46e-4 | 6.59e-4 | 2.56e-2 | 7.24e-3 | 1.17e-4 |

QA vacuum, same grid: DESC lambda at 2.06e-2 against GVEC 10 at 2.39e-3
(`gvec_grid.png`).

GVEC matches DESC on `R`, beats it on `Z`, and beats it on lambda by an
order of magnitude — including near the axis, where DESC's half-mesh
misfit lives. That is the control: a second code refitting the same wout,
with a radial basis that has no half-mesh confusion, lands where the
source reading said it should.

## 3. Converged from a VMEC versus a DESC initial condition (`scripts/desc_figures.py --figure relax`)

One sequence `(8, 12, 12)` p=2, one `TimeStepper`, two initial fields,
5000 steps each, li383.

| | initial | final |
|---|---|---|
| `\|B_d - B_v\|_M / \|B_v\|_M` | 7.67e-3 | **3.45e-4** |

(`desc_vmec_relax.png`). The difference **shrinks by 22×** through
relaxation — amplification 0.045.
MRX relaxation is a helicity-preserving energy minimisation, so two
initial conditions carrying different helicity need not reach the same
state, and the honest assertion is only that the discrepancy is not
amplified. It is in fact strongly contracted: the two initial fields
differ by 7.67e-3 while their helicities differ by 5.70e-5, and the
converged states settle a factor 22 closer. At 1000 steps the same pair
was 6.79e-4 (11×); the extra budget keeps contracting.

Both runs are well-behaved and agree on the physics: energy monotone
throughout, force residual 5.53e-2 to 3.98e-4 (VMEC) and 5.64e-2 to 3.79e-4
(DESC), helicity drift 8.9e-7 and 1.8e-6 of `2 E_0`, final energies within
4.6e-5 relative, `beta_vol` 0.04229 against 0.04220 (2.1e-3 relative).

The per-step traces used to look unphysically smooth: `BLOCK = 100` on a
1000-step run drew ten points, and the lower panel plotted the cumulative
`E_0 - E`, which is monotone by construction. The traces themselves were
always per-step (`mrx.relaxation` records `dE` and `F_norm` every step).
The figure now uses an adaptive block width (`~200` points), a raw
per-step underlay, and the per-step release rate `-dE`
(`desc_traces.png`).

The Poincaré sections make the 22× contraction visible
(`poincare_li383.png`). All four panels are nested, the two initial
conditions already agree at the eye, and relaxation does not open an
island or scramble the surfaces.

## 4. The poloidal flip, which is the trap

`VMECIO.load` runs `ensure_positive_jacobian`, which for a left-handed wout
— li383 among them — negates theta: `m < 0` modes of R and Z, `m >= 0`
modes of lambda, and iota. The first grid run reported an iota difference
of exactly 2.000 relative, i.e. a clean sign flip, which is what exposed it.

MRX does not need this undone to *read* a file: `build_gvec_map` measures
the handedness giving `det DF > 0`. But comparing at equal theta compares
different points. `match_orientation` measures the relative orientation and
`flip_poloidal_angle` undoes it in the block representation. Every DESC
run above reports `orientation = -1`, and after matching, iota agrees to
2e-3 rather than differing by a factor 2.

One subtlety cost a test: orientation must be measured from **R and Z
together**. On an up-down-symmetric cross-section R alone is blind to the
flip, and the synthetic circular torus is exactly that case.

## 5. Sweep over DESC's own saved equilibria (`scripts/desc_example_sweep.py`)

Relaxing MRX from DESC's shipped `_output.h5` files directly, `(8, 14, 8)`
p=2, 2000 steps on an H200. Axisymmetric SOLOVEV and DSHAPE are dropped:
they are 2-D. The three remaining iota-constrained cases are genuinely
3-D. The current-constrained ones (NCSX, ARIES-CS, precise_QA, HSX,
WISTELL-A) need DESC for iota and have not been run.

| case | nfp | vac | iota axis file/MRX | div | F in → out | dH/2E0 | reader | relax |
|---|---|---|---|---|---|---|---|---|
| ATF | 12 | n | 0.35000/0.35000 | 1.3e-15 | 1.50e-4 → 1.47e-6 | 3.69e-5 | ok | ok |
| HELIOTRON | 19 | n | 1.00000/1.00001 | 4.3e-16 | 8.19e-5 → 7.64e-6 | 1.36e-2 | ok | ok |
| W7-X | 5 | n | 0.85605/0.85605 | 3.6e-16 | 5.69e-3 → 1.76e-8 | 3.68e-6 | ok | ok |

(`desc_sweep.png`). **The reader checks pass on all three** — iota matched
to the file's own
profile at both axis and edge to 5-6 digits, `div` at round-off,
`wall_discarded` negligible, initial field finite. That includes nfp=19
(HELIOTRON) and nfp=12 (ATF), which stress the toroidal mode mapping
hardest, and W7-X with `Psi = -2.133`, which exercises the negative-flux
branch.

At 500 steps HELIOTRON's residual *rose* (5.61e-5 → 3.94e-4): a mid-run
spike had not decayed when `stop=steps` ended the run, and the old
`relaxation` dict never asked whether the residual fell. 2000 steps
brings it down 11× (8.19e-5 → 7.64e-6). The new `residual_decreased`
check would have flagged the short run; ATF and W7-X already floored
inside 500.

`--figure cases` draws an initial-versus-relaxed Poincaré section and a
mesh (`scripts/plot_mesh.py`) for HELIOTRON, W7-X, ATF and Landreman–Paul
QA (`poincare_case_{heliotron,w7x,atf,qa}.png`,
`mesh_{heliotron,w7x,atf,qa}.png`). 5000 steps on `(8, 12, 12)` p=2;
every case kept 64/64 lines.

Landreman–Paul QA is also the vacuum check against VMEC: both start and
finish on the same nested surfaces (`poincare_qa.png`).

## 6. Tests

`test/test_desc.py` plus the GVEC orientation pair in
`test/test_readers.py`. The two that carry the most weight:

- `test/synthetic_desc.py` writes a closed-form DESC-layout file that
  inverts `read_desc`, and describes the **same torus** as
  `test/synthetic_gvec.py`, so the two readers are held against each other
  through independent parsers of independent formats.
- the tracked fixtures (`data/desc_HELIOTRON_lowres.h5`, the external
  DESC example, and `data/desc_QA_lowres.h5`) are checked against their
  Fourier-Zernike series rebuilt in DESC's own product form, independent
  of the conversion under test.

Plus product-to-sum on a two-mode field, Zernike radials against explicit
low-order polynomials, negative `Psi`, the `_sym = False` and current-
constrained guards, dispatch through `geometry_kind` / `geometry_nfp` /
`initial_field` on HELIOTRON (nfp=19), `flip_poloidal_angle` as an
involution, the starved-`n_rho` refusal, and `match_orientation` on a
synthetic GVEC torus.

## 7. Verification

`ruff check . --ignore F403,F405` clean. Locally (`mrx` conda env, no
DESC): 80 tests pass, the three DESC-gated tests skip. On Torch in the
`mrx.ext3` overlay with DESC 0.17.2, one H200 (`h200_courant` /
`torch_pr_292_courant`): 83 tests pass, including the live `eq.compute`
cross-check and the current-constrained iota path; `mrx.desc` is at
100% coverage. Figures regenerate from cache with
`python -u scripts/desc_figures.py --figure all`.

Two things worth knowing for anyone repeating this:

- **pytest is not in the overlay** and the overlay is read-only; it was
  installed with `pip install --target /scratch/aak572/pytest-libs` and put
  on `PYTHONPATH`, leaving the overlay untouched.
- **The overlay's jax 0.8.1 cannot take `batch_size=0`.** MRX's default
  `MAP_BATCH_SIZE_INNER = 0` means "one `vmap`". That is a documented
  jax contract as of [jax#33965](https://github.com/jax-ml/jax/pull/33965),
  first released in **0.8.2** (2025-12-18); `pyproject.toml` pins
  `jax>=0.8.2`. The overlay predates the fix (0.8.1, 2025-11-18) and is
  read-only, so the Torch runner still sets `MAP_BATCH_SIZE_INNER = 4096`
  from outside the repo. This is a version floor, not an MRX bug.

The suite must run on a **compute node**: on the login node the sequence
build in the dispatch test had not finished after 50 minutes, against
tens of seconds on an H200.

## 8. Open

- **Current-constrained files are Torch-only.** Structural: iota is not in
  the file. The DESC-backed path works; CI cannot exercise it.
- **DESC's wout lambda half-mesh bug (§2).** Recorded, not worked around,
  and now backed by a GVEC control. Worth a DESC issue; it makes any
  lambda comparison through `VMECIO.load` misleading near the axis for
  everyone, not just MRX.
- **The current-constrained cases have not been run.** The iota-constrained
  trio is done at `(8, 14, 8)` p=2, 2000 steps on an H200. NCSX,
  ARIES-CS, precise_QA, HSX and WISTELL-A still need DESC for iota.
