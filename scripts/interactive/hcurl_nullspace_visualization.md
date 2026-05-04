# Hcurl Nullspace Visualization

This note explains how to use [hcurl_nullspace_visualization.py](/scratch/tblickhan/mrx/scripts/interactive/hcurl_nullspace_visualization.py) and what each stage in the script does.

## Purpose

The script uses the analytical rotating ellipse geometry directly, computes the free `k=1` harmonic field on that geometry, and visualizes the result in three ways:

- logical-domain poloidal slices,
- physical `R-Z` cross-sections over the full torus with quiver overlays,
- a 3D full-torus outer-surface plot colored by `|u|`.

It also prints the first three eigenvalues of the dense `k=1` Schur complement

$$
S_1 = K_1 + D_0 M_0^{-1} D_0^T,
$$

using `np.linalg.eigh` as a dense debugging sanity check.

## How To Run It

Open [hcurl_nullspace_visualization.py](/scratch/tblickhan/mrx/scripts/interactive/hcurl_nullspace_visualization.py) in the editor and run it cell by cell.

Recommended order:

1. Run the imports and `Config` cell.
2. Run `build_case()`.
3. Run `assemble_case()`.
4. Run `compute_hcurl_nullspace()`.
5. Run `plot_logical_cross_section()`.
6. Run `plot_poloidal_cross_section()`.
7. Run `plot_surface_magnitude()`.

The bottom of the script already calls those functions in that order, so running the whole file also works. The staged route is better for debugging because each step leaves objects cached in the module globals.

## What Each Stage Does

### Stage 1: `build_case()`

This creates the `DeRhamSequence`, evaluates the 1D basis data, builds the analytical rotating ellipse map, and attaches that geometry directly to the sequence.

Concretely it does two things:

1. constructs the logical spline spaces,
2. calls `seq.set_map(rotating_ellipse_map(...))`.

After this stage:

- `SEQ` is the sequence,
- `ANALYTIC_MAP` is the one-field-period rotating-ellipse map used for assembly,
- `FULL_ANALYTIC_MAP` is the full-torus plotting map built with `extend_map_nfp(...)`.

### Stage 2: `assemble_case()`

This assembles the operator slice needed by the single-vector `k=1` nullspace routine and by the dense Schur-complement sanity check.

The script currently assembles:

- mass operators for `k = 0, 1, 2`,
- the tensor mass preconditioner for `k = 0, 1`,
- incidence operators for `k = 0, 1`,
- derivative operators for `k = 0, 1`,
- Hodge operators for `k = 0, 1`.

That is the minimal slice needed for the free `k=1` inverse iteration without going through the broader assemble-all path.

After this stage:

- `OPERATORS` holds the assembled `SequenceOperators` bundle,
- `SEQ.operators` points at the same bundle.

### Stage 3: `compute_hcurl_nullspace()`

This stage does two separate things.

First, it forms the dense `k=1` Schur complement

$$
S_1 = K_1 + D_0 M_0^{-1} D_0^T
$$

and prints its first three eigenvalues.

This is only a debugging check.

Second, it calls the library single-vector nullspace routine

`seq._find_nullspace_vectors(1, 1, config.nullspace_eps, dirichlet=False)`

to compute exactly one free `k=1` harmonic vector.

The resulting coefficient vector is stored in:

- `NULL_VECTOR` as the discrete 1-form DoF vector,
- `NULL_FIELD` as the one-field-period pushed-forward physical field,
- `FULL_NULL_FIELD` as the full-torus pushed-forward physical field built from a localized logical-field wrapper and `extend_map_nfp(...)`.

The script also computes the smallest dense Schur-complement eigenvector and checks that it approximately matches the library nullspace vector in the `M1` norm, up to sign.

The printed `nullspace info` is the returned convergence report from the iterative nullspace solve.

### Stage 4: `plot_logical_cross_section()`

This evaluates the discrete 1-form field directly on logical-domain poloidal slices.

The script plots one panel for each entry in `zeta_cuts`.

Because the physical full-torus cuts live on the extended map while the discrete field lives on one field period, the markdown titles show both:

- `\zeta_{\mathrm{full}}`, the full-device toroidal cut,
- `\zeta_{\mathrm{loc}}`, the localized toroidal coordinate inside one field period.

The background is `|u|` in logical coordinates, and the quiver overlay uses the logical `(r, \theta)` components of the discrete 1-form.

### Stage 5: `plot_poloidal_cross_section()`

This evaluates the full-torus pushed-forward field on the same family of cuts and plots one physical `R-Z` cross-section per `zeta` value.

The key detail is that the plotting map is not `ANALYTIC_MAP` but `FULL_ANALYTIC_MAP = extend_map_nfp(ANALYTIC_MAP, rotating_nfp)`, and the pushed-forward field uses a localized logical-field wrapper so each full-device `zeta` value is mapped back to the correct local field-period coordinate before evaluation.

The background is `|u|` and the quiver overlay uses:

- `v_R` from the Cartesian `x-y` components,
- `v_Z` from the Cartesian `z` component.

### Stage 6: `plot_surface_magnitude()`

This evaluates the same full-torus physical field on the outer surface `r = 1` and plots the full torus surface in 3D, colored by `|u|`.

This is useful once the 2D cuts look reasonable and you want to check whether the magnitude is smooth over all field periods.

## Main Configuration Knobs

The `Config` dataclass at the top of the script controls the resolution and plotting choices.

Most important fields:

- `ns`: number of elements in each logical direction.
- `p`: spline degree.
- `nullspace_eps`: shift used by the one-form nullspace inverse iteration.
- `zeta_cuts`: tuple of full-device toroidal cut locations for the 2D slice figures.
- `cut_nx`, `cut_ny`: resolution of the cross-section grid.
- `surface_ntheta`, `surface_nzeta`: surface grid resolution for the 3D plot.
- `quiver_stride_r`, `quiver_stride_t`: downsampling for the quiver arrows.

The rotating-ellipse geometry itself is controlled by:

- `rotating_eps`,
- `rotating_kappa`,
- `rotating_r0`,
- `rotating_nfp`.

## Module Globals

The script is intentionally stateful so you can debug it interactively without rebuilding everything every time.

The main globals are:

- `SEQ`: the sequence object,
- `OPERATORS`: the current operator bundle,
- `ANALYTIC_MAP`: the one-field-period analytic rotating ellipse map,
- `FULL_ANALYTIC_MAP`: the full-torus map obtained from `extend_map_nfp(...)`,
- `NULL_VECTOR`: the nullspace DoF vector,
- `NULL_FIELD`: the one-field-period pushed-forward field,
- `FULL_NULL_FIELD`: the full-torus pushed-forward field,
- `DENSE_EIGENVALUES`: the three printed dense Schur-complement eigenvalues.

If one stage is rerun, later cached data may become stale. In practice the safe order is always:

1. `build_case()`
2. `assemble_case()`
3. `compute_hcurl_nullspace()`
4. plotting

## Typical Debug Workflow

If you are debugging geometry or resolution effects:

1. Change `ns` or `p`.
2. Rerun `build_case()`.
3. Rerun `assemble_case()`.
4. Rerun `compute_hcurl_nullspace()`.

If you are only changing the slice or plotting density:

1. Update `zeta_cuts`, `cut_nx`, `cut_ny`, `surface_ntheta`, `surface_nzeta`, or quiver strides.
2. Rerun only the plotting functions.

If you are checking the nullspace solver itself:

1. Change `nullspace_eps`.
2. Rerun `compute_hcurl_nullspace()`.
3. Compare the printed eigenvalues and inverse-iteration info.

## Expected Outputs

When the script works as intended, you should see:

- a line reporting the built analytical-map case,
- a line reporting the assembled operator slice,
- the first three dense eigenvalues of the `k=1` Schur complement,
- an `M1`-norm mismatch check between the dense and library nullspace vectors,
- a nullspace convergence report for the library routine,
- one multi-panel logical-domain slice figure,
- one multi-panel physical full-torus slice figure,
- one 3D full-torus outer-surface figure.

For a solid torus setup with `betti = (1, 1, 0, 0)`, the free 1-form problem should have one harmonic mode, so the smallest generalized eigenvalue should be close to zero.