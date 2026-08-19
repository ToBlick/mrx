"""
NFS tensor-grid map interpolation (``interpolate_map_from_points``).

This module exists so drivers such as ``relax_from_nfs`` can import
``interpolate_map_from_points`` even when a partial / stale ``mrx/io.py`` on a
cluster omits it. Also used for the harmonic solve when only an h5 is available as opposed to a GVEC folder.
"""

from __future__ import annotations

from typing import Callable, Tuple

import numpy.typing as npt

import jax
import jax.numpy as jnp
import numpy as np

# Dense Cholesky/solve fallback for 0-form L² projection when CG returns NaNs
# (e.g. GPU preconditioned CG). Peak RAM ~ n²·8 bytes for the reduced mass
# matrix; 10_000 → ~800 MiB.  Map fits with ``ns`` ~ (14,22,22) and ``p``=1 can
# exceed 6_000 DOFs, so the old 6_000 cap blocked fallback entirely.
_MAX_DENSE_MAP_MASS_N: int = 10000


def _axis_index_lookup(axis: np.ndarray, x: float, atol: float) -> int:
    """Find *i* such that ``abs(axis[i] - x) <= atol``."""
    j = int(np.searchsorted(axis, x))
    for idx in (j, j - 1):
        if 0 <= idx < len(axis) and abs(float(axis[idx]) - x) <= atol:
            return idx
    raise ValueError(
        f"Coordinate {x!r} not aligned with axis (tol={atol}); "
        f"nearest span [{axis[0]}, {axis[-1]}]."
    )


def mrx_logical_eval_points_index_grid(
    grid_shape: tuple[int, int, int],
    *,
    flip_r: bool = False,
) -> np.ndarray:
    """
    Build MRX logical ``(ρ, θ, ζ) ∈ [0, 1]³`` from tensor indices (no periodic wrap).

    Row-major flatten order matches ``R_3d.ravel()`` / GVEC export layout
  ``(nρ, nθ, nζ)`` with ``indexing='ij'``.  Uses ``θ = j/nθ``, ``ζ = k/nζ`` so the
    poloidal/toroidal **endpoints are not folded** to 0 (avoids duplicate logical
    cells when physical angles include ``2π``).
    """
    nr, nt, nz = (int(grid_shape[0]), int(grid_shape[1]), int(grid_shape[2]))
    ir = np.arange(nr, dtype=np.float64)
    it = np.arange(nt, dtype=np.float64)
    iz = np.arange(nz, dtype=np.float64)
    Ir, It, Iz = np.meshgrid(ir, it, iz, indexing="ij")
    r_mrx = Ir / max(nr - 1, 1)
    theta_mrx = It / max(nt, 1)
    zeta_mrx = Iz / max(nz, 1)
    if flip_r:
        r_mrx = 1.0 - r_mrx
    return np.stack(
        [r_mrx.ravel(), theta_mrx.ravel(), zeta_mrx.ravel()], axis=1
    )


def repair_tensor_eval_points_if_needed(
    pts: np.ndarray,
    *,
    flip_r: bool = False,
) -> tuple[np.ndarray, bool]:
    """
    If ``pts`` are a perfect cube but unique axes do not multiply to ``N``, rebuild
    logical coordinates from grid indices (GVEC export with ``angle % 1``).

    Returns ``(pts_out, repaired)``; ``repaired`` is ``True`` when coordinates were
    replaced (``R``, ``Z``, ``B`` row order is unchanged).
    """
    pts = np.asarray(pts, dtype=np.float64)
    n = int(pts.shape[0])
    side = int(round(n ** (1.0 / 3.0)))
    if side**3 != n:
        return pts, False
    u = np.unique(pts[:, 0])
    v = np.unique(pts[:, 1])
    w = np.unique(pts[:, 2])
    if int(len(u) * len(v) * len(w)) == n:
        return pts, False
    fixed = mrx_logical_eval_points_index_grid((side, side, side), flip_r=flip_r)
    print(
        "WARNING: eval_points are not a full tensor grid after unique-axis check "
        f"(N={n}, nρ={len(u)}, nθ={len(v)}, nζ={len(w)}, product={len(u) * len(v) * len(w)}). "
        "Replacing logical (ρ,θ,ζ) with index-based MRX coordinates "
        f"({side}×{side}×{side}, θ=j/nθ, ζ=k/nζ) so R,Z,B row order is preserved. "
        "Re-export with index-based eval_points to silence this.",
        flush=True,
    )
    return fixed, True


def _tensor_axes_and_grids(
    pts: np.ndarray,
    R: np.ndarray,
    Z: np.ndarray,
    atol: float = 1e-9,
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray, np.ndarray]:
    """
    Infer sorted logical axes and 3-D *R*, *Z* grids from points on a full tensor grid.

    Points must be samples on a Cartesian product of one-dimensional coordinate
    lists (Fortran / ``indexing='ij'`` flatten order is *not* required; cells are
    filled by matching coordinates).
    """
    pts = np.asarray(pts, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64).ravel()
    Z = np.asarray(Z, dtype=np.float64).ravel()
    if pts.shape[1] != 3:
        raise ValueError(f"pts must have shape (N, 3); got {pts.shape}")
    n = pts.shape[0]
    if R.shape[0] != n or Z.shape[0] != n:
        raise ValueError("R, Z must have the same length as pts")

    u = np.unique(pts[:, 0])
    v = np.unique(pts[:, 1])
    w = np.unique(pts[:, 2])
    n1, n2, n3 = len(u), len(v), len(w)
    if n1 * n2 * n3 != n:
        raise ValueError(
            f"Data do not form a full tensor grid in (ρ, θ, ζ): N={n} samples, but "
            f"unique coordinates along each axis are nρ={n1}, nθ={n2}, nζ={n3}, "
            f"so a volume product grid would need nρ·nθ·nζ={n1 * n2 * n3} points (not {n}). "
            "This usually means the file is a surface slice, scattered samples, or "
            "non-Cartesian-product points. "
            "`interpolate_map_from_points` / `relax_from_nfs` need a precomputed **volume** "
            "NFS HDF5 (e.g. from `precompute_desc_data.py`) with eval_points, R, Z on a full "
            "rho–theta–zeta product grid."
        )

    R_grid = np.full((n1, n2, n3), np.nan, dtype=np.float64)
    Z_grid = np.full((n1, n2, n3), np.nan, dtype=np.float64)
    count = np.zeros((n1, n2, n3), dtype=np.int32)

    for row in range(n):
        i = _axis_index_lookup(u, float(pts[row, 0]), atol)
        j = _axis_index_lookup(v, float(pts[row, 1]), atol)
        k = _axis_index_lookup(w, float(pts[row, 2]), atol)
        R_grid[i, j, k] = R[row]
        Z_grid[i, j, k] = Z[row]
        count[i, j, k] += 1

    if np.any(count != 1):
        raise ValueError(
            "Duplicate or missing grid cells when mapping scattered points to "
            f"the tensor grid (count min={int(count.min())}, max={int(count.max())})."
        )
    return (u, v, w), R_grid, Z_grid


def _clip_quad_to_data_axes(
    xq: jnp.ndarray,
    x1: jnp.ndarray,
    x2: jnp.ndarray,
    x3: jnp.ndarray,
) -> jnp.ndarray:
    """Clamp quadrature coordinates to sampled-axis spans."""
    c0 = jnp.clip(xq[:, 0], x1[0], x1[-1])
    c1 = jnp.clip(xq[:, 1], x2[0], x2[-1])
    c2 = jnp.clip(xq[:, 2], x3[0], x3[-1])
    return jnp.stack([c0, c1, c2], axis=-1)


def _k0_dof_flat_index(nr: int, nt: int, nz: int, i: int, j: int, k: int) -> int:
    """
    Row-major flat index for a scalar 0-form DOF ``(i,j,k)`` in ``(ρ,θ,ζ)`` order.

    Matches :meth:`mrx.differential_forms.DifferentialForm._ravel_index` for ``k=0``.
    """
    return int(i * (nt * nz) + j * nz + k)


def _rho0_theta_indep_constraint_matrix(nr: int, nt: int, nz: int) -> npt.NDArray[np.float64]:
    """
    Build ``C`` such that ``C @ c = 0`` enforces ``c[0,j,k] = c[0,0,k]`` on the
    tensor-product 0-form coefficient tensor (clamped ``ρ``, first radial index ``0``).

    With standard clamped B-splines, only the first radial basis is nonzero at ``ρ=0``,
    so this is equivalent to θ-independence of the fitted scalar on the ``ρ=0`` face.
    """
    n_dof = nr * nt * nz
    n_con = (nt - 1) * nz if nt > 1 else 0
    if n_con == 0:
        return np.zeros((0, n_dof), dtype=np.float64)
    cmat = np.zeros((n_con, n_dof), dtype=np.float64)
    row = 0
    for k in range(nz):
        i0 = _k0_dof_flat_index(nr, nt, nz, 0, 0, k)
        for j in range(1, nt):
            cmat[row, _k0_dof_flat_index(nr, nt, nz, 0, j, k)] = 1.0
            cmat[row, i0] = -1.0
            row += 1
    return cmat


def _solve_qp_equality_spd(
    m: npt.NDArray[np.float64],
    b: npt.NDArray[np.float64],
    cmat: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Solve ``min_c ½ cᵀ M c − bᵀ c`` subject to ``C c = 0`` with symmetric positive
    definite ``M`` via the KKT saddle system.
    """
    m = np.asarray(m, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64).ravel()
    n_dof = m.shape[0]
    if cmat.size == 0:
        return np.linalg.solve(m, b)
    cmat = np.asarray(cmat, dtype=np.float64)
    n_c = int(cmat.shape[0])
    kkt = np.block(
        [
            [m, cmat.T],
            [cmat, np.zeros((n_c, n_c), dtype=np.float64)],
        ]
    )
    rhs = np.concatenate([b, np.zeros(n_c, dtype=np.float64)])
    sol = np.linalg.solve(kkt, rhs)
    return sol[:n_dof]


def _project_sampled_scalar_0form_dense_rho0_theta_indep(
    axes: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    values: jnp.ndarray,
    seq,
    *,
    dirichlet: bool,
) -> jnp.ndarray:
    """
    Same L² projection as :func:`_project_sampled_scalar_0form_dense`, but enforce
    θ-independence on the ``ρ=0`` face (equal 0-form coefficients along θ at the
    inner clamped radial index).
    """
    from jax.scipy.interpolate import RegularGridInterpolator

    from mrx.assembly import assemble_dense_mass_matrix
    from mrx.utils import integrate_against

    x1, x2, x3 = axes
    n1, n2, n3 = len(x1), len(x2), len(x3)
    xq = seq.quad.x
    xq_i = _clip_quad_to_data_axes(xq, x1, x2, x3)
    comp_info, comp_shapes = seq._form_comp_info(0)
    quad_shape = (seq.quad.ny, seq.quad.nx, seq.quad.nz)
    grid = values.reshape(n1, n2, n3)
    interp = RegularGridInterpolator(
        points=(x1, x2, x3), values=grid, method="linear"
    )
    f_q = interp(xq_i)[:, None]
    w_jk = f_q * (seq.quad.w * seq.jacobian_j)[:, None]
    e = seq.e0_dbc if dirichlet else seq.e0
    rhs = e @ integrate_against(w_jk, comp_info, comp_shapes, quad_shape)
    m = np.asarray(
        assemble_dense_mass_matrix(seq, 0, dirichlet=dirichlet), dtype=np.float64
    )
    rhs_np = np.asarray(rhs, dtype=np.float64).ravel()
    nr, nt, nz = int(seq.basis_0.nr), int(seq.basis_0.nt), int(seq.basis_0.nz)
    cmat = _rho0_theta_indep_constraint_matrix(nr, nt, nz)
    c_np = _solve_qp_equality_spd(m, rhs_np, cmat)
    return jnp.asarray(c_np)


def _project_sampled_scalar_0form_dense(
    axes: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    values: jnp.ndarray,
    seq,
    *,
    dirichlet: bool,
) -> jnp.ndarray:
    """
    L²-project a scalar sampled on a tensor grid onto 0-forms via a dense reduced
    mass solve (same RHS as :func:`mrx.io.project_sampled_field`).

    Used when :meth:`mrx.derham_sequence.DeRhamSequence.apply_inverse_mass_matrix`
    (preconditioned CG) returns non-finite coefficients.
    """
    from jax.scipy.interpolate import RegularGridInterpolator

    from mrx.assembly import assemble_dense_mass_matrix
    from mrx.utils import integrate_against

    x1, x2, x3 = axes
    n1, n2, n3 = len(x1), len(x2), len(x3)
    xq = seq.quad.x
    xq_i = _clip_quad_to_data_axes(xq, x1, x2, x3)
    comp_info, comp_shapes = seq._form_comp_info(0)
    quad_shape = (seq.quad.ny, seq.quad.nx, seq.quad.nz)
    grid = values.reshape(n1, n2, n3)
    interp = RegularGridInterpolator(
        points=(x1, x2, x3), values=grid, method="linear"
    )
    f_q = interp(xq_i)[:, None]
    w_jk = f_q * (seq.quad.w * seq.jacobian_j)[:, None]
    e = seq.e0_dbc if dirichlet else seq.e0
    rhs = e @ integrate_against(w_jk, comp_info, comp_shapes, quad_shape)
    m = assemble_dense_mass_matrix(seq, 0, dirichlet=dirichlet)
    return jnp.linalg.solve(m, rhs)


def _project_map_scalar_0form_with_fallback(
    axes: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    grid: jnp.ndarray,
    seq_init,
    *,
    dirichlet: bool,
    label: str,
    rho0_theta_independent: bool = False,
) -> jnp.ndarray:
    """Run :func:`mrx.io.project_sampled_field` for k=0; fall back to dense mass if needed."""
    from mrx.io import project_sampled_field

    if rho0_theta_independent:
        n_red = seq_init.n0_dbc if dirichlet else seq_init.n0
        if n_red > _MAX_DENSE_MAP_MASS_N:
            raise RuntimeError(
                f"Map fit: ρ=0 θ-independence uses a dense KKT solve; n0={n_red} exceeds "
                f"{_MAX_DENSE_MAP_MASS_N}. Lower map resolution or raise "
                "``mrx.io_nfs_map._MAX_DENSE_MAP_MASS_N``."
            )
        dof = _project_sampled_scalar_0form_dense_rho0_theta_indep(
            axes, grid, seq_init, dirichlet=dirichlet
        )
        if not bool(jnp.all(jnp.isfinite(dof))):
            raise RuntimeError(
                f"Map fit: constrained L² projection of {label} is non-finite "
                "(check R,Z samples and logical axes)."
            )
        return dof

    dof = project_sampled_field(axes, grid, seq_init, 0, dirichlet)
    if bool(jnp.all(jnp.isfinite(dof))):
        return dof
    n_red = seq_init.n0_dbc if dirichlet else seq_init.n0
    if n_red > _MAX_DENSE_MAP_MASS_N:
        raise RuntimeError(
            f"Map fit: L² projection of {label} gave non-finite DOFs (iterative mass "
            f"solve failed) and dense fallback is disabled for reduced n0={n_red} "
            f"(limit {_MAX_DENSE_MAP_MASS_N}). "
            "Try: lower ``--map-nr/--map-ntheta/--map-nzeta`` (or spline degree), "
            "``export JAX_PLATFORMS=cpu`` for the map fit, or raise "
            "``mrx.io_nfs_map._MAX_DENSE_MAP_MASS_N`` if you have enough RAM."
        )
    dof = _project_sampled_scalar_0form_dense(
        axes, grid, seq_init, dirichlet=dirichlet)
    if not bool(jnp.all(jnp.isfinite(dof))):
        raise RuntimeError(
            f"Map fit: L² projection of {label} is still non-finite after dense "
            "mass solve (check R,Z HDF5 samples and logical axes)."
        )
    return dof


def evaluate_map_rz_residual_stats(
    map_func: Callable,
    pts,
    R,
    Z,
    *,
    exclude_axis_tol: float = 0.0,
    rho_lo: float | None = None,
    rho_hi: float | None = None,
) -> dict[str, float]:
    """
    Pointwise map misfit of fitted ``map_func`` against sampled ``R``, ``Z``.

    Returns max and RMS absolute errors over the evaluation mask.

    Parameters
    ----------
    map_func
        Fitted stellarator map ``(ρ, θ, ζ) -> (X, Y, Z)``.
    pts, R, Z
        Logical grid samples and cylindrical targets (same length).
    exclude_axis_tol
        Exclude points with ``ρ <= tol`` or ``ρ >= 1 - tol``.
    rho_lo, rho_hi
        If set, further restrict to ``rho_lo < ρ < rho_hi`` (interior shell).

    Returns
    -------
    dict
        Keys ``max_R``, ``max_Z``, ``rms_R``, ``rms_Z``, ``n_points``.
    """
    pts_j = jnp.asarray(pts, dtype=jnp.float64)
    R_j = jnp.asarray(R, dtype=jnp.float64).ravel()
    Z_j = jnp.asarray(Z, dtype=jnp.float64).ravel()
    xyz = jax.vmap(map_func)(pts_j)
    R_pred = jnp.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2)
    Z_pred = xyz[:, 2]
    dR = jnp.abs(R_pred - R_j)
    dZ = jnp.abs(Z_pred - Z_j)

    mask = jnp.ones(pts_j.shape[0], dtype=bool)
    if exclude_axis_tol > 0.0:
        mask = mask & (pts_j[:, 0] > exclude_axis_tol) & (
            pts_j[:, 0] < 1.0 - exclude_axis_tol
        )
    if rho_lo is not None:
        mask = mask & (pts_j[:, 0] > float(rho_lo))
    if rho_hi is not None:
        mask = mask & (pts_j[:, 0] < float(rho_hi))
    if not bool(jnp.any(mask)):
        mask = jnp.ones(pts_j.shape[0], dtype=bool)

    dR_m = dR[mask]
    dZ_m = dZ[mask]
    return {
        "max_R": float(jnp.max(dR_m)),
        "max_Z": float(jnp.max(dZ_m)),
        "rms_R": float(jnp.sqrt(jnp.mean(dR_m ** 2))),
        "rms_Z": float(jnp.sqrt(jnp.mean(dZ_m ** 2))),
        "n_points": float(dR_m.shape[0]),
    }


def interpolate_map_from_points(
    pts,
    R,
    Z,
    nfp: int,
    *,
    ns: Tuple[int, int, int],
    ps: Tuple[int, int, int],
    quad_order: int = 6,
    flip_zeta: bool = False,
    exclude_axis_tol: float = 0.0,
    rho0_theta_independent: bool = False,
) -> Tuple[Callable, jnp.ndarray, jnp.ndarray, Tuple[float, float, float, float]]:
    """
    Build a stellarator coordinate map by L²-projecting *R* and *Z* onto 0-forms.

    The samples ``(pts, R, Z)`` must lie on a **full** tensor product grid in
    logical coordinates ``(ρ, θ, ζ) ∈ [0, 1]³``.  Axes are inferred from unique
    coordinate values along each dimension.

    Parameters
    ----------
    pts : array_like, shape (N, 3)
        Logical evaluation points (same layout as NFS / precompute HDF5 files).
    R, Z : array_like, shape (N,)
        Cylindrical major radius and height consistent with the target map.
    nfp : int
        Number of field periods for :func:`mrx.mappings.stellarator_map`.
    ns, ps : tuple of int
        Spline resolution ``(nρ, nθ, nζ)`` and polynomial degrees for the
        auxiliary :class:`mrx.derham_sequence.DeRhamSequence` used on the
        **identity** logical domain (``seq.set_map(lambda x: x)`` after build).
    quad_order : int
        Quadrature order for the projection.
    flip_zeta : bool
        Passed to :func:`mrx.mappings.stellarator_map`.
    exclude_axis_tol : float
        When computing reported max residuals, only use points with
        ``ρ ∈ (exclude_axis_tol, 1 - exclude_axis_tol)``.  Does not remove
        points from the fit (the grid must remain complete).
    rho0_theta_independent : bool
        If ``True``, the L² fits for *R* and *Z* enforce that spline coefficients
        at the inner clamped radial index are **independent of θ** (all θ-rows
        tied to the θ₀ row).  For clamped B-splines this matches a θ-independent
        trace on the ``ρ=0`` logical face, closing spurious θ variation at the
        axis in the map.  Uses a dense KKT solve (same size cap as the dense
        mass fallback).

    Returns
    -------
    map_func : callable
        ``F(ρ, θ, ζ) -> (X, Y, Z)`` stellarator map.
    R_dof, Z_dof : jnp.ndarray
        0-form DOF vectors for the projected *R* and *Z*.
    resid_RZ : tuple of float
        ``(max |ΔR|, max |ΔZ|, rms |ΔR|, rms |ΔZ|)`` over the residual mask.
    """
    import mrx

    if getattr(mrx, "MAP_BATCH_SIZE_INNER", 0) <= 0:
        mrx.MAP_BATCH_SIZE_INNER = 1

    # Local import avoids circular import: :mod:`mrx.io` may re-export this function.
    from mrx.derham_sequence import DeRhamSequence
    from mrx.differential_forms import DiscreteFunction
    from mrx.mappings import stellarator_map

    pts_np, _repaired = repair_tensor_eval_points_if_needed(np.asarray(pts))
    axes_np, R_np, Z_np = _tensor_axes_and_grids(
        pts_np, np.asarray(R), np.asarray(Z)
    )
    axes = tuple(jnp.asarray(a) for a in axes_np)
    R_grid = jnp.asarray(R_np)
    Z_grid = jnp.asarray(Z_np)

    ps_use = tuple(min(p, n - 1) for p, n in zip(ps, ns))
    seq_init = DeRhamSequence(
        ns,
        ps_use,
        quad_order,
        ("clamped", "periodic", "periodic"),
        polar=False,
    )
    # Identity logical map for L² projection of R,Z on the reference cell only.
    seq_init.set_map(lambda x: jnp.asarray(x, dtype=jnp.float64))
    if not bool(jnp.all(jnp.isfinite(seq_init.jacobian_j))):
        raise RuntimeError(
            "Map fit: identity DeRhamSequence has non-finite jacobian_j (often "
            "jax.lax.map(..., batch_size<=0) on this JAX build). "
            f"mrx.MAP_BATCH_SIZE_INNER={getattr(mrx, 'MAP_BATCH_SIZE_INNER', None)}. "
            "Use current mrx (DeRhamSequence coerces batch size >= 1) or export "
            "MAP_BATCH_SIZE_INNER=1 before import."
        )
    seq_init.evaluate_1d()
    seq_init.assemble_mass_matrix(0)

    R_dof = _project_map_scalar_0form_with_fallback(
        axes,
        R_grid,
        seq_init,
        dirichlet=False,
        label="R",
        rho0_theta_independent=bool(rho0_theta_independent),
    )
    Z_dof = _project_map_scalar_0form_with_fallback(
        axes,
        Z_grid,
        seq_init,
        dirichlet=False,
        label="Z",
        rho0_theta_independent=bool(rho0_theta_independent),
    )
    R_h = DiscreteFunction(R_dof, seq_init.basis_0, seq_init.e0)
    Z_h = DiscreteFunction(Z_dof, seq_init.basis_0, seq_init.e0)
    map_func = stellarator_map(R_h, Z_h, nfp=nfp, flip_zeta=flip_zeta)

    stats = evaluate_map_rz_residual_stats(
        map_func,
        pts,
        R,
        Z,
        exclude_axis_tol=exclude_axis_tol,
    )
    resid_RZ = (
        stats["max_R"],
        stats["max_Z"],
        stats["rms_R"],
        stats["rms_Z"],
    )

    return map_func, R_dof, Z_dof, resid_RZ
