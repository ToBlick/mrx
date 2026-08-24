"""Toroidal circulation integrals for harmonic vacuum pushforward fields.

Computes line integrals

    Γ = ∮ B · dℓ

on the standard logical loop ``(ρ = ρ_b, θ = 0, ζ ∈ [0, 1))`` using the native geometry. Uses trapezoidal quadrature.

The primary use is amplitude normalization if need to compare MRX's solution to SIMSOPT's.
"""
from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from mrx.differential_forms import DiscreteFunction, Pushforward
from mrx.derham_sequence import DeRhamSequence

if hasattr(np, "trapezoid"):
    _numpy_trapz = np.trapezoid
else:
    _numpy_trapz = np.trapz

DEFAULT_NATIVE_CIRCULATION_RHO: float = 0.6
DEFAULT_EXTEND_MAP_CIRCULATION_RHO: float = 1.0 - 1.0e-10


def default_circulation_rho(*, circulation_map: str) -> float:
    """Return the recommended ``ρ_b`` for the chosen circulation map mode."""
    mode = str(circulation_map).strip().lower().replace("-", "_")
    if mode in ("native", "single_period"):
        return float(DEFAULT_NATIVE_CIRCULATION_RHO)
    return float(DEFAULT_EXTEND_MAP_CIRCULATION_RHO)


def boundary_toroidal_loop_logical(
    *,
    n_quad: int,
    rho_boundary: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build quadrature points for the toroidal loop ``(ρ=ρ_b, θ=0, ζ∈[0,1))``.

    Returns
    -------
    zeta, logical
        Parameter samples ``zeta`` (length ``n_quad``) and logical points
        ``(N, 3)`` with columns ``(ρ, θ, ζ)``.
    """
    n = int(n_quad)
    if n < 8:
        raise ValueError(f"n_quad must be >= 8; got {n}")
    zeta = np.linspace(0.0, 1.0, n, endpoint=False, dtype=np.float64)
    logical = np.stack(
        [
            np.full(n, float(rho_boundary), dtype=np.float64),
            np.zeros(n, dtype=np.float64),
            zeta,
        ],
        axis=1,
    )
    return zeta, logical


def circulation_line_integral_T_m(
    b_cart: np.ndarray,
    xyz: np.ndarray,
    parameter: np.ndarray,
) -> float:
    """
    Compute ``∮ B · dℓ`` [T·m] via trapezoid rule along ``parameter``.

    Parameters
    ----------
    b_cart
        Cartesian magnetic field ``(N, 3)`` [T].
    xyz
        Cartesian positions ``(N, 3)`` [m].
    parameter
        1D parameter matching the loop orientation (typically ``ζ``).
    """
    b_cart = np.asarray(b_cart, dtype=np.float64).reshape(-1, 3)
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    param = np.asarray(parameter, dtype=np.float64).reshape(-1)
    if b_cart.shape[0] != xyz.shape[0] or b_cart.shape[0] != param.shape[0]:
        raise ValueError(
            f"b_cart, xyz, parameter length mismatch: "
            f"{b_cart.shape[0]}, {xyz.shape[0]}, {param.shape[0]}"
        )
    dx = np.gradient(xyz, param, axis=0, edge_order=2)
    integrand = np.sum(b_cart * dx, axis=1)
    return float(_numpy_trapz(integrand, param))


def map_logical_points(
    map_fn: Callable[[jnp.ndarray], jnp.ndarray],
    logical: np.ndarray,
    *,
    batch_size: int = 256,
) -> np.ndarray:
    """Evaluate a geometry map on logical points with bounded JAX batching."""
    points = np.asarray(logical, dtype=np.float64).reshape(-1, 3)
    if points.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    mapped = np.asarray(
        jax.lax.map(
            jax.jit(map_fn),
            jnp.asarray(points, dtype=jnp.float64),
            batch_size=int(batch_size),
        ),
        dtype=np.float64,
    )
    return mapped


def evaluate_harmonic_pushforward_B(
    seq: DeRhamSequence,
    u: jnp.ndarray,
    map_fn: Callable[[jnp.ndarray], jnp.ndarray],
    logical: np.ndarray,
    *,
    k: int,
    nfp: int = 1,
    circulation_map: str = "native",
    batch_size: int = 256,
) -> np.ndarray:
    """
    Evaluate ``Pushforward(u)`` as Cartesian **B** on logical sample points.

    Parameters
    ----------
    k
        Form degree ``1`` or ``2``.
    circulation_map
        ``native`` uses ``map_fn`` on one toroidal period. ``extend_map`` uses
        ``extend_map_nfp`` with one-period localization (legacy).
    """
    mode = str(circulation_map).strip().lower().replace("-", "_")
    u = jnp.asarray(u, dtype=jnp.float64)
    logical_np = np.asarray(logical, dtype=np.float64).reshape(-1, 3)
    logical_j = jnp.asarray(logical_np, dtype=jnp.float64)

    if int(k) == 1:
        disc = DiscreteFunction(u, seq.basis_1, seq.e1)
        form_k = 1
    elif int(k) == 2:
        disc = DiscreteFunction(u, seq.basis_2, seq.e2_dbc)
        form_k = 2
    else:
        raise ValueError(f"k must be 1 or 2; got {k}")

    if mode in ("native", "single_period"):
        map_ev = jax.jit(map_fn)
        fld = Pushforward(disc, map_ev, form_k)
    elif mode in ("extend_map", "extend_map_nfp", "full_torus"):
        from mrx.mappings import extend_map_nfp

        nfp_i = int(nfp)

        def localized(x: jnp.ndarray) -> jnp.ndarray:
            x = jnp.asarray(x, dtype=jnp.float64).reshape(3)
            xi = x[2] * float(nfp_i)
            zloc = xi - jnp.floor(xi)
            return disc(x.at[2].set(zloc))

        full_map = extend_map_nfp(map_fn, nfp_i)
        fld = Pushforward(localized, jax.jit(full_map), form_k)
    else:
        raise ValueError(
            f"circulation_map must be 'native' or 'extend_map'; got {circulation_map!r}"
        )

    fld_jit = jax.jit(fld)
    return np.asarray(
        jax.lax.map(fld_jit, logical_j, batch_size=int(batch_size)),
        dtype=np.float64,
    )


def evaluate_harmonic_curl_pushforward_B(
    seq: DeRhamSequence,
    u: jnp.ndarray,
    map_fn: Callable[[jnp.ndarray], jnp.ndarray],
    logical: np.ndarray,
    *,
    circulation_map: str = "native",
    nfp: int = 1,
    batch_size: int = 256,
) -> np.ndarray:
    """Evaluate ``Pushforward(strong_curl(u))`` as Cartesian **B** (k=1 only)."""
    B_dof = seq.apply_strong_curl(
        jnp.asarray(u, dtype=jnp.float64),
        dirichlet_in=False,
        dirichlet_out=True,
    )
    disc = DiscreteFunction(B_dof, seq.basis_2, seq.e2_dbc)
    mode = str(circulation_map).strip().lower().replace("-", "_")
    logical_j = jnp.asarray(np.asarray(logical, dtype=np.float64).reshape(-1, 3))

    if mode in ("native", "single_period"):
        fld = Pushforward(disc, jax.jit(map_fn), 2)
    elif mode in ("extend_map", "extend_map_nfp", "full_torus"):
        from mrx.mappings import extend_map_nfp

        nfp_i = int(nfp)

        def localized(x: jnp.ndarray) -> jnp.ndarray:
            x = jnp.asarray(x, dtype=jnp.float64).reshape(3)
            xi = x[2] * float(nfp_i)
            zloc = xi - jnp.floor(xi)
            return disc(x.at[2].set(zloc))

        fld = Pushforward(localized, jax.jit(extend_map_nfp(map_fn, nfp_i)), 2)
    else:
        raise ValueError(f"unsupported circulation_map={circulation_map!r}")

    return np.asarray(
        jax.lax.map(jax.jit(fld), logical_j, batch_size=int(batch_size)),
        dtype=np.float64,
    )


def circulation_cross_checks(
    b_cart: np.ndarray,
    xyz: np.ndarray,
    zeta: np.ndarray,
    *,
    reference_gamma: float | None = None,
) -> dict[str, Any]:
    """
    Basic consistency checks for a toroidal circulation quadrature.

    Returns ``ok`` plus forward/reversed ``Γ`` and relative change when ``n_quad``
    is doubled.
    """
    b_cart = np.asarray(b_cart, dtype=np.float64)
    xyz = np.asarray(xyz, dtype=np.float64)
    zeta = np.asarray(zeta, dtype=np.float64)
    gamma_fwd = circulation_line_integral_T_m(b_cart, xyz, zeta)
    gamma_rev = circulation_line_integral_T_m(b_cart, xyz[::-1], zeta[::-1])
    n = int(zeta.size)
    zeta_dense = np.linspace(0.0, 1.0, 2 * n, endpoint=False, dtype=np.float64)
    # Re-interpolate B and xyz linearly in zeta for a cheap refinement check.
    b_dense = np.stack(
        [np.interp(zeta_dense, zeta, b_cart[:, i]) for i in range(3)],
        axis=1,
    )
    xyz_dense = np.stack(
        [np.interp(zeta_dense, zeta, xyz[:, i]) for i in range(3)],
        axis=1,
    )
    gamma_dense = circulation_line_integral_T_m(b_dense, xyz_dense, zeta_dense)
    rel_refine = abs(gamma_dense - gamma_fwd) / max(abs(gamma_fwd), 1.0e-30)
    rel_reverse = abs(gamma_fwd + gamma_rev) / max(abs(gamma_fwd), 1.0e-30)
    ok = rel_reverse < 1e-8
    if reference_gamma is not None:
        rel_to_ref = abs(gamma_fwd - float(reference_gamma)) / max(
            abs(float(reference_gamma)), 1.0e-30
        )
        ok = ok and rel_to_ref < 1.0e-10
    else:
        rel_to_ref = float("nan")
    if rel_refine > 0.25:
        ok = False
    return {
        "ok": bool(ok),
        "gamma_forward_T_m": float(gamma_fwd),
        "gamma_reversed_T_m": float(gamma_rev),
        "gamma_refined_T_m": float(gamma_dense),
        "rel_reverse_sum": float(rel_reverse),
        "rel_refine_change": float(rel_refine),
        "rel_to_reference": float(rel_to_ref),
    }


def boundary_toroidal_circulation(
    seq: DeRhamSequence,
    u: jnp.ndarray,
    map_fn: Callable[[jnp.ndarray], jnp.ndarray],
    *,
    k: int = 2,
    n_quad: int = 256,
    rho_boundary: float | None = None,
    circulation_map: str = "native",
    nfp: int = 1,
    b_field: str = "pushforward",
    batch_size: int = 256,
    reference_b_cart: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Toroidal circulation on the standard boundary loop for harmonic DOF ``u``.

    When ``reference_b_cart`` is supplied (same length as loop samples), also
    returns ``alpha = Γ_ref / Γ_MRX`` and ``B0_derived = Γ_ref / (2π R_ref)``.
    """
    mode = str(circulation_map).strip().lower().replace("-", "_")
    rho_b = float(
        default_circulation_rho(circulation_map=mode)
        if rho_boundary is None
        else rho_boundary
    )
    zeta, logical = boundary_toroidal_loop_logical(n_quad=int(n_quad), rho_boundary=rho_b)
    xyz = map_logical_points(map_fn, logical, batch_size=batch_size)

    b_field_eff = str(b_field).strip().lower()
    if int(k) == 1 and b_field_eff == "curl":
        b_mrx = evaluate_harmonic_curl_pushforward_B(
            seq,
            u,
            map_fn,
            logical,
            circulation_map=circulation_map,
            nfp=nfp,
            batch_size=batch_size,
        )
        map_label = f"{mode}_pushforward_curl_u1"
    else:
        b_mrx = evaluate_harmonic_pushforward_B(
            seq,
            u,
            map_fn,
            logical,
            k=int(k),
            nfp=nfp,
            circulation_map=circulation_map,
            batch_size=batch_size,
        )
        map_label = f"{mode}_pushforward_u{int(k)}"

    gamma_mrx = circulation_line_integral_T_m(b_mrx, xyz, zeta)
    R = np.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2)
    r_ref = float(_numpy_trapz(R, zeta))
    cross = circulation_cross_checks(b_mrx, xyz, zeta, reference_gamma=gamma_mrx)

    out: dict[str, Any] = {
        "circulation_map": mode,
        "map": map_label,
        "method": "harmonic_dof",
        "form_degree": int(k),
        "n_quadrature": int(n_quad),
        "rho_boundary": rho_b,
        "gamma_mrx_T_m": float(gamma_mrx),
        "R_ref_m": r_ref,
        "cross_checks": cross,
    }
    if int(k) == 1:
        out["b_field"] = b_field_eff

    if reference_b_cart is not None:
        b_ref = np.asarray(reference_b_cart, dtype=np.float64).reshape(-1, 3)
        if b_ref.shape[0] != b_mrx.shape[0]:
            raise ValueError(
                f"reference_b_cart length {b_ref.shape[0]} != loop samples {b_mrx.shape[0]}"
            )
        gamma_ref = circulation_line_integral_T_m(b_ref, xyz, zeta)
        if abs(gamma_mrx) < 1.0e-30:
            alpha = float("nan")
            b0_derived = float("nan")
        else:
            alpha = float(gamma_ref / gamma_mrx)
            b0_derived = float(gamma_ref / (2.0 * np.pi * r_ref)) if r_ref > 0.0 else float("nan")
        out.update(
            {
                "gamma_reference_T_m": float(gamma_ref),
                "alpha": alpha,
                "B0_derived_T": b0_derived,
                "cross_checks_reference": circulation_cross_checks(
                    b_ref, xyz, zeta, reference_gamma=gamma_ref
                ),
            }
        )
    return out


