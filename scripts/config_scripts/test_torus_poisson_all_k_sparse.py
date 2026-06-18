"""Convergence study for the Hodge–Laplacians on a toroidal domain, all BCs.

A single script that sweeps the eight (k, boundary-condition) Laplace problems
that share manufactured solutions.  Cases pair up under Hodge duality
⋆: k ↔ (3-k), NBC ↔ DBC, into four generators:

  generator        cases                       exact field / source
  ---------------  --------------------------  --------------------------------------
  cos(2πζ)         k0 NBC, k3 DBC              u = cos(2πζ),  f₀ = cos(2πζ)/R²
  cos(πr²/2)       k0 DBC, k3 NBC              u = cos(πr²/2),  f₀ = (2πs + π²r²c)/ε² + πrs cos(2πχ)/(εR)
    cos(2πζ) dζ      k1 NBC, k2 DBC              ω₁ = cos(2πζ) dζ, ω₂ = ⋆ω₁;  f₁ = grad σ, f₂ = ⋆f₁
    cos(πr²/2)cos(2πζ) dζ  k1 DBC, k2 NBC        ω = cos(πr²/2)cos(2πζ) dζ;  f₁ = dσ + curl·curl ω
  (s = sin(πr²/2), c = cos(πr²/2))

Harmonic (nullspace) dimensions per case (betti = (1,1,0,0)):
  k0 NBC: 1 (constant)         k0 DBC: 0
  k1 NBC: 1 (toroidal 1-form)  k1 DBC: 0
  k2 NBC: 0                    k2 DBC: 1 (toroidal 2-form)
  k3 NBC: 0                    k3 DBC: 1 (constant 3-form)

Currently enabled cases (see CASES below): all eight (k, BC) pairs
  (k0 NBC, k0 DBC, k1 NBC, k1 DBC, k2 NBC, k2 DBC, k3 NBC, k3 DBC).

ω₁ = cos(2πζ) dζ is closed (curl-free), so f₁ = L₁ω₁ = grad σ with σ = -div ω₁,
and is orthogonal to the harmonic 1-form because cos has zero mean.  The k1 DBC
field ω = cos(πr²/2)cos(2πζ) dζ is not divergence-free in the interior, but its
boundary conditions still hold and the source probes all three covariant slots:
f₁ = dσ + curl·curl ω;
its tangential trace vanishes at the wall (u×n = 0) and σ = 0 there (both essential
k1 DBC conditions), and k1 DBC has no nullspace.  ω₂ = ⋆ω₁ is co-closed with zero
normal trace (the essential k=2 DBC).  The Hodge star ⋆: Ω¹→Ω² uses the framework's
cyclic vector-proxy convention (all-positive in the diagonal metric).
See docs/manufactured_solutions.md.

All enabled cases share one DeRhamSequence and one assembly pass.  Both frame='ref'
and frame='phys' loads are assembled per case as a consistency check (≈ 0).
One SLURM job per (n, p) pair, sweeping n=8,12,16,20 and p=1,2,3.

Usage (from repo root):
    python scripts/config_scripts/test_torus_poisson_all_k_sparse.py -m p=1,2,3 n=8,12,16,20
"""
import json
import os
import time

import hydra
import jax
import jax.numpy as jnp
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

import mrx
import mrx.config  # noqa: F401 — register structured configs in ConfigStore
from mrx.derham_sequence import DeRhamSequence
from mrx.mappings import toroid_map
from mrx.nullspace import _n_vectors, compute_nullspaces_iterative, get_nullspace
from mrx.operators import (
    assemble_incidence_operators,
    assemble_projection_operators,
    assemble_tensor_laplacian_preconditioner,
    assemble_tensor_mass_preconditioner,
)
from mrx.preconditioners import (
    MassPreconditionerSpec,
    SaddlePointPreconditionerSpec,
    SchurPreconditionerSpec,
)
from mrx.quadrature import evaluate_at_xq

jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------------------
# Problem constants
# ---------------------------------------------------------------------------
types = ("clamped", "periodic", "periodic")
π = jnp.pi
BETTI = (1, 1, 0, 0)


# ---------------------------------------------------------------------------
# Source / exact-solution functions
# ---------------------------------------------------------------------------
# --- Generator A: u = cos(2πζ)  (k=0 NBC, k=3 DBC) ----------------------
def u_cos(x):
    """Scalar exact solution u = cos(2πζ)."""
    return jnp.cos(2 * π * x[2]) * jnp.ones(1)


def make_f0_cos(a: float):
    """Scalar source f₀ = -Δ cos(2πζ) = cos(2πζ)/R²  (k=0 NBC, k=3 DBC)."""
    def f(x):
        r, chi, z = x
        R = 1.0 + a * r * jnp.cos(2 * π * chi)
        return jnp.cos(2 * π * z) / R**2 * jnp.ones(1)
    return f


# --- Generator B: u = cos(πr²/2)  (k=0 DBC, k=3 NBC) ---------------------
# Smooth (non-polynomial) radial profile that still vanishes on the wall
# (cos(π/2) = 0 at r=1), so it is not exactly representable by the spline basis.
def u_par(x):
    """Scalar exact solution u = cos(πr²/2)  (vanishes on the wall r=1)."""
    return jnp.cos(0.5 * π * x[0]**2) * jnp.ones(1)


def make_f0_par(a: float):
    """Scalar source f₀ = -Δcos(πr²/2)  (k=0 DBC, k=3 NBC).

    f₀ = (2π s + π²r² c)/ε² + π r s cos(2πχ)/(ε R),  s = sin(πr²/2), c = cos(πr²/2).
    """
    def f(x):
        r, chi, z = x
        R = 1.0 + a * r * jnp.cos(2 * π * chi)
        s = jnp.sin(0.5 * π * r**2)
        c = jnp.cos(0.5 * π * r**2)
        return ((2 * π * s + π**2 * r**2 * c) / a**2
                + π * r * s * jnp.cos(2 * π * chi) / (a * R)) * jnp.ones(1)
    return f


# --- k=1 source: decoupled toroidal field --------------------------------
# ω₁ = cos(2πζ) dζ is closed (dω₁ = 0, curl-free), satisfies the natural BC
# u·n = 0 (only the ζ-covariant component is nonzero), and is orthogonal to the
# harmonic 1-form dζ because cos(2πζ) has zero mean.  Since curl·curl ω₁ = 0,
# f₁ = L₁ω₁ = grad σ with σ = -div ω₁ = sin(2πζ)/(2π R²).
# See docs/manufactured_solutions.md.
def _hodge_star_1to2_ref(a, alpha_cov, x):
    """All-positive Hodge star ⋆: Ω¹ → Ω² in ref proxy slot order (χζ, rζ, rχ).

    Cyclic convention: (J/g_rr, J/g_χχ, J/g_ζζ) = (4π²rR, R/r, ε²r/R), all positive.
    """
    r, chi, z = x
    R = 1.0 + a * r * jnp.cos(2 * π * chi)
    return jnp.array([
        4.0 * π**2 * r * R * alpha_cov[0],
        R / r * alpha_cov[1],
        a**2 * r / R * alpha_cov[2],
    ])


def _sigma1(a):
    """σ = δ₁ω₁ = -div ω₁ = sin(2πζ)/(2π R²)  for ω₁ = cos(2πζ) dζ."""
    def s(x):
        r, chi, z = x
        R = 1.0 + a * r * jnp.cos(2 * π * chi)
        return jnp.sin(2 * π * z) / (2 * π * R**2)
    return s


def _f1_cov_nbc(a: float):
    """Covariant 1-form source f₁ = grad σ (σ = -div ω₁), since curl·curl ω₁ = 0."""
    return jax.jacfwd(_sigma1(a))


def make_f1_ref(a: float, F):
    """f₁ contravariant ref = G⁻¹ (grad σ).

    ``load`` pairs the covariant 1-form basis Λ¹ with the field via a plain dot
    product (no metric), while the weak-form RHS is (f, Λ_i) = ∫ f_cov·G⁻¹·Λ_i J
    (M1 = ∫ Λ¹·G⁻¹·Λ¹ J).  Hence the load argument is the raised (contravariant)
    source G⁻¹ f_cov.
    """
    DF = jax.jacfwd(F)
    f_cov = _f1_cov_nbc(a)
    def f(x):
        J_DF = DF(x)
        G = J_DF.T @ J_DF
        return jnp.linalg.solve(G, f_cov(x))
    return f


def make_f1_phys(a: float, F):
    """f₁_phys = DF @ f₁_contra (physical Cartesian vector)."""
    DF = jax.jacfwd(F)
    f1r = make_f1_ref(a, F)
    def f(x):
        return DF(x) @ f1r(x)
    return f


# --- k=1 DBC source: mixed radial-zeta toroidal field ----------------------
# ω = φ(r) cos(2πζ) dζ with φ(r)=cos(πr²/2).  This is NOT divergence-free in the
# interior, but both essential k=1 DBC conditions still hold because φ(1)=0:
#   u×n = 0 at r=1, and σ = -div ω = φ(r) sin(2πζ)/(2πR²) also vanishes at r=1.
# The positive Hodge-Laplacian source is the full f₁ = dσ + δdω, and now has
# non-zero r, χ, and ζ covariant components (a less symmetry-protected test).
def _f1_cov_dbc(a: float):
    """Covariant 1-form source f₁ = dσ + δdω for ω = cos(πr²/2)cos(2πζ) dζ.

    With s(r)=sin(πr²/2), c(r)=cos(πr²/2), Z=cos(2πζ), S=sin(2πζ),
    Cχ=cos(2πχ), Sχ=sin(2πχ), R=1+εrCχ,

      f₁_r = -ε Cχ c(r) S / (π R³)
      f₁_χ =  2ε r Sχ c(r) S / R³
      f₁_ζ =  Z [ c(r)/R² + (2π s(r) + π²r² c(r))/ε² - π r s(r) Cχ/(ε R) ]

    See docs/manufactured_solutions.md appendix for the derivation.
    """
    def f(x):
        r, chi, z = x
        R = 1.0 + a * r * jnp.cos(2 * π * chi)
        s = jnp.sin(0.5 * π * r**2)
        c = jnp.cos(0.5 * π * r**2)
        z_cos = jnp.cos(2 * π * z)
        z_sin = jnp.sin(2 * π * z)
        chi_cos = jnp.cos(2 * π * chi)
        chi_sin = jnp.sin(2 * π * chi)
        fr = -a * chi_cos * c * z_sin / (π * R**3)
        fchi = 2.0 * a * r * chi_sin * c * z_sin / (R**3)
        fz = z_cos * (c / R**2
                      + (2.0 * π * s + π**2 * r**2 * c) / a**2
                      - π * r * s * chi_cos / (a * R))
        return jnp.array([fr, fchi, fz])
    return f


def make_f1_dbc_ref(a: float, F):
    """f₁ contravariant ref = G⁻¹ (δd ω)  (raised source, see make_f1_ref)."""
    DF = jax.jacfwd(F)
    f_cov = _f1_cov_dbc(a)
    def f(x):
        J_DF = DF(x)
        G = J_DF.T @ J_DF
        return jnp.linalg.solve(G, f_cov(x))
    return f


def make_f1_dbc_phys(a: float, F):
    """f₁_phys = DF @ f₁_contra (physical Cartesian vector)."""
    DF = jax.jacfwd(F)
    f1r = make_f1_dbc_ref(a, F)
    def f(x):
        return DF(x) @ f1r(x)
    return f


# --- k=2 source: ⋆f₁ (Hodge dual of the k=1 field) -------------------------
def make_f2_ref(a: float, F):
    """f₂ = L₂ω₂ = ⋆(L₁ω₁) = ⋆f₁ as reference 2-form proxy (χζ, rζ, rχ).

    ω₂ = ⋆ω₁ is the Hodge dual of the k=1 toroidal field; it is co-closed
    (δ₂ω₂ = 0), has zero normal trace (slot χζ = 0, so u·n = 0 — the essential
    k=2 DBC), and is orthogonal to the harmonic 2-form because ⋆ is an isometry.

    ``load`` for k=2 has measure w (no J) while M2 = ∫ Λ²·G·Λ²/J, so the load
    argument is the scaled source (G/J)·(⋆f₁ proxy).
    """
    DF = jax.jacfwd(F)
    f_cov = _f1_cov_nbc(a)
    def f(x):
        proxy = _hodge_star_1to2_ref(a, f_cov(x), x)
        J_DF = DF(x)
        G = J_DF.T @ J_DF
        return (G @ proxy) / jnp.linalg.det(J_DF)
    return f


def make_f2_phys(a: float, F):
    """f₂_phys = DF⁻ᵀ @ f₂_ref (physical proxy vector)."""
    DF = jax.jacfwd(F)
    f2r = make_f2_ref(a, F)
    def f(x):
        return jnp.linalg.solve(DF(x).T, f2r(x))
    return f


# --- k=2 NBC source: ⋆f₁ (Hodge dual of the k=1 DBC field, generator 4) -----
# ω₂ = ⋆ω = ⋆(cos(πr²/2)cos(2πζ) dζ) is the Hodge dual of the mixed k=1 DBC field.
# It is no longer closed in the interior, but the boundary conditions still hold:
# the tangential trace vanishes at the wall because the rχ slot carries a factor
# cos(πr²/2), and the natural div condition is imposed through the solved PDE, not
# by exact closure of ω₂.  The source is the strong-form dual f₂ = ⋆f₁.
def make_f2_nbc_ref(a: float, F):
    """f₂ load argument (k=2 NBC) = (G/J)·(⋆f₁ proxy), dual of the k=1 DBC source."""
    DF = jax.jacfwd(F)
    f_cov = _f1_cov_dbc(a)
    def f(x):
        proxy = _hodge_star_1to2_ref(a, f_cov(x), x)
        J_DF = DF(x)
        G = J_DF.T @ J_DF
        return (G @ proxy) / jnp.linalg.det(J_DF)
    return f


def make_f2_nbc_phys(a: float, F):
    """f₂_phys = DF⁻ᵀ @ f₂_ref (physical proxy vector)."""
    DF = jax.jacfwd(F)
    f2r = make_f2_nbc_ref(a, F)
    def f(x):
        return jnp.linalg.solve(DF(x).T, f2r(x))
    return f


# --- k=3 source: coefficient in A dr∧dχ∧dζ (ref frame) -------------------
def make_f3_ref(f0_fn, F):
    """A = f₀·J,  coefficient in f₀ J dr∧dχ∧dζ, for the given scalar source f₀."""
    DF = jax.jacfwd(F)
    def f(x):
        J = jnp.linalg.det(DF(x))
        return f0_fn(x) * J
    return f


# ---------------------------------------------------------------------------
# Exact solutions for error (1-form / 2-form fields; scalars are u_cos/u_par)
# ---------------------------------------------------------------------------
def v1_exact_ref(x):
    """ω₁ ref covariant: (0, 0, cos 2πζ)  — the toroidal field cos(2πζ) dζ."""
    return jnp.array([0.0, 0.0, jnp.cos(2 * π * x[2])])


def v1_dbc_exact_ref(x):
    """ω ref covariant: (0, 0, cos(πr²/2)cos(2πζ))  — the mixed k=1 DBC field."""
    return jnp.array([0.0, 0.0, jnp.cos(0.5 * π * x[0]**2) * jnp.cos(2 * π * x[2])])


def make_w2_exact_ref(a: float):
    """ω₂ = ⋆ω₁ ref proxy (χζ, rζ, rχ): only slot rχ is non-zero.

    (⋆ω₁)_{rχ} = (J/g_ζζ) cos 2πζ = (ε²r/R) cos 2πζ.
    """
    def w(x):
        r, chi, z = x
        R = 1.0 + a * r * jnp.cos(2 * π * chi)
        return jnp.array([0.0, 0.0, a**2 * r * jnp.cos(2 * π * z) / R])
    return w


def make_w2_nbc_exact_ref(a: float):
    """ω₂ = ⋆(cos(πr²/2)cos(2πζ) dζ) ref proxy (χζ, rζ, rχ): only slot rχ is non-zero.

    (⋆ω)_{rχ} = (J/g_ζζ) cos(πr²/2)cos(2πζ) = (ε²r/R) cos(πr²/2)cos(2πζ).
    """
    def w(x):
        r, chi, z = x
        R = 1.0 + a * r * jnp.cos(2 * π * chi)
        return jnp.array([0.0, 0.0, a**2 * r * jnp.cos(0.5 * π * r**2) * jnp.cos(2 * π * z) / R])
    return w


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _log(msg: str):
    """Print with a timestamp and flush immediately."""
    import sys
    ts = time.strftime("%H:%M:%S")
    print(f"  [{ts}] {msg}", flush=True)
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Case registry
# ---------------------------------------------------------------------------
# Each enabled (k, dirichlet) case is solved with a shared manufactured
# solution.  The duality ⋆: k ↔ (3-k), NBC ↔ DBC groups them into generators
# (see module docstring).  All eight (k, BC) cases are enabled.
CASES = [
    (0, False),   # cos(2πζ)            — null: constant
    (0, True),    # cos(πr²/2)
    (1, False),   # cos(2πζ) dζ         — null: toroidal 1-form
    (1, True),    # cos(πr²/2)cos(2πζ) dζ       — mixed generator 4
    (2, False),   # ⋆(cos(πr²/2)cos(2πζ) dζ)    — k=2 NBC (dual of generator 4)
    (2, True),    # ⋆(cos(2πζ) dζ)      — null: toroidal 2-form
    (3, False),   # cos(πr²/2)
    (3, True),    # cos(2πζ)            — null: constant 3-form
]


def _case_tag(k: int, dirichlet: bool) -> str:
    return f"k{k}_{'dbc' if dirichlet else 'nbc'}"


def _case_specs(epsilon: float, F):
    """Return {(k, dirichlet): {src_ref, src_phys, exact}} for the enabled cases.

    src_ref / src_phys are the reference- and physical-frame load callables;
    ``exact`` is the analytic field used for the L² error (ref-frame covariant
    for k=1, physical scalar for k=0/3).
    """
    f0_cos = make_f0_cos(epsilon)
    f0_par = make_f0_par(epsilon)
    return {
        (0, False): dict(src_ref=f0_cos, src_phys=f0_cos, exact=u_cos),
        (0, True):  dict(src_ref=f0_par, src_phys=f0_par, exact=u_par),
        (1, False): dict(src_ref=make_f1_ref(epsilon, F),
                         src_phys=make_f1_phys(epsilon, F),
                         exact=v1_exact_ref),
        (1, True):  dict(src_ref=make_f1_dbc_ref(epsilon, F),
                         src_phys=make_f1_dbc_phys(epsilon, F),
                         exact=v1_dbc_exact_ref),
        (2, False): dict(src_ref=make_f2_nbc_ref(epsilon, F),
                         src_phys=make_f2_nbc_phys(epsilon, F),
                         exact=make_w2_nbc_exact_ref(epsilon)),
        (2, True):  dict(src_ref=make_f2_ref(epsilon, F),
                         src_phys=make_f2_phys(epsilon, F),
                         exact=make_w2_exact_ref(epsilon)),
        (3, False): dict(src_ref=make_f3_ref(f0_par, F), src_phys=f0_par, exact=u_par),
        (3, True):  dict(src_ref=make_f3_ref(f0_cos, F), src_phys=f0_cos, exact=u_cos),
    }


def _compute_error(seq, k: int, dirichlet: bool, u_hat, exact_fn, quad_shape):
    """Relative L² error of the discrete solution against the manufactured field."""
    comp_info, comp_shapes = seq._form_comp_info(k)
    eT = getattr(seq, f"e{k}_dbc_T" if dirichlet else f"e{k}_T")
    if k == 0:
        u_h = evaluate_at_xq(eT @ u_hat, comp_info, comp_shapes, quad_shape, 1)
        u_ex = jax.vmap(exact_fn)(seq.quad.x)
        diff = u_h - u_ex
        num = jnp.einsum("ik,ik,i,i->", diff, diff, seq.jacobian_j, seq.quad.w)
        den = jnp.einsum("ik,ik,i,i->", u_ex, u_ex, seq.jacobian_j, seq.quad.w)
    elif k == 1:
        v_h = evaluate_at_xq(eT @ u_hat, comp_info, comp_shapes, quad_shape, 3)
        DF_xq = jax.vmap(jax.jacfwd(seq.map))(seq.quad.x)
        v_h_phys = jnp.einsum('qij,qjk,qk->qi', DF_xq, seq.metric_inv_jkl, v_h)
        v_ex = jax.vmap(exact_fn)(seq.quad.x)
        v_ex_phys = jnp.einsum('qij,qjk,qk->qi', DF_xq, seq.metric_inv_jkl, v_ex)
        diff = v_h_phys - v_ex_phys
        num = jnp.einsum('qi,qi,q,q->', diff, diff, seq.jacobian_j, seq.quad.w)
        den = jnp.einsum('qi,qi,q,q->', v_ex_phys, v_ex_phys, seq.jacobian_j, seq.quad.w)
    elif k == 2:
        w_h = evaluate_at_xq(eT @ u_hat, comp_info, comp_shapes, quad_shape, 3)
        w_ex = jax.vmap(exact_fn)(seq.quad.x)
        g_inv = seq.metric_inv_jkl
        weights2 = jnp.stack([
            g_inv[:, 1, 1] * g_inv[:, 2, 2],
            g_inv[:, 0, 0] * g_inv[:, 2, 2],
            g_inv[:, 0, 0] * g_inv[:, 1, 1],
        ], axis=1)
        diff = w_h - w_ex
        num = jnp.einsum('qi,qi,qi,q->', diff, diff, weights2, seq.jacobian_j * seq.quad.w)
        den = jnp.einsum('qi,qi,qi,q->', w_ex, w_ex, weights2, seq.jacobian_j * seq.quad.w)
    else:  # k == 3
        u_h = evaluate_at_xq(eT @ u_hat, comp_info, comp_shapes, quad_shape, 1)
        u_h_phys = u_h / seq.jacobian_j[:, None]
        u_ex = jax.vmap(exact_fn)(seq.quad.x)
        diff = u_h_phys - u_ex
        num = jnp.einsum("ik,ik,i,i->", diff, diff, seq.jacobian_j, seq.quad.w)
        den = jnp.einsum("ik,ik,i,i->", u_ex, u_ex, seq.jacobian_j, seq.quad.w)
    return float(jnp.sqrt(num / den))


def _null_diag(seq, null_info, k: int, dirichlet: bool):
    """Nullspace diagnostics for one (k, dirichlet) case (handles dim-0 cases)."""
    n_vec = _n_vectors(BETTI, k, dirichlet)
    if n_vec == 0:
        return {
            "null_dim": 0,
            "null_iters": 0,
            "null_final_residual": float("nan"),
            "null_Lh_norm": 0.0,
            "null_curl_norm": 0.0,
            "null_div_norm": 0.0,
        }
    h = get_nullspace(seq.get_operators(), k, dirichlet)[0]
    residual = float(jnp.linalg.norm(seq.apply_laplacian(h, k, dirichlet=dirichlet)))
    iters_res = null_info.get((k, dirichlet), [(0, float("nan"))])[0]
    curl_norm = (float(jnp.linalg.norm(seq.apply_derivative_matrix(
        h, k, dirichlet_in=dirichlet, dirichlet_out=dirichlet))) if k < 3 else 0.0)
    div_norm = (float(jnp.linalg.norm(seq.apply_derivative_matrix(
        h, k - 1, dirichlet_in=dirichlet, dirichlet_out=dirichlet, transpose=True)))
        if k > 0 else 0.0)
    return {
        "null_dim": n_vec,
        "null_iters": iters_res[0],
        "null_final_residual": iters_res[1],
        "null_Lh_norm": residual,
        "null_curl_norm": curl_norm,
        "null_div_norm": div_norm,
    }


def _solve_case(seq, k: int, dirichlet: bool, spec, quad_shape, timings,
                saddle_preconditioner):
    """Load, solve (compile + exec passes), and compute the L² error for one case."""
    tag = _case_tag(k, dirichlet)
    _log(f"--- {tag} (k={k}, dirichlet={dirichlet}) ---")

    _log("  Assembling load vectors (ref + phys frame consistency check)...")
    b_ref = seq.load(spec["src_ref"], k, dirichlet=dirichlet, frame='ref')
    b_phys = seq.load(spec["src_phys"], k, dirichlet=dirichlet, frame='phys')
    load_frame_diff = float(jnp.linalg.norm(b_ref - b_phys))
    _log(f"  ||b_ref - b_phys|| = {load_frame_diff:.3e}")

    _log("  Solving (compile pass)...")
    t0 = time.perf_counter()
    preconditioner = 'auto' if k == 0 else saddle_preconditioner
    u_hat, info = seq.apply_inverse_laplacian(
        b_ref, k, dirichlet=dirichlet, preconditioner=preconditioner,
        return_info=True)
    jax.block_until_ready(u_hat)
    timings[f"solve_{tag}_compile"] = time.perf_counter() - t0
    _log(f"  Compile pass done ({timings[f'solve_{tag}_compile']:.2f}s)")

    _log("  Solving (exec pass)...")
    t0 = time.perf_counter()
    u_hat, info = seq.apply_inverse_laplacian(
        b_ref, k, dirichlet=dirichlet, preconditioner=preconditioner,
        return_info=True)
    jax.block_until_ready(u_hat)
    timings[f"solve_{tag}_exec"] = time.perf_counter() - t0
    _log(f"  Solve done: iters={abs(int(info))} converged={int(info) < 0}"
         f" ({timings[f'solve_{tag}_exec']:.2f}s)")

    _log("  Computing L2 error...")
    error = _compute_error(seq, k, dirichlet, u_hat, spec["exact"], quad_shape)
    _log(f"  Relative L2 error = {error:.6e}")

    return {
        "k": k,
        "dirichlet": dirichlet,
        "error": error,
        "iters": abs(int(info)),
        "converged": int(info) < 0,
        "load_frame_diff": load_frame_diff,
    }


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------
def compute_all_k(n: int, p: int, epsilon: float,
                  cg_tol: float, cg_maxiter: int,
                  quad_order, quad_order_offset: int):
    timings = {}
    ns = (n, 2 * n, n)
    ps = (p, p, p)
    q = 2 * p + quad_order_offset if quad_order is None else quad_order

    F = toroid_map(epsilon=epsilon)
    cp_kwargs = {"maxiter": 100, "tol": 1e-9, "ridge": 1e-12}
    saddle_preconditioner = SaddlePointPreconditionerSpec(
        mass=MassPreconditionerSpec(kind='tensor', surgery_schur=True),
        schur=SchurPreconditionerSpec(
            inner=MassPreconditionerSpec(kind='tensor'),
            outer=MassPreconditionerSpec(
                kind='richardson',
                steps=4,
                power_iterations=30,
                damping_safety=0.8,
            ),
        ),
        coupled=False,
    )

    # --- Sequence setup ------------------------------------------------
    _log(f"Building DeRhamSequence: ns={ns} ps={ps} q={q}")
    t0 = time.perf_counter()
    seq = DeRhamSequence(
        ns, ps, q, types, polar=True,
        tol=cg_tol, maxiter=cg_maxiter,
        betti_numbers=BETTI,
    )
    seq.set_map(F)
    timings["init"] = time.perf_counter() - t0
    _log(f"  DeRhamSequence built: n0={seq.n0} n1={seq.n1} n2={seq.n2} n3={seq.n3}"
         f"  n0_dbc={seq.n0_dbc} n1_dbc={seq.n1_dbc} n2_dbc={seq.n2_dbc} n3_dbc={seq.n3_dbc}"
         f"  ({timings['init']:.2f}s)")

    _log("Evaluating 1D basis functions and geometry...")
    t0 = time.perf_counter()
    seq.evaluate_1d()
    timings["evaluate_1d"] = time.perf_counter() - t0
    _log(f"  evaluate_1d done ({timings['evaluate_1d']:.2f}s)")

    quad_shape = (seq.quad.ny, seq.quad.nx, seq.quad.nz)

    # --- Assembly -------------------------------------------------------
    _log("Assembly: incidence + projection operators...")
    t0 = time.perf_counter()
    ops = assemble_incidence_operators(seq)
    ops = assemble_projection_operators(seq, operators=ops)
    _log("  Assembling tensor mass preconditioner (k=0,1,2,3)...")
    ops = assemble_tensor_mass_preconditioner(seq, ops, ks=(0, 1, 2, 3), rank=1, cp_kwargs=cp_kwargs)
    _log("  Assembling tensor Hodge-Laplacian preconditioner (k=0)...")
    ops = assemble_tensor_laplacian_preconditioner(seq, ops, ks=(0,), rank=1, cp_kwargs=cp_kwargs)
    _log("  Skipping Schur-Jacobi diagonal probing (using schur.outer=richardson)...")
    ops = seq.set_operators(ops)
    jax.block_until_ready(ops)
    timings["assembly"] = time.perf_counter() - t0
    _log(f"  Assembly done ({timings['assembly']:.2f}s)")

    # --- Nullspace (iterative for all 4 non-trivial pairs) ---------------
    _log("Computing nullspaces iteratively (k=0 NBC, k=1 NBC, k=2 DBC, k=3 DBC)...")
    t0 = time.perf_counter()
    nullspace_shift = 1.0e-3 / (float(n) ** 2)
    # 1e-8 is sufficient for deflation quality; the analytic initial guesses
    # for k1_nbc (~1.2e-9) and the iterated k2_dbc/k3_dbc modes (~1e-10 to
    # 5e-10) all satisfy this threshold, so most modes will be accepted
    # immediately without burning iterations at maxiter.
    nullspace_abs_tol = 1.0e-8
    # inner_tol only needs to be tight enough for outer iterations to make
    # progress; for shift-and-invert power iteration 1e-6 is sufficient.
    nullspace_inner_tol = 1.0e-6
    _log(f"  Nullspace shift eps={nullspace_shift:.3e} (scaled as 1e-3 / n^2)")
    _log(f"  Nullspace tolerances: abs_tol={nullspace_abs_tol:.1e}, inner_tol={nullspace_inner_tol:.1e}")
    ops, null_info = compute_nullspaces_iterative(seq, seq.get_operators(), BETTI,
                                                   eps=nullspace_shift,
                                                   abs_tol=nullspace_abs_tol,
                                                   inner_tol=nullspace_inner_tol,
                                                   maxiter=100)
    ops = seq.set_operators(ops)
    jax.block_until_ready(ops)
    timings["nullspace"] = time.perf_counter() - t0
    _log(f"  Nullspace iteration done ({timings['nullspace']:.2f}s)")

    # --- Nullspace diagnostics + per-case solves -------------------------
    specs = _case_specs(epsilon, F)
    results = {}
    for k, dirichlet in CASES:
        res = _solve_case(
            seq,
            k,
            dirichlet,
            specs[(k, dirichlet)],
            quad_shape,
            timings,
            saddle_preconditioner,
        )
        nd = _null_diag(seq, null_info, k, dirichlet)
        _log(f"  null {_case_tag(k, dirichlet)}: dim={nd['null_dim']}"
             f" iters={nd['null_iters']} ||Lh||={nd['null_Lh_norm']:.3e}"
             f" ||curl||={nd['null_curl_norm']:.3e} ||div||={nd['null_div_norm']:.3e}")
        results[_case_tag(k, dirichlet)] = {**res, **nd}

    timings["TOTAL"] = sum(timings.values())
    return {
        "n": n, "p": p, "q": q,
        "cases": [_case_tag(k, d) for k, d in CASES],
        "timings": timings,
        **results,
    }


# ---------------------------------------------------------------------------
# Hydra entry point
# ---------------------------------------------------------------------------
@hydra.main(config_path="../../conf", config_name="config_poisson_test", version_base=None)
def main(cfg: DictConfig):
    print(f"x64 enabled: {jax.config.jax_enable_x64}")
    ns = [cfg.n] if isinstance(cfg.n, int) else list(cfg.n)
    p = cfg.p
    mrx.MAP_BATCH_SIZE_INNER = cfg.map_batch_size_inner
    mrx.MAP_BATCH_SIZE_OUTER = cfg.map_batch_size_outer
    case_tags = [_case_tag(k, d) for k, d in CASES]
    print(f"Hodge–Laplacian convergence | n={ns} p={p} ε={cfg.epsilon}")
    print(f"Cases: {', '.join(case_tags)}")
    print(f"JAX devices: {jax.devices()}")

    output_dir = HydraConfig.get().runtime.output_dir
    outfile = os.path.join(output_dir, "result.json")

    all_results = []
    for n in ns:
        print(f"\n{'='*68}\n  n={n}, p={p}\n{'='*68}")
        result = compute_all_k(
            n, p, cfg.epsilon, cfg.cg_tol, cfg.cg_maxiter,
            cfg.quad_order, cfg.quad_order_offset,
        )
        all_results.append(result)

        print(f"\n  --- Timings ---")
        for label, dt in result["timings"].items():
            print(f"  {label:.<40s} {dt:8.3f}s")

        print(f"\n  --- Frame consistency (||b_ref - b_phys||) ---")
        for tag in case_tags:
            print(f"  {tag}: {result[tag]['load_frame_diff']:.3e}")

        print(f"\n  --- Nullspace diagnostics ---")
        hdr = (f"  {'case':>8s}  {'dim':>3s}  {'iters':>6s}  {'resid':>10s}"
               f"  {'||Lh||':>10s}  {'||curl||':>10s}  {'||div||':>10s}")
        print(hdr)
        for tag in case_tags:
            r = result[tag]
            print(f"  {tag:>8s}  {r['null_dim']:3d}  {r['null_iters']:6d}"
                  f"  {r['null_final_residual']:10.3e}  {r['null_Lh_norm']:10.3e}"
                  f"  {r['null_curl_norm']:10.3e}  {r['null_div_norm']:10.3e}")

        print(f"\n  --- Convergence ---")
        hdr2 = f"  {'case':>8s}  {'error':>12s}  {'iters':>6s}  {'conv':>5s}"
        print(hdr2)
        for tag in case_tags:
            r = result[tag]
            print(f"  {tag:>8s}  {r['error']:12.6e}  {r['iters']:6d}  {str(r['converged']):>5s}")

        with open(outfile, "w") as fh:
            json.dump(all_results, fh, indent=2)
        print(f"\n  Saved → {outfile}")

    print(f"\n{'='*68}\n  Summary  p={p}\n{'='*68}")
    header = "  " + f"{'n':>5s}" + "".join(f"  {tag + ' err':>14s}" for tag in case_tags)
    print(header)
    for r in all_results:
        row = "  " + f"{r['n']:5d}" + "".join(f"  {r[tag]['error']:14.6e}" for tag in case_tags)
        print(row)


if __name__ == "__main__":
    main()

