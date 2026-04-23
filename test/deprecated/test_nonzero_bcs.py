"""Tests for Dirichlet BC enforcement via lift / stiffness-correction.

k=0  Manufactured solution:  u = x²+y²,  source f = -Δu = -4.
     BC u|_{∂Ω} = x²+y² enforced as a nonzero Dirichlet condition.

k=1  Manufactured solution:  E = (-y, x, 0),  curl E = (0,0,2), div E = 0.
     Source L_1 E = curl curl E + grad div E = 0.
     Tangential trace E_t|_{∂Ω} enforced as a nonzero H(curl) BC.

All test resolutions share a single module-scoped `seqs` fixture to avoid
redundant assembly across both test classes.
"""

import jax
import jax.numpy as jnp
import pytest

from mrx.derham_sequence import DeRhamSequence
from mrx.differential_forms import DiscreteFunction, Pushforward
from mrx.mappings import toroid_map
from mrx.utils import inv33

jax.config.update("jax_enable_x64", True)

EPSILON = 1.0 / 3
R0 = 1.0
types = ("clamped", "periodic", "periodic")


NS = (4, 6, 8)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_seq(n, p=2):
    """Build a fully assembled DeRham sequence (k=0 and k=1 matrices)."""
    F = toroid_map(epsilon=EPSILON, R0=R0)
    seq = DeRhamSequence(
        (n, n, n), (p, p, p), 2 * p, types, F,
        polar=True, tol=1e-12, maxiter=1000,
    )
    seq.evaluate_1d()
    seq.assemble_mass_matrix(0)
    seq.assemble_mass_matrix(1)
    seq.assemble_hodge_laplacian(0)
    seq.assemble_derivative_matrix(0)
    seq.assemble_hodge_laplacian(1)
    # solid torus: betti=(1,1,0,0), so null_0_dbc, null_1_dbc are already empty.
    return seq


@pytest.fixture(scope="module")
def seqs():
    """Build one sequence per resolution, shared across all tests in this module."""
    return {n: _build_seq(n) for n in NS}


def _u_exact_func(F):
    """Return u_exact as a function of logical coordinates ξ."""
    def _f(xi):
        xyz = F(xi)
        return jnp.array([xyz[0] ** 2 + xyz[1] ** 2])
    return _f


def _f_source(xi):
    """Source term: -Δu = -4 (constant)."""
    return jnp.array([-4.0])


def _solve(seq):
    """Solve -Δu = -4 with u = x^2+y^2 on ∂Ω.

    Returns
    -------
    u_0   : DOF vector in the DBC (interior) subspace,  shape (n0_dbc,)
    g_bc  : DOF vector in the BC subspace,               shape (n0_bc,)
    """
    u_exact = _u_exact_func(seq.map)

    # 1. BC DOF values: L2-project the exact solution onto the full 0-form
    #    space, then select the BC component.
    u_exact_load_full = seq.p0(u_exact)             # (n0,)  load vector
    u_exact_full = seq.apply_inverse_mass_matrix(
        u_exact_load_full, 0, dirichlet=False)       # (n0,)  DOF values
    # e0_bc selects from the full spline space (basis_0.n,); lift from (n0,)
    g_bc = seq.e0_bc @ (seq.e0_T @ u_exact_full)    # (n0_bc,)

    # 2. Load vector for the DBC-space Poisson problem: ∫ f φ_i dx.
    rhs = seq.p0_dbc(_f_source)                      # (n0_dbc,)

    # 3. Stiffness correction for non-zero BC:
    #    rhs_corrected_i = rhs_i - Σ_j S_{ij}^{dbc,bc} g_j
    correction = seq.e0_dbc @ (seq.grad_grad_sp @ (seq.e0_bc_T @ g_bc))
    rhs -= correction

    # 4. Solve.
    u_0 = seq.apply_inverse_hodge_laplacian(
        rhs, 0, dirichlet=True)  # (n0_dbc,)

    return u_0, g_bc


def _relative_l2_error(seq, u_0, g_bc):
    """Relative L2 error ‖u_h - u_exact‖ / ‖u_exact‖ over the full domain."""
    u_exact = _u_exact_func(seq.map)
    u_h_int = DiscreteFunction(u_0,   seq.basis_0, seq.e0_dbc)
    u_h_bc = DiscreteFunction(g_bc,  seq.basis_0, seq.e0_bc)

    def u_h(xi):
        return u_h_int(xi) + u_h_bc(xi)   # shape (1,) for k=0

    diff_vals = jax.lax.map(
        lambda xi: u_exact(xi) - u_h(xi), seq.quad.x, batch_size=0)
    exact_vals = jax.vmap(u_exact)(seq.quad.x)

    wJ = seq.jacobian_j * seq.quad.w
    L2_diff = jnp.einsum("ik,ik,i->", diff_vals,  diff_vals,  wJ)
    L2_exact = jnp.einsum("ik,ik,i->", exact_vals, exact_vals, wJ)
    return float(jnp.sqrt(L2_diff / L2_exact))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPoissonNonzeroDBC:
    """Poisson with nonzero Dirichlet BC: correctness and convergence."""

    def test_small_error(self, seqs):
        """Relative L2 error < 10 % at n=4, p=2."""
        u_0, g_bc = _solve(seqs[4])
        rel_err = _relative_l2_error(seqs[4], u_0, g_bc)
        print(f"\n  [n=4, p=2] relative L2 error = {rel_err:.4e}")
        assert rel_err < 3e-2, f"Relative L2 error too large: {rel_err:.2e}"

    def test_convergence(self, seqs):
        """L2 error decreases as resolution goes from n=4 to n=6 to n=8."""
        errors = []
        for n in NS:
            u_0, g_bc = _solve(seqs[n])
            errors.append(_relative_l2_error(seqs[n], u_0, g_bc))
            print(f"\n  [n={n}, p=2] relative L2 error = {errors[-1]:.4e}")
        for i in range(1, len(errors)):
            if errors[i] > 0 and errors[i-1] > 0:
                rate = jnp.log(errors[i-1] / errors[i]) / \
                    jnp.log(NS[i] / NS[i-1])
                print(
                    f"  convergence rate n={NS[i-1]}→{NS[i]}: {float(rate):.2f}")
        assert errors[1] < errors[0] and errors[2] < errors[1], (
            f"Error did not decrease: {errors[0]:.4e} → {errors[1]:.4e} → {errors[2]:.4e}"
        )
        assert errors[1] < 4e-3, f"Relative L2 error at n=6 too large: {errors[1]:.4e}"
        assert errors[2] < 2e-3, f"Relative L2 error at n=8 too large: {errors[2]:.4e}"


# ---------------------------------------------------------------------------
# k=1 helpers
# ---------------------------------------------------------------------------

def _E_exact_phys(seq):
    """E = (-y, x, 0) in physical frame (azimuthal field about the z-axis).

    curl E = (0, 0, 2),  div E = 0  =>  L_1 E = curl curl E + grad div E = 0.
    """
    F = seq.map

    def _f(xi):
        xyz = F(xi)
        return jnp.array([-xyz[1], xyz[0], 0.0])
    return _f


def _solve_k1(seq):
    """Solve L_1 E = 0 with nonzero tangential-trace BC E = (-y, x, 0).

    Returns
    -------
    E_0   : interior DOFs, shape (n1_dbc,)
    g_bc  : BC DOFs,       shape (n1_bc,)
    """
    E_exact = _E_exact_phys(seq)

    # 1. BC DOF values via L2-projection of E_exact onto the full 1-form space.
    E_load_full = seq.p1(E_exact)                            # (n1,)
    E_full = seq.apply_inverse_mass_matrix(
        E_load_full, 1, dirichlet=False)                     # (n1,)
    g_bc = seq.e1_bc @ (seq.e1_T @ E_full)                  # (n1_bc,)

    # 2. RHS is zero (L_1 E = 0 exactly for this manufactured solution).
    rhs = jnp.zeros(seq.n1_dbc)

    # 3. Stiffness correction: subtract (L_1)_{dbc,bc} g_bc from RHS.
    #    L_1 = S_1 + D_0 M_0^{-1} D_0^T   where S_1 = curl-curl.

    # Curl-curl part: e1_dbc @ curl_curl_sp @ e1_bc_T @ g_bc
    cc_corr = seq.e1_dbc @ (seq.curl_curl_sp @ (seq.e1_bc_T @ g_bc))

    # Grad-div part: D_0 M_0^{-1} D_0^T applied to BC lift, projected to DBC
    Dt_g = seq.e0_dbc @ (seq.d0_sp_T @ (seq.e1_bc_T @ g_bc))
    MinvDt_g = seq.apply_inverse_mass_matrix(Dt_g, 0, dirichlet=True)
    gd_corr = seq.apply_derivative_matrix(
        MinvDt_g, 0, dirichlet_in=True, dirichlet_out=True)

    rhs = rhs - (cc_corr + gd_corr)

    # 4. Solve.
    E_0 = seq.apply_inverse_hodge_laplacian(rhs, 1, dirichlet=True)

    return E_0, g_bc


def _relative_l2_error_k1(seq, E_0, g_bc):
    """Relative L2 error in physical space."""
    E_exact = _E_exact_phys(seq)

    E_h_int = Pushforward(DiscreteFunction(
        E_0,  seq.basis_1, seq.e1_dbc), seq.map, 1)
    E_h_bc = Pushforward(DiscreteFunction(
        g_bc, seq.basis_1, seq.e1_bc), seq.map, 1)

    def E_h(x_hat):
        return E_h_int(x_hat) + E_h_bc(x_hat)

    diff_vals = jax.lax.map(
        lambda x_hat: E_exact(x_hat) - E_h(x_hat), seq.quad.x, batch_size=0)
    exact_vals = jax.vmap(E_exact)(seq.quad.x)

    wJ = seq.jacobian_j * seq.quad.w
    L2_diff = jnp.einsum("ik,ik,i->", diff_vals,  diff_vals,  wJ)
    L2_exact = jnp.einsum("ik,ik,i->", exact_vals, exact_vals, wJ)
    return float(jnp.sqrt(L2_diff / L2_exact))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPoissonNonzeroDBC_k1:
    """k=1 Hodge Laplacian (curl-curl + grad-div) with nonzero tangential-trace BC."""

    def test_small_error(self, seqs):
        """Relative L2 error < 10 % at n=4, p=2."""
        E_0, g_bc = _solve_k1(seqs[4])
        rel_err = _relative_l2_error_k1(seqs[4], E_0, g_bc)
        print(f"\n  [k=1, n=4, p=2] relative L2 error = {rel_err:.4e}")
        assert rel_err < 3e-2, f"Relative L2 error too large: {rel_err:.2e}"

    def test_convergence(self, seqs):
        """L2 error decreases as resolution goes from n=4 to n=6 to n=8."""
        errors = []
        for n in NS:
            E_0, g_bc = _solve_k1(seqs[n])
            errors.append(_relative_l2_error_k1(seqs[n], E_0, g_bc))
            print(
                f"\n  [k=1, n={n}, p=2] relative L2 error = {errors[-1]:.4e}")
        for i in range(1, len(errors)):
            if errors[i] > 0 and errors[i - 1] > 0:
                rate = jnp.log(errors[i - 1] / errors[i]) / \
                    jnp.log(NS[i] / NS[i - 1])
                print(
                    f"  convergence rate n={NS[i-1]}→{NS[i]}: {float(rate):.2f}")
        assert errors[1] < errors[0] and errors[2] < errors[1], (
            f"Error did not decrease: {errors[0]:.4e} → {errors[1]:.4e} → {errors[2]:.4e}"
        )
        assert errors[1] < 4e-3, f"Relative L2 error at n=6 too large: {errors[1]:.4e}"
        assert errors[2] < 2e-3, f"Relative L2 error at n=8 too large: {errors[2]:.4e}"
