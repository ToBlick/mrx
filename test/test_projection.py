# test_projection.py
"""L2-projection convergence tests on a toroid."""

import jax
import jax.numpy as jnp
import pytest

from mrx.derham_sequence import DeRhamSequence
from mrx.differential_forms import DiscreteFunction, Pushforward
from mrx.mappings import toroid_map
from mrx.utils import integrate_against_tp, inv33

jax.config.update("jax_enable_x64", True)

types = ("clamped", "periodic", "periodic")
NS = (4, 5, 6)


def _build_toroid_seq(n, p=2):
    a = 1 / 3
    F = toroid_map(epsilon=a)
    seq = DeRhamSequence((n, n, n), (p, p, p), 2*p, types, F,
                         polar=True, tol=1e-12, maxiter=1000)
    seq.evaluate_1d()
    for k in range(4):
        seq.assemble_mass_matrix_tp(k)
    return seq


# Module-scoped fixtures: one sequence per resolution, reused by all tests
@pytest.fixture(scope="module")
def seqs():
    return {n: _build_toroid_seq(n) for n in NS}


class TestProjectionConvergence:
    """L2-project known functions onto the FE space and check that
    the error is small and decreases with resolution."""

    # -- toy functions -------------------------------------------------------
    # k=0 / k=3 scalar (DBC-compatible: vanishes at r=0 and r=1)
    @staticmethod
    def _scalar_dbc(x):
        r, χ, z = x
        π = jnp.pi
        return r ** 2 * (1 - r) ** 2 * jnp.cos(2 * π * χ) * jnp.ones(1)

    # k=0 / k=3 scalar (no DBC requirement)
    @staticmethod
    def _scalar_free(x):
        r, χ, z = x
        π = jnp.pi
        return (1 + r * jnp.cos(2 * π * z)) * jnp.ones(1)

    # k=1 / k=2 vector (vanishes at r=0 and r=1, periodic)
    @staticmethod
    def _vector_dbc(x):
        r, χ, z = x
        π = jnp.pi
        return jnp.array([
            r * (1 - r) * jnp.cos(2 * π * z),
            r * (1 - r) * jnp.sin(2 * π * χ),
            r * (1 - r),
        ])

    # k=1 / k=2 vector (vanishes at r=0 only, periodic)
    @staticmethod
    def _vector_free(x):
        r, χ, z = x
        π = jnp.pi
        return jnp.array([
            r * jnp.cos(2 * π * z),
            r * jnp.sin(2 * π * χ),
            r * (1 - r),
        ])

    # -- helper --------------------------------------------------------------
    @staticmethod
    def _project_tp(seq, k, f, dirichlet):
        """L2-project *f* as a k-form using TP integrate_against."""
        F = seq.map
        DF = jax.jacfwd(F)
        quad_shape = (seq.quad.ny, seq.quad.nx, seq.quad.nz)
        comp_info, comp_shapes = seq._form_comp_info(k)
        es_dbc = [seq.e0_dbc, seq.e1_dbc, seq.e2_dbc, seq.e3_dbc]
        es = [seq.e0, seq.e1, seq.e2, seq.e3]
        e = es_dbc[k] if dirichlet else es[k]

        if k == 0:
            f_jk = jax.lax.map(f, seq.quad.x, batch_size=0)
            w_jk = f_jk * (seq.quad.w * seq.jacobian_j)[:, None]
        elif k == 1:
            def _v(x):
                return inv33(DF(x)) @ f(x)
            f_jk = jax.lax.map(_v, seq.quad.x, batch_size=0)
            w_jk = f_jk * (seq.quad.w * seq.jacobian_j)[:, None]
        elif k == 2:
            def _v(x):
                return DF(x).T @ f(x)
            f_jk = jax.lax.map(_v, seq.quad.x, batch_size=0)
            w_jk = f_jk * (seq.quad.w)[:, None]
        elif k == 3:
            f_jk = jax.lax.map(f, seq.quad.x, batch_size=0)
            w_jk = f_jk * (seq.quad.w)[:, None]
        else:
            raise ValueError(f"Invalid k: {k}")

        rhs = integrate_against_tp(w_jk, comp_info, comp_shapes, quad_shape)
        return e @ rhs

    @staticmethod
    def _project_and_error(seq, k, f, dirichlet):
        """L2-project *f* as a k-form, return relative L2 error."""
        F = seq.map
        bases = [seq.basis_0, seq.basis_1, seq.basis_2, seq.basis_3]
        es = [seq.e0, seq.e1, seq.e2, seq.e3]
        es_dbc = [seq.e0_dbc, seq.e1_dbc, seq.e2_dbc, seq.e3_dbc]
        e = es_dbc[k] if dirichlet else es[k]

        b = TestProjectionConvergence._project_tp(seq, k, f, dirichlet)
        u = seq.apply_inverse_mass_matrix(b, k, dirichlet=dirichlet)
        u_h = Pushforward(DiscreteFunction(u, bases[k], e), F, k)

        diff = jax.lax.map(
            lambda x: f(x) - u_h(x), seq.quad.x, batch_size=0)
        ref = jax.lax.map(f, seq.quad.x, batch_size=0)
        L2_d = jnp.einsum("ik,ik,i,i->",
                          diff, diff, seq.jacobian_j, seq.quad.w)
        L2_r = jnp.einsum("ik,ik,i,i->",
                          ref, ref, seq.jacobian_j, seq.quad.w)
        return float((L2_d / L2_r) ** 0.5)

    # -- convergence runner --------------------------------------------------
    @staticmethod
    def _convergence(seqs, k, f, dirichlet):
        errors = []
        for n in NS:
            err = TestProjectionConvergence._project_and_error(
                seqs[n], k, f, dirichlet)
            errors.append(err)
            print(f"  k={k} dbc={dirichlet} n={n}: error={err:.4e}")
        return errors

    @staticmethod
    def _assert_convergence(errs, label, threshold):
        assert errs[0] < threshold, (
            f"{label} error too large at n={NS[0]}: {errs[0]:.2e}")
        for i in range(1, len(errs)):
            assert errs[i] < errs[i-1], (
                f"{label} error did not decrease: {errs}")

    # -- k = 0 (DBC-compatible function, both spaces) -----------------------
    def test_proj_k0_dbc(self, seqs):
        errs = self._convergence(seqs, 0, self._scalar_dbc, dirichlet=True)
        self._assert_convergence(errs, "k=0 DBC", 0.2)

    def test_proj_k0_free(self, seqs):
        errs = self._convergence(seqs, 0, self._scalar_dbc, dirichlet=False)
        self._assert_convergence(errs, "k=0 free", 0.2)

    def test_proj_k0_free_nondbc_func(self, seqs):
        errs = self._convergence(seqs, 0, self._scalar_free, dirichlet=False)
        self._assert_convergence(errs, "k=0 free (non-DBC func)", 0.1)

    # -- k = 1 (DBC-compatible function, both spaces) -----------------------
    def test_proj_k1_dbc(self, seqs):
        errs = self._convergence(seqs, 1, self._vector_dbc, dirichlet=True)
        self._assert_convergence(errs, "k=1 DBC", 0.4)

    def test_proj_k1_free(self, seqs):
        errs = self._convergence(seqs, 1, self._vector_dbc, dirichlet=False)
        self._assert_convergence(errs, "k=1 free", 0.4)

    def test_proj_k1_free_nondbc_func(self, seqs):
        errs = self._convergence(seqs, 1, self._vector_free, dirichlet=False)
        self._assert_convergence(errs, "k=1 free (non-DBC func)", 0.5)

    # -- k = 2 (DBC-compatible function, both spaces) -----------------------
    def test_proj_k2_dbc(self, seqs):
        errs = self._convergence(seqs, 2, self._vector_dbc, dirichlet=True)
        self._assert_convergence(errs, "k=2 DBC", 0.4)

    def test_proj_k2_free(self, seqs):
        errs = self._convergence(seqs, 2, self._vector_dbc, dirichlet=False)
        self._assert_convergence(errs, "k=2 free", 0.4)

    def test_proj_k2_free_nondbc_func(self, seqs):
        errs = self._convergence(seqs, 2, self._vector_free, dirichlet=False)
        self._assert_convergence(errs, "k=2 free (non-DBC func)", 0.4)

    # -- k = 3 (DBC-compatible function, both spaces) -----------------------
    def test_proj_k3_dbc(self, seqs):
        errs = self._convergence(seqs, 3, self._scalar_dbc, dirichlet=True)
        self._assert_convergence(errs, "k=3 DBC", 0.4)

    def test_proj_k3_free(self, seqs):
        errs = self._convergence(seqs, 3, self._scalar_dbc, dirichlet=False)
        self._assert_convergence(errs, "k=3 free", 0.4)

    def test_proj_k3_free_nondbc_func(self, seqs):
        errs = self._convergence(seqs, 3, self._scalar_free, dirichlet=False)
        self._assert_convergence(errs, "k=3 free (non-DBC func)", 0.1)


class TestCrossProductCorrectness:
    """Check that e_y × e_z = e_x via cross-product projection.

    Projects constant physical vector fields (0,1,0) and (0,0,1), computes
    the cross-product projection for various (n,m,k) combinations, solves
    for the result DOFs, and pushes forward to physical space.  The metric
    terms cancel exactly in the pushforward, so the result should converge
    to (1,0,0) with increasing resolution.
    """

    @staticmethod
    def _e_x(x):
        return jnp.array([1.0, 0.0, 0.0])

    @staticmethod
    def _e_y(x):
        return jnp.array([0.0, 1.0, 0.0])

    @staticmethod
    def _e_z(x):
        return jnp.array([0.0, 0.0, 1.0])

    @staticmethod
    def _cross_error(seq, w_func, u_func, expected_func, n, m, k):
        """Project w (m-form) and u (k-form), compute cross product,
        solve, push forward as n-form, return relative L2 error vs expected."""
        proj_m = seq.p1 if m == 1 else seq.p2
        proj_k = seq.p1 if k == 1 else seq.p2
        w_dofs = seq.apply_inverse_mass_matrix(
            proj_m(w_func), m, dirichlet=False)
        u_dofs = seq.apply_inverse_mass_matrix(
            proj_k(u_func), k, dirichlet=False)

        rhs = seq.cross_product_projection(
            w_dofs, u_dofs, n, m, k,
            dirichlet_n=False, dirichlet_m=False, dirichlet_k=False)

        result_dofs = seq.apply_inverse_mass_matrix(rhs, n, dirichlet=False)

        bases = [seq.basis_0, seq.basis_1, seq.basis_2, seq.basis_3]
        es = [seq.e0, seq.e1, seq.e2, seq.e3]
        result_h = Pushforward(
            DiscreteFunction(result_dofs, bases[n], es[n]), seq.map, n)

        diff = jax.lax.map(
            lambda x: expected_func(x) - result_h(x),
            seq.quad.x, batch_size=50_000)
        ref = jax.lax.map(expected_func, seq.quad.x, batch_size=50_000)
        L2_d = jnp.einsum("ik,ik,i,i->",
                          diff, diff, seq.jacobian_j, seq.quad.w)
        L2_r = jnp.einsum("ik,ik,i,i->",
                          ref, ref, seq.jacobian_j, seq.quad.w)
        return float((L2_d / L2_r) ** 0.5)

    @staticmethod
    def _convergence(seqs, w_func, u_func, expected_func, n, m, k):
        errors = []
        for n_res in NS:
            err = TestCrossProductCorrectness._cross_error(
                seqs[n_res], w_func, u_func, expected_func, n, m, k)
            errors.append(err)
            print(f"  ({n},{m},{k}) n={n_res}: error={err:.4e}")
        return errors

    @staticmethod
    def _assert_convergence(errs, label, threshold):
        assert errs[0] < threshold, (
            f"{label} error too large at n={NS[0]}: {errs[0]:.2e}")
        for i in range(1, len(errs)):
            assert errs[i] < errs[i - 1], (
                f"{label} error did not decrease: {errs}")

    def test_cross_211(self, seqs):
        """(n=2,m=1,k=1): two 1-forms → 2-form, ∫ Λ2 G(w×u)/J dx."""
        errs = self._convergence(
            seqs, self._e_x, self._e_y, self._e_z, 2, 1, 1)
        self._assert_convergence(errs, "cross 211", 0.2)

    def test_cross_111(self, seqs):
        """(n=1,m=1,k=1): two 1-forms → 1-form, ∫ Λ1 (w×u) dx."""
        errs = self._convergence(
            seqs, self._e_x, self._e_y, self._e_z, 1, 1, 1)
        self._assert_convergence(errs, "cross 111", 0.2)

    def test_cross_222(self, seqs):
        """(n=2,m=2,k=2): two 2-forms → 2-form, ∫ Λ2 (w×u)/J dx."""
        errs = self._convergence(
            seqs, self._e_x, self._e_y, self._e_z, 2, 2, 2)
        self._assert_convergence(errs, "cross 222", 0.2)
