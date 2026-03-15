# test_sampled_projection.py
"""
Test projection of a field given as sampled data on a regular grid,
using :func:`mrx.io.project_sampled_field`.
"""

import jax
import jax.numpy as jnp
import pytest

from mrx.derham_sequence import DeRhamSequence
from mrx.differential_forms import DiscreteFunction, Pushforward
from mrx.io import project_sampled_field
from mrx.mappings import toroid_map

jax.config.update("jax_enable_x64", True)


# --- analytic vector field (v = 0 on boundary) ---
def v_field(x):
    r, theta, z = x
    pi = jnp.pi
    return jnp.array([
        jnp.cos(2 * pi * z),
        jnp.sin(2 * pi * theta),
        1,
    ]) * r * (1 - r)


# --- shared helpers ---
_MAP = toroid_map(epsilon=1 / 3)
_P = 3


def _make_seq(n):
    s = DeRhamSequence(
        (n, n, n), (_P, _P, _P), 2 * _P,
        ("clamped", "periodic", "periodic"),
        _MAP, polar=True,
    )
    s.evaluate_1d()
    s.assemble_mass_matrix(2)
    return s


# --- fixtures ---
@pytest.fixture(scope="module")
def coarse_seq():
    return _make_seq(6)


@pytest.fixture(scope="module")
def fine_seq():
    return _make_seq(7)


@pytest.fixture(scope="module")
def sampled_data():
    """Evaluate the vector field on a 50³ regular grid over [0,1]³."""
    ng = 50
    _r = jnp.linspace(0, 1, ng)
    _t = jnp.linspace(0, 1, ng)
    _z = jnp.linspace(0, 1, ng)
    pts = jnp.stack(
        jnp.meshgrid(_r, _t, _z, indexing="ij"), axis=-1
    ).reshape(-1, 3)
    v_vals = jax.vmap(v_field)(pts)
    return _r, _t, _z, v_vals


class TestSampledProjection:
    """Project a vector field from sampled data and compare to direct projection."""

    @staticmethod
    def _project_from_samples(seq, _r, _t, _z, v_vals, dirichlet=True):
        """Project a 2-form from sampled data via project_sampled_field."""
        return project_sampled_field(
            axes=(_r, _t, _z), values=v_vals,
            seq=seq, k=2, dirichlet=dirichlet,
        )

    @staticmethod
    def _project_direct(seq, dirichlet=True):
        """Project the same vector field directly using the Projector (ground truth)."""
        proj = seq.p2_dbc if dirichlet else seq.p2
        rhs = proj(v_field)
        return seq.apply_inverse_mass_matrix(rhs, k=2, dirichlet=dirichlet)

    @staticmethod
    def _l2_error(seq, dof, f, dirichlet=True):
        """Relative L2 error of a 2-form DOF vector vs analytic function f."""
        e2 = seq.e2_dbc if dirichlet else seq.e2
        u_h = Pushforward(DiscreteFunction(dof, seq.basis_2, e2), seq.map, 2)

        diff = jax.lax.map(
            lambda x: f(x) - u_h(x), seq.quad.x, batch_size=0)
        ref = jax.lax.map(f, seq.quad.x, batch_size=0)
        L2_d = jnp.einsum("ik,ik,i,i->", diff, diff,
                          seq.jacobian_j, seq.quad.w)
        L2_r = jnp.einsum("ik,ik,i,i->", ref, ref, seq.jacobian_j, seq.quad.w)
        return float((L2_d / L2_r) ** 0.5)

    def test_sampled_projection_dbc(self, coarse_seq, sampled_data):
        """Sampled-data 2-form projection (DBC) recovers the field within 5%."""
        _r, _t, _z, v_vals = sampled_data
        v_dof = self._project_from_samples(
            coarse_seq, _r, _t, _z, v_vals, dirichlet=True)
        err = self._l2_error(coarse_seq, v_dof, v_field, dirichlet=True)
        print(f"Sampled projection (DBC): relative L2 error = {err:.4e}")
        assert err < 0.05, f"Sampled DBC projection error too large: {err:.2e}"

    def test_sampled_projection_free(self, coarse_seq, sampled_data):
        """Sampled-data 2-form projection (free BC) recovers the field within 5%."""
        _r, _t, _z, v_vals = sampled_data
        v_dof = self._project_from_samples(
            coarse_seq, _r, _t, _z, v_vals, dirichlet=False)
        err = self._l2_error(coarse_seq, v_dof, v_field, dirichlet=False)
        print(f"Sampled projection (free): relative L2 error = {err:.4e}")
        assert err < 0.05, f"Sampled free projection error too large: {err:.2e}"

    def test_convergence_dbc(self, coarse_seq, fine_seq, sampled_data):
        """Error decreases when refining the FEM mesh (DBC)."""
        _r, _t, _z, v_vals = sampled_data

        v_dof_c = self._project_from_samples(
            coarse_seq, _r, _t, _z, v_vals, dirichlet=True)
        err_coarse = self._l2_error(
            coarse_seq, v_dof_c, v_field, dirichlet=True)

        v_dof_f = self._project_from_samples(
            fine_seq, _r, _t, _z, v_vals, dirichlet=True)
        err_fine = self._l2_error(fine_seq, v_dof_f, v_field, dirichlet=True)

        print(
            f"Convergence (DBC): n=6 err={err_coarse:.4e}, n=7 err={err_fine:.4e}")
        assert err_fine < err_coarse, (
            f"Error did not decrease with refinement: {err_fine:.2e} >= {err_coarse:.2e}"
        )

    def test_sampled_vs_direct_projection(self, coarse_seq, sampled_data):
        """Sampled-data projection DOFs are close to direct-projection DOFs."""
        _r, _t, _z, v_vals = sampled_data
        v_dof_sampled = self._project_from_samples(
            coarse_seq, _r, _t, _z, v_vals, dirichlet=True)
        v_dof_direct = self._project_direct(coarse_seq, dirichlet=True)

        # Compare via the mass-matrix norm: ||u - v||_M / ||v||_M
        diff = v_dof_sampled - v_dof_direct
        norm_diff = (diff @ coarse_seq.apply_mass_matrix(diff,
                     2, dirichlet=True)) ** 0.5
        norm_ref = (
            v_dof_direct @ coarse_seq.apply_mass_matrix(
                v_dof_direct, 2, dirichlet=True)) ** 0.5
        rel = float(norm_diff / norm_ref)
        print(f"Sampled vs direct DOF diff (mass norm): {rel:.4e}")
        assert rel < 0.05, f"Sampled-vs-direct difference too large: {rel:.2e}"
