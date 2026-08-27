"""The weak pressure (``mrx.relaxation.weak_pressure``) on ``tiny_seq`` (ci tier).

1. ``test_dirichlet_leray_recovers_gradient`` -- the k=1 Leray projection
   with the Dirichlet scalar space returns ``p_w = q`` for ``v = grad q``,
   ``q`` in the Dirichlet 0-form space, to solver tolerance, and the
   remainder vanishes.
2. ``test_pressure_diagnostics_analytic`` -- for ``q = 1 - r^2`` and the
   vacuum toroidal field ``B = e_phi / R`` on the spline-interpolated torus,
   ``beta_vol`` and ``beta_axis`` match the closed-form integrals, the
   wall-normal gradient equals the largest gradient (it is largest at the
   wall), and the two strong/weak comparisons are small for a strong
   pressure built from the same ``q``.
"""

import jax.numpy as jnp
import numpy as np

import mrx
from mrx.relaxation import pressure_diagnostics
from test.conftest import TORUS_EPSILON, TORUS_R0


def test_dirichlet_leray_recovers_gradient(tiny_seq):
    seq = tiny_seq
    q = jnp.asarray(np.random.default_rng(11).standard_normal(seq.n(0, True)), dtype=mrx.DTYPE)
    q = q / seq.l2_norm(q, 0)
    # The exact strong gradient into the natural 1-form space: v is a
    # gradient to round-off, so the decomposition must return it whole.
    v = seq.apply_incidence_matrix(q, 0, dirichlet_in=True, dirichlet_out=False)
    F_w, p_w = seq.apply_leray_projection(v, k=1, dirichlet_p=True)
    err = float(seq.l2_norm(p_w - q, 0) / seq.l2_norm(q, 0))
    rem = float(seq.l2_norm(F_w, 1, dirichlet=False) / seq.l2_norm(v, 1, dirichlet=False))
    print(f"\n  Dirichlet k=1 Leray: ||p_w - q||/||q|| {err:.2e}, ||F_w||/||v|| {rem:.2e}  "
          f"(tol {seq.tol:.1e})")
    # The k=0 solve stops at seq.tol on its residual; the error in p_w is
    # that times the condition number of the preconditioned Laplacian, and
    # F_w = v - M_1^-1 D_0 p_w adds one mass solve's residual.
    assert err <= 100 * seq.tol
    assert rem <= 100 * seq.tol
    assert int(p_w.shape[0]) == int(seq.n(0, True))


def test_pressure_diagnostics_analytic(tiny_seq):
    """Closed forms on the torus ``R = R0 + eps r cos 2 pi theta``,
    ``dV = 4 pi^2 eps^2 r R dr dtheta dzeta``:

        int (1 - r^2) dV = pi^2 eps^2 R0
        E = int |e_phi / R|^2 / 2 dV = 2 pi^2 (R0 - sqrt(R0^2 - eps^2))
        beta_axis = (1 - 0) / (1 / R0^2 / 2) = 2 R0^2   (read at r = x_r[0], O(r^2) off)

    The map is the spline interpolant of the torus and ``B`` its L2
    projection at (4, 6, 4) p=2, so the closed forms hold to the
    approximation error of that resolution, not to solver tolerance.
    """
    seq = tiny_seq
    eps, R0 = TORUS_EPSILON, TORUS_R0
    two_pi = 2.0 * jnp.pi

    q = seq.interpolate(lambda x: jnp.ones(1) * (1.0 - x[0] ** 2), 0, dirichlet=True)
    v = seq.apply_incidence_matrix(q, 0, dirichlet_in=True, dirichlet_out=False)
    F_w, p_w = seq.apply_leray_projection(v, k=1, dirichlet_p=True)

    def B_phys(x):
        r, th, ze = x
        R = R0 + eps * r * jnp.cos(two_pi * th)
        return jnp.array([-jnp.sin(two_pi * ze), -jnp.cos(two_pi * ze), 0.0]) / R
    B = seq.apply_inverse_mass_matrix(seq.load(B_phys, 2, dirichlet=True), 2, dirichlet=True)
    # The strong pressure of the comparison: the k=2 Leray multiplier of the
    # Dirichlet 2-form projection of grad q, i.e. the 3-form whose weak
    # gradient is that projection -- the same function as q up to the
    # discretisation, so both comparisons read the discretisation only.
    gq2 = seq.apply_inverse_mass_matrix(
        seq.apply_projection_matrix(v, 1, 2, dirichlet_in=False, dirichlet_out=True), 2)
    _, p = seq.apply_leray_projection(gq2, k=2)

    d = {k: float(val) for k, val in pressure_diagnostics(B, p, p_w, F_w, v, seq).items()}
    int_q = jnp.pi ** 2 * eps ** 2 * R0
    energy = 2.0 * jnp.pi ** 2 * (R0 - jnp.sqrt(R0 ** 2 - eps ** 2))
    beta_vol_exact = float(int_q / energy)
    beta_axis_exact = 2.0 * R0 ** 2
    print(f"\n  beta_vol {d['beta_vol']:.5f} (exact {beta_vol_exact:.5f})  "
          f"beta_axis {d['beta_axis']:.5f} (exact {beta_axis_exact:.5f})  "
          f"dpdn_wall {d['dpdn_wall']:.4f}  JxBn_wall {d['JxBn_wall']:.4f}  "
          f"gradp_cmp {d['gradp_cmp']:.3e}  p_cmp {d['p_cmp']:.3e}  weak_resid {d['weak_resid']:.2e}")
    assert abs(d["beta_vol"] / beta_vol_exact - 1.0) < 1e-1
    assert abs(d["beta_axis"] / beta_axis_exact - 1.0) < 1e-1
    # |grad q| = 2 r / eps peaks at the wall, where it is purely normal; the
    # interior maximum sits at the outermost Gauss layer, r < 1.
    assert 1.0 <= d["dpdn_wall"] < 1.1
    assert abs(d["JxBn_wall"] - d["dpdn_wall"]) < 1e-2
    assert d["weak_resid"] <= 100 * seq.tol
    assert d["gradp_cmp"] < 1e-1
    # As FUNCTIONS the two differ more: the 3-form's value is p_ref / J, and
    # its degree-(p-1) D-spline in r does not vanish at the axis exactly, so
    # p / J carries a 1/r layer there. Measured 0.24 at (4,6,4) p=2 against
    # 2.8e-10 for the projected gradients above.
    assert d["p_cmp"] < 0.5
