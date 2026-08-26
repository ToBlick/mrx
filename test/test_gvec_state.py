"""``mrx.gvec_state``: the radial basis, and (``needs_data``) the evaluation
of a GVEC state against the pyGVEC export of the same equilibrium --
``MRX_GVEC_STATE`` and ``MRX_W7X_FILE`` name the two files."""
import os

import numpy as np
import pytest

from mrx.gvec_state import StateField, evaluate, profile_spline, radial_design, read_state


def test_radial_basis_is_a_clamped_partition_of_unity():
    sp = np.linspace(0.0, 1.0, 11)
    s = np.linspace(0.0, 1.0, 97)
    D = radial_design(sp, 5, s)
    assert D.shape == (97, len(sp) - 1 + 5)
    assert np.allclose(D.sum(1), 1.0)
    assert D[0, 0] == 1.0 and D[-1, -1] == 1.0            # clamped ends


def test_state_field_matches_the_numpy_series():
    """The JAX evaluator and the tensor-grid series agree on a made-up state."""
    import jax
    import jax.numpy as jnp
    rng = np.random.default_rng(3)
    sp = np.linspace(0.0, 1.0, 7)
    m = np.array([0, 1, 2, 3, 1, 2])
    n = np.array([0, 0, 5, -5, 10, -10])
    block = dict(m=m, n=n, coef=rng.normal(size=(6, len(sp) - 1 + 3)), sin_cos=2, deg=3)
    field = StateField(block, sp, 5)
    s_, th, ze = np.array([0.15, 0.6, 0.97]), np.array([0.1, 0.45]), np.array([0.0, 0.8])
    grid = evaluate(block, sp, s_, 2 * np.pi * th, 2 * np.pi * ze / 5)
    pts = jnp.array([[a, b, c] for a in s_ for b in th for c in ze])
    mine = np.asarray(jax.vmap(field)(pts)).reshape(grid.shape)
    assert np.abs(mine - grid).max() < 1e-12
    block["sin_cos"] = 1
    grid = evaluate(block, sp, s_, 2 * np.pi * th, 2 * np.pi * ze / 5)
    mine = np.asarray(jax.vmap(StateField(block, sp, 5))(pts)).reshape(grid.shape)
    assert np.abs(mine - grid).max() < 1e-12


@pytest.mark.needs_data
def test_state_reproduces_the_export():
    state, export = os.environ.get("MRX_GVEC_STATE"), os.environ.get("MRX_W7X_FILE")
    if not (state and export and os.path.isfile(state) and os.path.isfile(export)):
        pytest.skip("set MRX_GVEC_STATE and MRX_W7X_FILE to the state and its export")
    import h5py
    st = read_state(state)
    with h5py.File(export, "r") as f:
        n = tuple(int(f.attrs[k]) for k in ("n_rho", "n_theta", "n_zeta"))
        assert int(f.attrs["nfp"]) == st["nfp"]
        ep = np.asarray(f["eval_points"])
        ref = {k: np.asarray(f[k]).reshape(n) for k in
               ("R", "Z", "pressure", "clebsch/dPhi_dr", "clebsch/dchi_dr", "clebsch/LA")}
    rho, th, ze = (np.unique(ep[:, i]) for i in range(3))
    theta, zeta = 2 * np.pi * th, 2 * np.pi * ze / st["nfp"]
    for blk, key in (("X1", "R"), ("X2", "Z"), ("LA", "clebsch/LA")):
        assert np.abs(evaluate(st[blk], st["sp"], rho, theta, zeta) - ref[key]).max() < 1e-10
    dphi = profile_spline(st, "phi").derivative()(rho)
    assert np.abs(dphi - ref["clebsch/dPhi_dr"][:, 0, 0]).max() < 1e-10
    dchi = profile_spline(st, "iota")(rho) * dphi
    scale = np.abs(ref["clebsch/dchi_dr"]).max()
    assert np.abs(dchi - ref["clebsch/dchi_dr"][:, 0, 0]).max() < 1e-4 * scale
    p = profile_spline(st, "pressure")(rho)
    assert np.abs(p - ref["pressure"][:, 0, 0]).max() < 1e-4 * ref["pressure"].max()
    # the closed-form lambda the initial condition uses, at the export's points
    import jax
    import jax.numpy as jnp
    from mrx.gvec import load_clebsch
    cb = load_clebsch(state)
    pts = jnp.array([[rho[i], th[j], ze[k]] for i in (1, 20, 49) for j in (0, 7) for k in (0, 31)])
    mine = np.asarray(jax.vmap(cb["lam_h"])(pts))
    want = np.array([ref["clebsch/LA"][i, j, k] for i in (1, 20, 49) for j in (0, 7) for k in (0, 31)])
    assert np.abs(mine - want).max() < 1e-10
