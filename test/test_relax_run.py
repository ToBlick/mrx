"""One run of ``scripts/relax.py`` with every control, on a mesh MRX builds itself and on the wout's mesh.

A VMEC input namelist holds the li383 wout's boundary, its toroidal flux, its pressure and the net toroidal current
inside every surface of the wout's field (``NCURR = 1``). Its map, the boundary filled about the guessed axis, meets
the boundary with a positive Jacobian, and its start, the vacuum field, has the namelist's flux, no divergence and no
current. Run A is the user's run from that namelist alone: the vacuum start on the mesh moved onto the vacuum's axis,
the pole kept on the magnetic axis and the pressure and the current as targets (``--mesh.align --pressure.target
--current.target``). Run B has the same targets on the wout's mesh from the wout's field, without motion. Both take
positive steps and lower ``W`` at every step, both end at the targets, A's moves keep ``div B`` and the toroidal flux
and put the magnetic axis on the pole, and the two runs end at the same field at common physical points and with the same transform on their own
labels: the controls are functions of the label alone, so the end state does not depend on the mesh or on the start.

``scripts/relax.py`` is loaded from its path. Its sequences are those of the fixtures (``Geometry.build`` returns
them): run B's the session's ``seq``, run A's the fill's, which its moves change.
"""
import importlib.util
import json
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mrx.differential_forms import DiscreteFunction, Pushforward
from mrx.flux_label import ANISOTROPY
from mrx.geometry import build_sequence
from mrx.nullspace import compute_nullspaces
from mrx.precision import DTYPE, eps
from mrx.relaxation.loop import read_field
from mrx.relaxation.shells import label, logical_points
from mrx.relaxation.twist import enclosed_current, iota_fit
from test.conftest import NS, P
from test.test_fill import _write_namelist
from test.test_twist import CELLS

_spec = importlib.util.spec_from_file_location(
    "relax_script", pathlib.Path(__file__).resolve().parents[1] / "scripts" / "relax.py")
relax_script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(relax_script)

#: The Newton steps of both runs, in chunks with the pole moved (A) and the label, the shells and the slips rebuilt
#: before each, and A's moves, one before each chunk.
STEPS, CHUNK = 20, 10
MOVES = STEPS // CHUNK
#: The distance of the magnetic axis from the pole in first radial cells on the start's final map.
AXIS_START = 1e-2
#: The interior of the labels where the transforms are compared.
S_TEST = np.linspace(0.05, 0.95, 19)
#: The pressure's and the current's error at the end of each run: 3.3e-3 and 3.8e-3 for A, 3.8e-3 and 6.8e-4 for B
#: measured (2026-11-19).
PRESSURE_ERROR, CURRENT_ERROR = 1e-2, 1e-2
#: The two end fields at common physical points, and the two transforms on their labels: 6.5e-3 and 9.2e-3 measured
#: (2026-11-19).
B_DIFF, IOTA_DIFF = 2e-2, 3e-2


def _profiles(seq, b0):
    """The namelist lines of the fixture wout's pressure and, as ``NCURR = 1``, of the current of its field inside the
    level sets of the field's label."""
    _, T, _ = label(seq, b0, ANISOTROPY)
    s_q, _, _ = label(seq, b0, ANISOTROPY, T)
    edges = np.linspace(0.0, 1.0, 41)
    I_z = np.concatenate([[0.0], enclosed_current(seq, b0, np.asarray(s_q), edges[1:])])
    flux = float(jnp.sum(seq.quad.w * seq.odd.evaluate_at_quadrature(b0, 2)[:, 2]))
    # the namelist's current is along phi: along the field with the sign of PHIEDGE, along +zeta with that of the flux
    I_phi = I_z * np.sign(flux) * np.sign(float(seq.equilibrium["profiles"]["phi"](1.0)))
    p_file = seq.equilibrium["profiles"]["pressure"]                    # Pa, a function of r = sqrt(s)
    s = np.linspace(0.0, 1.0, 101)
    row = lambda x: " ".join(f"{v:.17g}" for v in x)  # noqa: E731
    return ["NCURR = 1", "PCURR_TYPE = 'cubic_spline_i'", f"CURTOR = {I_phi[-1] / (4e-7 * np.pi):.17g}",
            f"AC_AUX_S = {row(edges)}", f"AC_AUX_F = {row(I_phi / I_phi[-1])}", "PMASS_TYPE = 'cubic_spline'",
            f"AM_AUX_S = {row(s)}", f"AM_AUX_F = {row(p_file(np.sqrt(s)))}"]


def _run(seq, geometry, namelist, out, extra=()):
    """``(relax.json, B)``: the record and the end field of ``relax.py`` on ``geometry`` with the namelist's
    pressure and current as targets, on the sequence ``seq``."""
    import tyro
    from mrx.relaxation.config import Geometry, RelaxConfig
    args = ["--geometry.path", str(geometry), "--geometry.resolution", *map(str, NS), "--geometry.graded",
            "--geometry.precision", np.dtype(DTYPE).name, "--budget.steps", str(STEPS), "--budget.chunk", str(CHUNK),
            "--pressure.target", str(namelist), "--pressure.cells", str(CELLS), "--current.target", str(namelist),
            "--output.out", str(out), *extra]
    cfg = tyro.cli(RelaxConfig, args=args)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(Geometry, "build", lambda self: seq)
        relax_script.main(cfg)
    rec = json.loads((out / "relax.json").read_text())
    B, step = read_field(str(out / "checkpoints" / f"state_{STEPS:06d}.h5"), seq)
    assert step == STEPS
    return rec, B


@pytest.fixture(scope="module")
def filled(seq, b0, tmp_path_factory):
    """``(seq, namelist, start)``: the sequence of the namelist (the wout's boundary filled about the guessed axis,
    with the wout's pressure and current) with its harmonic forms, the namelist's path, and the checks of the map and
    of the vacuum start, made before run A moves the map. The build costs 118 s in float64 and 165 s in float32 on
    four CPU cores."""
    from mrx.relaxation.initial_conditions import initial_field
    from mrx.relaxation.loop import radial_cell_sq
    from mrx.relaxation.physics import current

    d = tmp_path_factory.mktemp("relax_run")
    path, boundary, nfp, phiedge = _write_namelist(d, _profiles(seq, b0), "input.li383_run")
    sq = build_sequence(path, NS, P, symmetry="stellarator", graded=True)
    compute_nullspaces(sq, verbose=False)
    # the map at r = 1 against the boundary
    t, z = np.meshgrid(np.arange(64) / 64, np.arange(12) / 12, indexing="ij")
    x = jnp.asarray(np.stack([np.ones(t.size), t.ravel(), z.ravel()], axis=1))
    # a point of the wall keeps its theta and slides along the boundary, so it lies on the boundary at its own phi
    X = np.asarray(jax.vmap(sq.map)(x))
    RZ = np.stack([np.hypot(X[:, 0], X[:, 1]), X[:, 2]], axis=1)
    u = 2.0 * np.pi * t.ravel() * (-1.0 if sq.equilibrium["theta_reversed"] else 1.0)
    v = np.arctan2(X[:, 1], X[:, 0])
    arg = boundary[:, 0][None, :] * u[:, None] - boundary[:, 1][None, :] * v[:, None]
    R_b, Z_b = np.cos(arg) @ boundary[:, 2], np.sin(arg) @ boundary[:, 3]
    # the vacuum start: its flux through every quadrature plane, its divergence and its weak current
    B, info = initial_field(sq)
    q = sq.quad
    planes = jnp.einsum("i,j,ijk->k", q.w_x, q.w_y, sq.odd.evaluate_at_quadrature(B, 2)[:, 2].reshape(q.shape))
    start = dict(kind=info["kind"], det_min=float(jnp.min(sq.jacobian_j)),
                 boundary_miss=float(np.max(np.hypot(RZ[:, 0] - R_b, RZ[:, 1] - Z_b))),
                 flux=float(jnp.max(jnp.abs(planes - phiedge))) / abs(phiedge),
                 div=info["div"] / float(sq.odd.l2_norm(B, 2)),
                 current=float(jnp.sqrt(radial_cell_sq(sq)) * sq.odd.l2_norm(current(B, sq), 1)
                               / sq.odd.l2_norm(B, 2)))
    return sq, path, start


def test_filled_map_meets_the_boundary_with_positive_jacobian(filled):
    """The spline map of the fill has a positive Jacobian at every quadrature point (smallest det DPhi 1.6e-2), and at
    r = 1 it lies on the namelist's boundary up to the projection onto the (16, 32, 12) degree-2 splines: 6.5e-4 m at
    most on li383 in both precisions (bound 1e-3 m, minor radius about 0.3 m)."""
    start = filled[2]
    print(f"\n  min det DPhi {start['det_min']:.3e}, max distance from the boundary at r = 1 {start['boundary_miss']:.2e} m")
    assert start["det_min"] > 0.0
    assert start["boundary_miss"] < 1e-3


def test_vacuum_start_has_the_flux_and_no_divergence_or_current(filled):
    """The start field of the namelist is the harmonic 2-form: its flux through every quadrature plane
    ``zeta = const`` is PHIEDGE to round-off (the planes' fluxes agree because div B = 0 and B . n = 0), its divergence
    is round-off, and its weak current ``M_1^-1 D_1^T M_2 B`` is at the solver tolerance or the round-off of the
    precision. Measured: the planes' fluxes 1.9 eps off PHIEDGE in both precisions (bound 10 eps), ``||div B|| / ||B||``
    11.5 eps in float64 and 1.1 eps in float32 (bound 100 eps), ``h_r ||J|| / ||B||`` 6.6e-10 in float64 (bound the
    solve tolerance 1e-8) and 5.4e-7 = 4.5 eps in float32 (bound 100 eps)."""
    sq, _, start = filled
    print(f"\n  flux per plane {start['flux']:.2e} off PHIEDGE ({start['flux'] / eps():.1f} eps), div {start['div']:.2e} "
          f"({start['div'] / eps():.1f} eps), h_r |J| / |B| {start['current']:.2e} (solve tol {sq.tol:.0e})")
    assert start["kind"] == "indata"
    assert start["flux"] < eps(10) and start["div"] < eps(100) and start["current"] < max(sq.tol, eps(100))


@pytest.fixture(scope="module")
def runs(filled, seq):
    """``{"A": (seq, rec, B), "B": (seq, rec, B)}``: run A from the namelist alone with the mesh motion, on the fill's
    sequence, which its moves change, and run B on the wout with the same targets. 773 s on four CPU cores in
    float64, 310 s of steps per run (2026-11-19). Skipped in float32, where a Newton step costs three times as much
    on the CPU."""
    if DTYPE != np.float64:
        pytest.skip("a float32 Newton step costs three times a float64 one on the CPU, the float64 run covers the claim")
    seq_a, namelist, _ = filled
    d = pathlib.Path(namelist).parent
    out = {"A": (seq_a, *_run(seq_a, namelist, namelist, d / "A",
                                 ("--mesh.align",))),
           "B": (seq, *_run(seq, seq.equilibrium["path"], namelist, d / "B"))}
    for k, (_, rec, _) in out.items():
        sm = rec["summary"]
        print(f"\n  run {k}: residual {sm['resid_final']:.3e}, pressure error {sm['pressure_error']:.3e}, current "
              f"error {sm['current_error']:.3e}, E {sm['E']:.6e}, steps' wall {sm['wall']:.0f} s")
    return out


def test_both_runs_descend(runs):
    """Every Newton step of both runs moves along its descent direction (``dt* > 0``) and lowers ``W``, the energy
    less the targets' work, up to the round-off of the energy."""
    for k, (_, rec, _) in runs.items():
        dW, dt = np.asarray(rec["trace"]["dW"]), np.asarray(rec["trace"]["dt_star"])
        E0 = rec["summary"]["E0"]
        print(f"\n  run {k}: min dt* {dt.min():.2e}, max dW / E0 {dW.max() / E0:+.1e}")
        assert dt.min() > 0.0
        assert dW.max() < eps() * E0, f"run {k}: W not monotone: {dW}"


def test_both_runs_meet_the_targets(runs):
    """Both runs end at the namelist's pressure and current, A from the vacuum, B from the wout's field. The errors
    are finite before every chunk."""
    for k, (_, rec, _) in runs.items():
        sm = rec["summary"]
        assert all(np.isfinite(r["error"]) for r in rec["pressure"]["rebuilds"] + rec["current"]["rebuilds"]), k
        assert sm["pressure_error"] < PRESSURE_ERROR, k
        assert sm["current_error"] < CURRENT_ERROR, k


def test_the_pole_follows_the_axis(runs):
    """Run A starts on the mesh moved onto the vacuum's axis, which ends within :data:`AXIS_START` first radial cells of
    the pole, so the move before the first chunk is that small too, and moves its pole onto the field's axis before
    each chunk without a fold."""
    mesh = runs["A"][1]["mesh"]
    print("\n  start: axis " + " -> ".join(f"{r['axis_cells']:.2e}" for r in mesh["start"]["rounds"])
          + f" first cells, {mesh['start']['axis_cells']:.2e} on the final map")
    assert mesh["start"]["axis_cells"] < AXIS_START
    for m in mesh["moves"]:
        print(f"  step {m['step']}: axis {m['axis_cells']:.2e} cells off the pole, move x{m['lam']:g}, residual "
              f"{m['resid_before']:.2e} -> {m['resid_after']:.2e}")
        assert m["lam"] == 1.0
    assert mesh["moves"][0]["axis_cells"] < AXIS_START
    assert len(mesh["moves"]) == MOVES


def test_the_two_runs_end_at_the_same_state(runs):
    """The end fields of A and B at the physical points of a logical grid of B's mesh agree to B_DIFF, and the
    transforms on their own labels on ``s`` in 0.05 .. 0.95 to IOTA_DIFF."""
    (seq_a, _, B_a), (seq_b, _, B_b) = runs["A"], runs["B"]
    r, t, z = np.meshgrid(np.linspace(0.05, 0.95, 10), np.arange(16) / 16, np.arange(8) / 8, indexing="ij")
    x_b = jnp.asarray(np.stack([r.ravel(), t.ravel(), z.ravel()], axis=1))
    x_a, miss = logical_points(seq_a, jax.vmap(seq_b.map)(x_b))

    def physical(sq, B, x):
        return np.asarray(jax.vmap(Pushforward(DiscreteFunction(B, sq.odd.basis_2, sq.odd.E(2)), sq.map, 2))(x))
    F_a, F_b = physical(seq_a, B_a, x_a), physical(seq_b, B_b, x_b)
    b_diff = float(np.sqrt(np.sum((F_a - F_b) ** 2) / np.sum(F_b ** 2)))

    def iota(sq, B):
        _, T, _ = label(sq, B, ANISOTROPY)
        s_q, T, _ = label(sq, B, ANISOTROPY, T)
        return np.abs(iota_fit(sq, B, T, np.asarray(s_q), 6)(S_TEST))
    i_a, i_b = iota(seq_a, B_a), iota(seq_b, B_b)
    i_diff = float(np.abs(i_a - i_b).max())
    print(f"\n  B at common points {b_diff:.2e} (location miss {float(jnp.max(miss)):.1e} m), iota {i_diff:.2e} "
          f"(B: {i_b.min():.4f} .. {i_b.max():.4f})")
    assert b_diff < B_DIFF
    assert i_diff < IOTA_DIFF
