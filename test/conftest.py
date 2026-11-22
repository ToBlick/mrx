"""Shared fixtures and constructions of the test suite (docs/source/concepts/testing_strategy.md).

The fixture is the li383 equilibrium at spline degree 2 on the graded polar space (fewer theta cells near the axis),
in two representations of the domain with the same mesh, each built once per session:

- ``half``: the production sequence, one field period with stellarator symmetry, integrating half of it.
- ``period``: one field period without the reflection.

No test builds a sequence or preconditioners of the li383 domain of its own. A statement on another map installs
that map on the fixture with the fixture's preconditioners (:func:`map_installed`), and the fixture is put back by
reference afterwards. The suite's time goes into the solves and into compiling the fixture's programs, which are
cached on disk between runs (see below).

Shared constructions: the ideal flow of a velocity and the energy along it (:func:`ideal_flow`,
:func:`energy_derivatives`), a random divergence-free velocity (:func:`divergence_free`) and the minor radius of a
boundary (:func:`minor_radius`), the unit of every length bound.
"""
import contextlib
import os
import time

import jax
import numpy as np
import pytest

# Compiled programs are cached on disk, keyed by their HLO, so a stale entry
# costs only a recompile. MRX_XLA_CACHE names the directory. An empty value disables the cache.
# Unset, it is $MRX_DATA/cache/xla, or outputs/xla_cache in the repository without MRX_DATA.
_CACHE = os.environ.get("MRX_XLA_CACHE",
                        os.path.join(os.environ["MRX_DATA"], "cache", "xla") if os.environ.get("MRX_DATA")
                        else os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "xla_cache"))
if _CACHE:
    jax.config.update("jax_compilation_cache_dir", _CACHE)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.1)

#: The mesh per field period of the shared sequence and the spline degree: the smallest graded mesh that resolves
#: shearedE (see docs/source/concepts/testing_strategy.md).
NS, P = (12, 16, 12), 2
#: The surfaces of the fixture's wout.
WOUT_SURFACES = 51
#: The default bound of a length error, as a fraction of the minor radius of the device under test (Tobias
#: 2026-11-21). A test with a tighter measured bound states it.
LENGTH = 1e-2


@pytest.fixture(scope="session")
def eq():
    """The exact equilibrium shearedE (:class:`test.landreman.Sheared`)."""
    from test.landreman import Sheared
    return Sheared()


@pytest.fixture(scope="session")
def files(eq, tmp_path_factory):
    """``(wout, namelist)``: the paths of the exact equilibrium's wout and of its namelist (boundary, pressure and
    transform), written once per session."""
    d = tmp_path_factory.mktemp("sheared")
    return (eq.write_wout(str(d / "wout_shearedE.nc"), ns=WOUT_SURFACES),
            eq.write_namelist(str(d / "input.shearedE")))


@pytest.fixture(scope="session")
def seq(files):
    """The shared sequence: the wout of shearedE at :data:`NS`, degree :data:`P`, graded, one field period with
    stellarator symmetry, with its preconditioners and harmonic forms. The only sequence of the suite besides the
    tiny ones of the de Rham statements."""
    from mrx.geometry import build_sequence
    from mrx.nullspace import compute_nullspaces

    t0 = time.perf_counter()
    s = build_sequence(files[0], NS, P, symmetry="stellarator", graded=True)
    compute_nullspaces(s, verbose=False)
    print(f"\n  shearedE: {s.ns} p={P}, nfp={s.nfp}, built in {time.perf_counter() - t0:.0f} s", flush=True)
    return s


@pytest.fixture(scope="session")
def minor(eq):
    """The minor radius of shearedE (:func:`minor_radius` of its boundary)."""
    xm, xn, R, Z = eq.surfaces(np.array([1.0]))
    return minor_radius(np.stack([xm, xn, R[0], Z[0]], axis=1), 2)


@pytest.fixture(scope="session")
def b0(seq):
    """The exact field projected: the curl of its interpolated vector potential, a divergence-free 2-form built by
    :func:`mrx.relaxation.initial_conditions.potential_two_form` from the wout."""
    from mrx.relaxation.initial_conditions import potential_two_form

    B, _ = potential_two_form(seq)
    return B


def field_error(eq, seq, B):
    """The relative L2 error of the 2-form ``B`` against the exact field over the plasma, at the quadrature points."""
    from mrx.differential_forms import DiscreteFunction, Pushforward
    X = np.asarray(jax.vmap(seq.map)(seq.quad.x))
    w = np.asarray(seq.quad.w * seq.jacobian_j)
    Bh = np.asarray(jax.vmap(Pushforward(DiscreteFunction(B, seq.odd.basis_2, seq.odd.E(2)), seq.map, 2))(seq.quad.x))
    Be = eq.field(X)
    return float(np.sqrt(np.sum(w[:, None] * (Bh - Be) ** 2) / np.sum(w[:, None] * Be ** 2)))


def minor_radius(rows, nfp, n_v=16, n_u=256):
    """VMEC's minor radius ``sqrt(<A> / pi)`` of the boundary ``rows`` ``(m, n, R_mn, Z_mn)`` (``n`` times nfp,
    ``R = sum R_mn cos(m u - n v)``, ``Z = sum Z_mn sin(m u - n v)``), with ``<A>`` the mean area of its sections
    ``v = const`` over a field period. The unit of every length bound of the suite."""
    m, n, R_mn, Z_mn = (np.asarray(rows, dtype=float)[:, i] for i in range(4))
    u = 2.0 * np.pi * np.arange(n_u) / n_u
    area = []
    for v in 2.0 * np.pi / nfp * np.arange(n_v) / n_v:
        arg = m[None, :] * u[:, None] - n[None, :] * v
        R, dZ = np.cos(arg) @ R_mn, (m * np.cos(arg)) @ Z_mn
        area.append(abs(np.mean(R * dZ)) * 2.0 * np.pi)
    return float(np.sqrt(np.mean(area) / np.pi))


@contextlib.contextmanager
def map_installed(seq, Phi, harmonic=False):
    """Install the map ``Phi`` on the fixture ``seq`` with the fixture's preconditioners, which precondition the
    operators of a nearby map (test_poisson.py), and with the harmonic forms of ``Phi`` if ``harmonic``. On exit, also
    after a failure, the fixture's geometry and operators are put back by reference, so nothing is built either way."""
    from mrx.nullspace import compute_nullspaces

    saved = (seq.geometry, seq.operators, {s: (v.geometry, v.operators) for s, v in seq._parity_views.items()})
    try:
        seq.set_geometry_from_map(Phi, keep_preconditioners=True)
        if harmonic:
            compute_nullspaces(seq, verbose=False)
        yield seq
    finally:
        seq.geometry, seq.operators = saved[0], saved[1]
        for s, (geometry, operators) in saved[2].items():
            seq._parity_views[s].geometry, seq._parity_views[s].operators = geometry, operators


def divergence_free(seq, key):
    """A random divergence-free velocity, an even 2-form of unit norm."""
    even = seq.even
    w, _ = even.leray(jax.random.normal(key, (even.n(2),)), k=2)
    return w / even.l2_norm(w, 2)


def ideal_flow(seq, u):
    """``X -> A_u X = G_1 M_1^{-1} load(u x Pi_1 X)`` for odd 2-forms ``X``, the increment of the ideal step of the
    velocity ``u``, with ``Pi_1 X`` the L2 projection onto the free 1-forms formed here from the mass matrix alone."""
    odd, even = seq.odd, seq.even
    free = odd.free
    u_q = even.evaluate_at_quadrature(u, 2)

    def A(X):
        h = free.M[1].solve(free.vector_load_values(odd.evaluate_at_quadrature(X, 2), 2, 1))
        return odd.G[1] @ odd.M[1].solve(odd.cross_product_load_values(u_q, free.evaluate_at_quadrature(h, 1), 1, 2, 1))
    return A


def energy_derivatives(seq, B, u):
    """``(E'(0), E''(0))`` of the energy ``||B(t)||^2_M / 2`` along the ideal flow ``B' = A_u B`` of ``u``:
    ``(B, A_u B)_M`` and ``||A_u B||^2_M + (B, A_u^2 B)_M``."""
    odd = seq.odd
    A = ideal_flow(seq, u)
    AB = A(B)
    return float(B @ (odd.M[2] @ AB)), float(odd.l2_norm_sq(AB, 2) + B @ (odd.M[2] @ A(AB)))
