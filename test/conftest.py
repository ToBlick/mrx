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

#: The wout geometry, tracked in the repository.
GEOMETRY = "data/wout_li383_low_res_reference.nc"
#: Resolution (r, theta, zeta) per field period and the spline degree.
NS, P = (16, 32, 12), 2
#: The two representations of the li383 domain, by their symmetry.
DOMAINS = {"half": "stellarator", "period": "field-period"}
#: The default bound of a length error, as a fraction of the minor radius of the device under test (Tobias
#: 2026-11-21). A test with a tighter measured bound states it.
LENGTH = 1e-2


def _build(name):
    from mrx.geometry import build_sequence
    from mrx.nullspace import compute_nullspaces

    t0 = time.perf_counter()
    s = build_sequence(GEOMETRY, NS, P, symmetry=DOMAINS[name], graded=True)
    compute_nullspaces(s, verbose=False)
    print(f"\n  li383 {name}: {s.ns} p={P}, nfp={s.nfp}, built in {time.perf_counter() - t0:.0f} s", flush=True)
    return s


_SEQUENCES = {}


@pytest.fixture(scope="session")
def domains():
    """A function ``name -> sequence`` for the names of :data:`DOMAINS`, each built once per session with its
    preconditioners and harmonic forms."""
    def get(name):
        if name not in _SEQUENCES:
            _SEQUENCES[name] = _build(name)
        return _SEQUENCES[name]
    return get


@pytest.fixture(scope="session")
def seq(domains):
    """The production sequence, ``half``."""
    return domains("half")


@pytest.fixture(scope="session")
def minor():
    """The minor radius of the fixture's wout boundary (:func:`minor_radius`), 0.326 m."""
    from scipy.io import netcdf_file

    from mrx.paths import data_file
    with netcdf_file(data_file(GEOMETRY), "r", mmap=False) as f:
        v = f.variables
        rows = np.stack([v["xm"].data, v["xn"].data, v["rmnc"].data[-1], v["zmns"].data[-1]], axis=1)
        return minor_radius(rows, int(v["nfp"].data))


@pytest.fixture(scope="session")
def b0(seq):
    """The equilibrium's own field ``B = dA'``, a divergence-free 2-form built by
    :func:`mrx.relaxation.initial_conditions.potential_two_form`."""
    from mrx.relaxation.initial_conditions import potential_two_form

    B, _ = potential_two_form(seq)
    return B


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
