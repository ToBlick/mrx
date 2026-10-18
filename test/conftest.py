"""Shared fixtures of the test suite: the li383 equilibrium at spline degree 2 in three representations of the
domain with the same mesh per field period (docs/source/concepts/testing_strategy.md):

- ``half``: the production sequence, one field period with stellarator symmetry, integrating half of it.
- ``period``: one field period without the reflection.
- ``full``: the whole torus as one period (``nfp = 1``), three times the toroidal splines.

The run time of the suite is dominated by XLA compilation, so a test costs the distinct solves it compiles,
not the size of the mesh. Compiled programs are cached on disk between runs (see below).
"""
import os
import time

import jax
import pytest

# Compiled programs are cached on disk, keyed by their HLO, so a stale entry
# costs only a recompile. MRX_XLA_CACHE names the directory. An empty value disables the cache.
_CACHE = os.environ.get("MRX_XLA_CACHE", os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                                       "outputs", "xla_cache"))
if _CACHE:
    jax.config.update("jax_compilation_cache_dir", _CACHE)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.1)

#: The wout geometry, tracked in the repository.
GEOMETRY = "data/wout_li383_low_res_reference.nc"
#: Resolution (r, theta, zeta) per field period and the spline degree.
NS, P = (12, 20, 12), 2
#: The three representations of the li383 domain: (symmetry, nfp override, toroidal splines).
DOMAINS = {"half": ("stellarator", None, NS[2]), "period": ("field-period", None, NS[2]),
           "full": ("none", 1, 3 * NS[2])}


def _build(name):
    from mrx.geometry import build_sequence
    from mrx.nullspace import compute_nullspaces

    symmetry, nfp, n_zeta = DOMAINS[name]
    t0 = time.perf_counter()
    s = build_sequence(GEOMETRY, (NS[0], NS[1], n_zeta), P, nfp=nfp, symmetry=symmetry)
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
def b0(seq):
    """The equilibrium's own field ``B = dA'``, a divergence-free 2-form built by
    :func:`mrx.relaxation.initial_conditions.potential_two_form`."""
    from mrx.relaxation.initial_conditions import potential_two_form

    B, _, _ = potential_two_form(seq)
    return B
