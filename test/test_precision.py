"""``mrx.precision``: the working dtype, its epsilons, and the matmul precision.

The last test is the one that matters on an H100: JAX runs float32
``dot_general`` in TF32 by default (10-bit mantissa, ~5e-4 per term), and a
spline derivative is a cancelling contraction -- the W7-X map's
``dR/dtheta`` on the innermost quadrature ring came out 19% wrong and
``det DF`` went negative (2026-08-26). ``mrx.precision`` sets
``jax_default_matmul_precision = 'highest'``; this reproduces the pattern
in miniature so the setting cannot silently go missing.
"""

import os
import subprocess
import sys
import textwrap

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import mrx
from mrx.precision import X64, cast_arrays

_NAME = os.environ.get("MRX_DTYPE", "float32")
_EPS = {"float64": 2.220446049250313e-16, "float32": 1.1920928955078125e-07}


def test_dtype_follows_mrx_dtype():
    """The working dtype is the environment's (float32 by default), 64-bit
    mode follows ``MRX_X64`` (on by default, since the float64 residual of a
    refined solve needs it), and the caster pins arrays to the working
    dtype: a fresh JAX array is float64 under 64-bit mode and says nothing
    about the working dtype."""
    assert mrx.DTYPE == jnp.dtype(_NAME)
    assert mrx.EPS == _EPS[_NAME]
    assert jax.config.jax_enable_x64 == X64
    assert cast_arrays(jnp.zeros(1, dtype=jnp.float64)).dtype == mrx.DTYPE
    assert cast_arrays(np.zeros(1)).dtype == mrx.DTYPE


def test_default_matmul_precision_is_highest():
    assert jax.config.jax_default_matmul_precision == "highest"


#: The variables :mod:`mrx.precision` reads. Cleared from the child of
#: :func:`_import_precision_with` before ``env`` is applied, so a case tests
#: the configuration it names and not the one the suite happens to run in
#: (``slurm/suite.sh`` runs it under three of them).
_PRECISION_ENV = ("MRX_DTYPE", "MRX_RESIDUAL_DTYPE", "MRX_X64")


def _import_precision_with(**env: str) -> subprocess.CompletedProcess:
    """Import :mod:`mrx.precision` in a fresh interpreter under ``env`` alone.

    The module reads its environment once, at import, and sets
    ``jax_enable_x64`` before any array exists; a configuration can
    therefore only be tested from a process that has not imported it yet.

    Args:
        **env: The precision variables to set. Every name in
            :data:`_PRECISION_ENV` that is not given is unset in the child,
            so the defaults under test are the package's own.

    Returns:
        The finished process, with ``stdout`` holding
        ``"<DTYPE> <RESIDUAL_DTYPE> <REFINE> <x64>"`` on success.
    """
    script = textwrap.dedent("""
        import jax
        from mrx.precision import DTYPE, REFINE, RESIDUAL_DTYPE
        print(DTYPE, RESIDUAL_DTYPE, REFINE, jax.config.jax_enable_x64)
    """)
    child = {k: v for k, v in os.environ.items() if k not in _PRECISION_ENV}
    child.update(env)
    # The MPS plugin makes itself the default backend where it is installed;
    # this is a configuration test and must not need a GPU.
    child["JAX_PLATFORMS"] = "cpu"
    return subprocess.run([sys.executable, "-c", script], capture_output=True,
                          text=True, env=child)


def test_x64_off_is_the_plain_float32_configuration():
    """``MRX_X64=0`` turns 64-bit mode off and takes the residual with it.

    This is the configuration of a backend that rejects float64 buffers
    rather than downcasting them (Apple Metal through ``jax-mps``): the
    solves are plain float32, so nothing builds the float64 residual view
    that the backend would refuse.
    """
    done = _import_precision_with(MRX_X64="0")
    assert done.returncode == 0, done.stderr
    assert done.stdout.split() == ["float32", "float32", "False", "False"]


def test_x64_on_is_the_default_and_refines():
    """Without ``MRX_X64`` nothing changes: 64-bit mode on, float64 residual."""
    done = _import_precision_with()
    assert done.returncode == 0, done.stderr
    assert done.stdout.split() == ["float32", "float64", "True", "True"]


@pytest.mark.parametrize("env,message", [
    ({"MRX_X64": "0", "MRX_DTYPE": "float64"}, "float32-only configuration"),
    ({"MRX_X64": "0", "MRX_RESIDUAL_DTYPE": "float64"}, "needs 64-bit mode"),
    ({"MRX_X64": "yes"}, "expected '0' or '1'"),
])
def test_x64_rejects_a_configuration_it_cannot_honour(env, message):
    """A float64 asked of a float32-only build fails at import, not at the
    first buffer transfer several minutes into a run."""
    done = _import_precision_with(**env)
    assert done.returncode != 0
    assert message in done.stderr


