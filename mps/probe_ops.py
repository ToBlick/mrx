"""Every JAX construct MRX depends on, run on the Apple GPU and against the CPU.

The MPS backend (``jax-mps``, an MLX-backed PJRT plugin) implements a subset
of StableHLO. A missing or wrong handler surfaces inside MRX as a compile
error thrown from a hundred-line kernel or, worse, as a silently wrong
number; either is expensive to bisect. This script pins each construct down
on its own: one small program per op, the same inputs on both backends, the
MPS result compared to the CPU one at a float32 tolerance.

The checks are the ops MRX actually issues, not a survey of JAX:
:func:`mrx.mass._to_quadrature`'s sum-factorisation ``einsum``,
:func:`mrx.extraction_operators._apply_coo`'s gather + ``segment_sum``,
the data-dependent ``while_loop`` of :func:`mrx.solvers.preconditioned_cg`,
the ``lax.scan`` of :func:`mrx.relaxation.chunk_runner`, the spline
``dynamic_slice``, and the dense ``cholesky``/``eigh``/``solve_triangular``
of :func:`mrx.preconditioners._simultaneous_diagonalize_pair`.

64-bit mode is off throughout: MLX has no float64 and the plugin raises
rather than downcasting it, which is what ``MRX_X64=0`` exists for.

    python mps/probe_ops.py            # both backends, compare
    python mps/probe_ops.py --verbose  # also print the tracebacks of failures
"""

from __future__ import annotations

import argparse
import sys
import traceback
from typing import Any, Callable

import jax

jax.config.update("jax_enable_x64", False)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

#: Relative tolerance of the MPS-against-CPU comparison. Loose enough that a
#: different-but-valid float32 summation order passes, tight enough that a
#: wrong handler does not.
RTOL = 2e-5
ATOL = 2e-6

#: ``(name, builder)`` of every check, in the order they are run.
CHECKS: list[tuple[str, Callable[[], Any]]] = []


def check(name: str) -> Callable[[Callable[[], Any]], Callable[[], Any]]:
    """Register ``fn`` under ``name`` as a probe.

    Args:
        name: Identifier printed in the report; name the MRX construct, not
            the JAX primitive, so a failure points at the code that breaks.

    Returns:
        The decorator, which registers and returns the function unchanged.
        The function takes no arguments, builds its own inputs from
        :data:`SEED` so both backends see identical bytes, and returns a
        pytree of arrays to compare.
    """

    def register(fn: Callable[[], Any]) -> Callable[[], Any]:
        CHECKS.append((name, fn))
        return fn

    return register


#: Seed of every input, so the two backends get bit-identical host data.
SEED = 0


def _rng() -> np.random.Generator:
    """A generator reseeded per check, so checks are order-independent."""
    return np.random.default_rng(SEED)


# --------------------------------------------------------------------------
# Contractions: mrx.mass
# --------------------------------------------------------------------------


@check("einsum_sumfact")
def _einsum_sumfact() -> jnp.ndarray:
    """The two-stage sum-factorised contraction of :func:`mrx.mass._to_quadrature`.

    Shapes are a (4, 6, 6) element grid with p=2, the scale of the test
    fixtures: the x contraction ``xqb,xyzbdf->xyzqdf`` then the fused y-z one
    ``yzQD,xyzqD->xyzqQ``.
    """
    rng = _rng()
    ne_x, ne_y, ne_z, q, nl = 4, 6, 6, 3, 3
    Bx = jnp.asarray(rng.standard_normal((ne_x, q, nl)), jnp.float32)
    Byz = jnp.asarray(rng.standard_normal((ne_y, ne_z, q * q, nl * nl)), jnp.float32)
    x_local = jnp.asarray(
        rng.standard_normal((ne_x, ne_y, ne_z, nl, nl, nl)), jnp.float32)

    @jax.jit
    def run(Bx: jnp.ndarray, Byz: jnp.ndarray, x_local: jnp.ndarray) -> jnp.ndarray:
        t1 = jnp.einsum("xqb,xyzbdf->xyzqdf", Bx, x_local)
        t1 = t1.reshape(ne_x, ne_y, ne_z, q, nl * nl)
        return jnp.einsum("yzQD,xyzqD->xyzqQ", Byz, t1)

    return run(Bx, Byz, x_local)


@check("matmul_precision_highest")
def _matmul_precision_highest() -> jnp.ndarray:
    """``precision='highest'`` on a ``dot_general``.

    ``mrx.precision`` sets ``jax_default_matmul_precision='highest'`` to keep
    float32 products off TF32. MLX has no TF32 path, so the setting should be
    a no-op here, but it must not make the handler reject the op.
    """
    rng = _rng()
    a = jnp.asarray(rng.standard_normal((64, 48)), jnp.float32)
    b = jnp.asarray(rng.standard_normal((48, 32)), jnp.float32)
    return jax.jit(lambda a, b: jnp.dot(a, b, precision="highest"))(a, b)


# --------------------------------------------------------------------------
# Indexed ops: mrx.extraction_operators
# --------------------------------------------------------------------------


@check("segment_sum_coo")
def _segment_sum_coo() -> jnp.ndarray:
    """:func:`mrx.extraction_operators._apply_coo` exactly: gather, scale, scatter-add.

    Duplicate segment indices are the point -- the extraction ``E`` sums
    several contributions into one degree of freedom.
    """
    rng = _rng()
    n_col, n_row, nnz = 96, 40, 400
    gather_idx = jnp.asarray(rng.integers(0, n_col, nnz), jnp.int32)
    segment_idx = jnp.asarray(rng.integers(0, n_row, nnz), jnp.int32)
    vals = jnp.asarray(rng.standard_normal(nnz), jnp.float32)
    x = jnp.asarray(rng.standard_normal(n_col), jnp.float32)

    @jax.jit
    def run(vals: jnp.ndarray, gather_idx: jnp.ndarray,
            segment_idx: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        return jax.ops.segment_sum(vals * x[gather_idx], segment_idx,
                                   num_segments=n_row)

    return run(vals, gather_idx, segment_idx, x)


@check("segment_sum_out_of_bounds")
def _segment_sum_out_of_bounds() -> jnp.ndarray:
    """Out-of-range segment indices, which must be dropped and not wrapped.

    jax-mps issue #240 reports ``scatter(mode='drop')`` landing out-of-bounds
    updates in the next row instead of discarding them. MRX does not
    deliberately pass out-of-range indices, but ``segment_sum`` lowers to a
    dropping scatter regardless, so a broken drop is a correctness risk
    wherever the scatter is emitted.
    """
    rng = _rng()
    nnz, n_row = 64, 8
    segment_idx = jnp.asarray(rng.integers(0, n_row * 3, nnz), jnp.int32)
    data = jnp.asarray(rng.standard_normal(nnz), jnp.float32)
    return jax.jit(
        lambda d, s: jax.ops.segment_sum(d, s, num_segments=n_row))(data, segment_idx)


@check("gather_clamped")
def _gather_clamped() -> jnp.ndarray:
    """``x[idx]`` with indices past the end, which JAX clamps rather than drops."""
    rng = _rng()
    x = jnp.asarray(rng.standard_normal(32), jnp.float32)
    idx = jnp.asarray(rng.integers(-8, 40, 24), jnp.int32)
    return jax.jit(lambda x, i: x[i])(x, idx)


@check("dynamic_slice_spline")
def _dynamic_slice_spline() -> jnp.ndarray:
    """The local-support read of :mod:`mrx.spline_bases`: a p+1 window at a
    traced start index, vmapped over evaluation points."""
    rng = _rng()
    p, n = 2, 24
    coeffs = jnp.asarray(rng.standard_normal(n), jnp.float32)
    starts = jnp.asarray(rng.integers(0, n - p - 1, 50), jnp.int32)

    @jax.jit
    def run(coeffs: jnp.ndarray, starts: jnp.ndarray) -> jnp.ndarray:
        return jax.vmap(
            lambda s: jax.lax.dynamic_slice(coeffs, (s,), (p + 1,)))(starts)

    return run(coeffs, starts)


# --------------------------------------------------------------------------
# Control flow: mrx.solvers, mrx.relaxation
# --------------------------------------------------------------------------


@check("while_loop_data_dependent")
def _while_loop_data_dependent() -> tuple[jnp.ndarray, jnp.ndarray]:
    """A preconditioned CG in the shape of :func:`mrx.solvers.preconditioned_cg`.

    The trip count depends on the residual, so the plugin's counted-loop fast
    path (jax-mps #194) does not apply and this runs the general
    ``stablehlo.while``. Returns the solution and the iteration count; the
    count must match the CPU's or the loop condition is being evaluated
    differently.
    """
    rng = _rng()
    n = 48
    root = rng.standard_normal((n, n)).astype(np.float32)
    a = jnp.asarray(root @ root.T + n * np.eye(n, dtype=np.float32), jnp.float32)
    b = jnp.asarray(rng.standard_normal(n), jnp.float32)

    @jax.jit
    def run(a: jnp.ndarray, b: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        def cond(carry):
            _, _, _, rs, it = carry
            return (rs > jnp.float32(1e-10)) & (it < 200)

        def body(carry):
            x, r, p, rs, it = carry
            ap = a @ p
            alpha = rs / jnp.dot(p, ap)
            x = x + alpha * p
            r = r - alpha * ap
            rs_new = jnp.dot(r, r)
            p = r + (rs_new / rs) * p
            return x, r, p, rs_new, it + 1

        x0 = jnp.zeros_like(b)
        carry = (x0, b, b, jnp.dot(b, b), jnp.int32(0))
        x, _, _, _, it = jax.lax.while_loop(cond, body, carry)
        return x, it

    return run(a, b)


@check("scan_stacked_outputs")
def _scan_stacked_outputs() -> tuple[jnp.ndarray, jnp.ndarray]:
    """:func:`mrx.relaxation.chunk_runner`'s ``lax.scan``, stacking a per-step diagnostic.

    jax-mps issue #215 reports the per-iteration cost of a stacking scan
    growing with the trip count. This checks correctness only; the cost shows
    up in the benchmark.
    """
    rng = _rng()
    n, steps = 32, 40
    a = jnp.asarray(rng.standard_normal((n, n)) / n, jnp.float32)
    x0 = jnp.asarray(rng.standard_normal(n), jnp.float32)

    @jax.jit
    def run(a: jnp.ndarray, x0: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        def body(x, _):
            x = jnp.tanh(a @ x)
            return x, jnp.dot(x, x)

        return jax.lax.scan(body, x0, jnp.arange(steps))

    return run(a, x0)


@check("lax_map_batched_jacfwd")
def _lax_map_batched_jacfwd() -> jnp.ndarray:
    """:func:`mrx.geometry.map_jacobian_at`: ``lax.map`` over a ``jacfwd``."""
    rng = _rng()
    pts = jnp.asarray(rng.uniform(0.1, 0.9, (64, 3)), jnp.float32)

    def chart(x: jnp.ndarray) -> jnp.ndarray:
        r, t, z = x
        return jnp.stack([(1.0 + r * jnp.cos(2 * jnp.pi * t)) * jnp.cos(2 * jnp.pi * z),
                          (1.0 + r * jnp.cos(2 * jnp.pi * t)) * jnp.sin(2 * jnp.pi * z),
                          r * jnp.sin(2 * jnp.pi * t)])

    return jax.jit(lambda p: jax.lax.map(jax.jacfwd(chart), p))(pts)


@check("grad_through_spline_sum")
def _grad_through_spline_sum() -> jnp.ndarray:
    """``jax.grad`` of a cancelling contraction, the pattern of the spline
    derivative tables that float32 accuracy was a worry for on GPU."""
    rng = _rng()
    c = jnp.asarray(rng.standard_normal(40), jnp.float32)

    def energy(c: jnp.ndarray) -> jnp.ndarray:
        d = jnp.diff(c)
        return jnp.sum(d * d) + jnp.sum(jnp.sin(c))

    return jax.jit(jax.grad(energy))(c)


# --------------------------------------------------------------------------
# Dense linear algebra: mrx.preconditioners, mrx.metric_lumping_laplacian
# --------------------------------------------------------------------------


@check("cholesky_solve_triangular_eigh")
def _cholesky_solve_triangular_eigh() -> jnp.ndarray:
    """:func:`mrx.preconditioners._simultaneous_diagonalize_pair` verbatim.

    Cholesky, two triangular solves, a symmetric eigendecomposition and a
    third triangular solve, at the 1-D axis size a preconditioner atom uses.
    Only the eigenvalues are returned: eigenvectors are sign- and
    degeneracy-ambiguous and would compare as spuriously different.
    """
    rng = _rng()
    n = 24
    rm = rng.standard_normal((n, n)).astype(np.float32)
    ra = rng.standard_normal((n, n)).astype(np.float32)
    m = jnp.asarray(rm @ rm.T + n * np.eye(n, dtype=np.float32), jnp.float32)
    a = jnp.asarray(ra + ra.T, jnp.float32)

    @jax.jit
    def run(m: jnp.ndarray, a: jnp.ndarray) -> jnp.ndarray:
        lower = jnp.linalg.cholesky(m)
        linv_a = jax.scipy.linalg.solve_triangular(lower, a, lower=True)
        b = jax.scipy.linalg.solve_triangular(lower, linv_a.T, lower=True).T
        b = 0.5 * (b + b.T)
        lam, u = jnp.linalg.eigh(b)
        jax.scipy.linalg.solve_triangular(lower.T, u, lower=False)
        return lam

    return run(m, a)


@check("solve_inv_det")
def _solve_inv_det() -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """The small dense ``solve``/``inv``/``det`` of the geometry and map code."""
    rng = _rng()
    root = rng.standard_normal((3, 3)).astype(np.float32)
    a = jnp.asarray(root @ root.T + 3 * np.eye(3, dtype=np.float32), jnp.float32)
    b = jnp.asarray(rng.standard_normal(3), jnp.float32)

    @jax.jit
    def run(a: jnp.ndarray, b: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        return jnp.linalg.solve(a, b), jnp.linalg.inv(a), jnp.linalg.det(a)

    return run(a, b)


@check("fft2_harmonic_preconditioner")
def _fft2_harmonic_preconditioner() -> jnp.ndarray:
    """:func:`mrx.hessian.harmonic_preconditioner`'s ``fft2``/``ifft2`` pair."""
    rng = _rng()
    x = jnp.asarray(rng.standard_normal((4, 16, 16)), jnp.float32)
    scale = jnp.asarray(rng.uniform(0.5, 2.0, (1, 16, 16)), jnp.float32)

    @jax.jit
    def run(x: jnp.ndarray, scale: jnp.ndarray) -> jnp.ndarray:
        return jnp.fft.ifft2(jnp.fft.fft2(x, axes=(1, 2)) * scale, axes=(1, 2)).real

    return run(x, scale)


@check("sort")
def _sort() -> jnp.ndarray:
    """``jnp.sort``, reached through the median and quantile diagnostics."""
    rng = _rng()
    x = jnp.asarray(rng.standard_normal(128), jnp.float32)
    return jax.jit(jnp.sort)(x)


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------


def _run_on(fn: Callable[[], Any], device: jax.Device) -> Any:
    """Run ``fn`` with ``device`` as the default, returning host NumPy arrays.

    Args:
        fn: A registered check.
        device: The device every array the check creates is placed on.

    Returns:
        A list of NumPy arrays, the flattened result pytree.
    """
    with jax.default_device(device):
        out = fn()
    return [np.asarray(leaf) for leaf in jax.tree_util.tree_leaves(out)]


def _compare(cpu: list[np.ndarray], mps: list[np.ndarray]) -> tuple[bool, str]:
    """Whether the two results agree, and the worst relative difference.

    Args:
        cpu: Reference leaves.
        mps: Leaves from the Apple GPU.

    Returns:
        ``(ok, detail)``; ``detail`` is the largest absolute difference and
        the shape it occurred at, or the mismatch that made comparison
        impossible.
    """
    if len(cpu) != len(mps):
        return False, f"leaf count {len(mps)} != {len(cpu)}"
    worst, where = 0.0, ""
    for i, (c, m) in enumerate(zip(cpu, mps)):
        if c.shape != m.shape:
            return False, f"leaf {i} shape {m.shape} != {c.shape}"
        if c.dtype.kind in "iub":
            if not np.array_equal(c, m):
                return False, f"leaf {i} ({c.dtype}) differs exactly"
            continue
        diff = float(np.max(np.abs(c - m))) if c.size else 0.0
        if diff > worst:
            worst, where = diff, f"leaf {i} {c.shape}"
        if not np.allclose(c, m, rtol=RTOL, atol=ATOL):
            scale = float(np.max(np.abs(c))) or 1.0
            return False, f"leaf {i} {c.shape}: max |diff| {diff:.3e} (scale {scale:.3e})"
    return True, f"max |diff| {worst:.2e}{' at ' + where if where else ''}"


def main(argv: list[str] | None = None) -> int:
    """Run every check on both backends and report.

    Args:
        argv: Command-line arguments; ``None`` reads :data:`sys.argv`.

    Returns:
        Process exit status: 0 if every check matched the CPU, 1 otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--verbose", action="store_true",
                        help="print the traceback of every failing check")
    parser.add_argument("--only", default=None,
                        help="run only checks whose name contains this substring")
    args = parser.parse_args(argv)

    try:
        cpu_device = jax.devices("cpu")[0]
    except RuntimeError:
        print("no CPU backend; cannot compare", file=sys.stderr)
        return 1
    try:
        mps_device = jax.devices("mps")[0]
    except RuntimeError:
        print("no MPS backend: is jax-mps installed in this environment?", file=sys.stderr)
        return 1

    print(f"cpu: {cpu_device}   mps: {mps_device}\n")
    width = max(len(name) for name, _ in CHECKS)
    failures = 0
    for name, fn in CHECKS:
        if args.only and args.only not in name:
            continue
        try:
            reference = _run_on(fn, cpu_device)
        except Exception as exc:  # the CPU is the reference; it should not fail
            print(f"{name:<{width}}  CPU ERROR  {type(exc).__name__}: {exc}")
            failures += 1
            continue
        try:
            actual = _run_on(fn, mps_device)
        except Exception as exc:
            print(f"{name:<{width}}  MPS ERROR  {type(exc).__name__}: "
                  f"{str(exc).splitlines()[0][:120]}")
            if args.verbose:
                traceback.print_exc()
            failures += 1
            continue
        ok, detail = _compare(reference, actual)
        print(f"{name:<{width}}  {'ok  ' if ok else 'WRONG'}      {detail}")
        failures += not ok

    total = sum(1 for n, _ in CHECKS if not args.only or args.only in n)
    print(f"\n{total - failures}/{total} checks match the CPU")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
