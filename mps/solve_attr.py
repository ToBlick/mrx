"""Which solve applies the Laplacian preconditioner, in one L-BFGS step.

A step applies that atom about 1,482 times and the shifted-stiffness atom
about 61 times. The names below are the functions that hand the atom to a
Krylov method. Each sets a context variable while it runs, and the atom's
``apply`` reports that name. The closure is built while the caller is on
the stack, so the name is a constant of that solve's executable.

Run on the CPU. ``jax.debug.callback`` needs a CPU device, and the iteration
count is the algorithm's rather than the backend's.

    JAX_PLATFORMS=cpu python mps/solve_attr.py
"""

from __future__ import annotations

import contextvars
import os
import time
from collections import Counter

os.environ.setdefault("MRX_X64", "0")

_CALLER: contextvars.ContextVar[int] = contextvars.ContextVar("mrx_solve_caller", default=0)

NAMES = {
    0: "outside a named solve",
    1: "hodge",
    2: "saddle",
    3: "laplacian",
    4: "smoothing",
}


def _install() -> tuple[Counter, Counter]:
    """Count Laplacian-atom and shifted-atom applies, tagged by their caller.

    Returns:
        ``(laplacian counts, shifted counts)``, keyed by :data:`NAMES`.
    """
    import jax
    import jax.numpy as jnp

    import mrx.metric_lumping_laplacian as lump
    import mrx.operators as operators

    lap: Counter = Counter()
    shifted: Counter = Counter()

    def bump_lap(code: int) -> None:
        lap[int(code)] += 1

    def bump_shift(code: int) -> None:
        shifted[int(code)] += 1

    def tag(code: int, fn):
        def wrapped(*args, **kwargs):
            token = _CALLER.set(code)
            try:
                return fn(*args, **kwargs)
            finally:
                _CALLER.reset(token)
        return wrapped

    operators.apply_inverse_laplacian_hodge = tag(1, operators.apply_inverse_laplacian_hodge)
    operators.apply_inverse_laplacian_saddle = tag(2, operators.apply_inverse_laplacian_saddle)
    operators.apply_inverse_laplacian = tag(3, operators.apply_inverse_laplacian)
    operators.apply_inverse_mass_plus_eps_laplace_matrix = tag(
        4, operators.apply_inverse_mass_plus_eps_laplace_matrix)

    orig_apply = lump.MetricLumpingLaplacian.apply

    def apply(self, x):
        jax.debug.callback(bump_lap, jnp.int32(_CALLER.get()))
        return orig_apply(self, x)

    lump.MetricLumpingLaplacian.apply = apply

    orig_shifted = lump.MetricLumpingLaplacian.shifted_stiffness_apply

    def shifted_stiffness_apply(self, eps):
        fn = orig_shifted(self, eps)
        code = _CALLER.get()

        def counted(x, code=code):
            jax.debug.callback(bump_shift, jnp.int32(code))
            return fn(x)
        return counted

    lump.MetricLumpingLaplacian.shifted_stiffness_apply = shifted_stiffness_apply

    import mrx.solvers as solvers
    cg_iters: list[tuple[int, int]] = []
    orig_cg = solvers.solve_singular_cg

    def solve_singular_cg(*args, **kwargs):
        x, info = orig_cg(*args, **kwargs)
        code = _CALLER.get()

        def record(n, code=code):
            cg_iters.append((code, int(n)))

        jax.debug.callback(record, jnp.abs(info))
        return x, info

    solvers.solve_singular_cg = solve_singular_cg
    operators.solve_singular_cg = solve_singular_cg
    return lap, shifted, cg_iters


def main() -> None:
    """One L-BFGS step at the benchmark mesh, then the call split."""
    import jax
    import numpy as np

    from mrx.geometry import build_sequence
    from mrx.initial_conditions import initial_field
    from mrx.nullspace import compute_nullspaces
    from mrx.relaxation import TimeStepper, initial_state

    print("setup", flush=True)
    t0 = time.perf_counter()
    seq, _ = build_sequence("data/wout_li383_low_res_reference.nc", (12, 24, 12), 3)
    compute_nullspaces(seq, gap_sweeps=0, verbose=False)
    B, _ = initial_field(seq)
    ts = TimeStepper(seq=seq, cfl=0.5, history_size=1, velocity_smoothing_order=1,
                     velocity_smoothing_scale=None)
    lap, shifted, cg_iters = _install()
    state = initial_state(jax.numpy.asarray(np.asarray(B)), ts)
    jax.block_until_ready(state.B_n)
    lap.clear()
    shifted.clear()
    print(f"  built in {time.perf_counter() - t0:.0f} s", flush=True)

    t0 = time.perf_counter()
    state = ts.relaxation_step(state)
    jax.block_until_ready(state.B_n)
    print(f"one step {time.perf_counter() - t0:.1f} s", flush=True)

    def show(title: str, counts: Counter) -> None:
        total = sum(counts.values())
        print(f"\n{title}: {total}")
        for code, name in NAMES.items():
            n = counts[code]
            if n:
                print(f"  {name:<24} {n:6d}  {n / total:6.1%}")

    show("Laplacian atom", lap)
    show("shifted stiffness atom", shifted)
    print("\nCG solves (caller, |iterations|)")
    for code, n in cg_iters:
        print(f"  {NAMES.get(code, code):<24} {n:6d}")


if __name__ == "__main__":
    main()
