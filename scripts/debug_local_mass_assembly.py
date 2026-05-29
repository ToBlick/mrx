"""Phase-1 proof of concept: element-local sum-factorized M0 assembly.

Validates a new element-local assembler for the k=0 mass matrix against the
existing global `assemble_scalar` path, and benchmarks how the two scale.

Run interactively in VS Code as a Jupyter file (cells delimited by `# %%`),
or top-to-bottom as a regular script.
"""

# %% Imports
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

import time

import jax
import jax.numpy as jnp
import numpy as np

from mrx.derham_sequence import DeRhamSequence
from mrx.mappings import toroid_map

jax.config.update("jax_enable_x64", True)
print("devices:", jax.devices())


# %% Element-local 1D basis evaluator
def evaluate_basis_local(basis, x_q_flat, q_per_elem):
    """Evaluate one of the 1D primal bases on each element at its local quad points.

    Parameters
    ----------
    basis : SplineBasis
        Primal spline basis (degree p, n DOFs).
    x_q_flat : (n_elem * q,) array
        Composite Gauss quadrature points along the axis, ordered element-by-element.
    q_per_elem : int
        Number of Gauss points per knot interval.

    Returns
    -------
    B_loc : (n_elem, q_per_elem, p+1) array
        Values of the locally-active bases at the local quad points.
    gdof : (n_elem, p+1) int array
        Global DOF index of each local basis on each element.
    """
    p = basis.p
    n = basis.n
    n_local = p + 1
    if basis.type == "periodic":
        n_elem = n
        elems = jnp.arange(n_elem)
        ks = jnp.arange(n_local)
        gdof = (elems[:, None] + ks[None, :]) % n
    elif basis.type == "clamped":
        n_elem = n - p
        elems = jnp.arange(n_elem)
        ks = jnp.arange(n_local)
        gdof = elems[:, None] + ks[None, :]
    else:
        raise NotImplementedError(basis.type)

    x_local = x_q_flat.reshape(n_elem, q_per_elem)

    def eval_e(x_e, dof_e):
        # x_e: (q,), dof_e: (p+1,)
        return jax.vmap(
            lambda x: jax.vmap(lambda i: basis(x, i))(dof_e)
        )(x_e)

    B_loc = jax.vmap(eval_e, in_axes=(0, 0))(x_local, gdof)  # (n_elem, q, p+1)
    return B_loc, gdof


# %% Element-local sum-factorized M0 assembler (dense output for now)
def _m0_local_dense_pure(Bx, By, Bz, gx, gy, gz, J_split, wx, wy, wz,
                         n_dof_x, n_dof_y, n_dof_z):
    """Pure inner kernel: element-local sum-factorized M0 → dense (n_dof, n_dof).

    Inputs are static-shape arrays; sizes (n_dof_*, etc.) are static Python ints.
    """
    ne_x, qx, nlx = Bx.shape
    ne_y, qy, nly = By.shape
    ne_z, qz, nlz = Bz.shape

    def element_block(Bx_e, By_e, Bz_e, J_e, wx_e, wy_e, wz_e):
        W = J_e * wx_e[:, None, None] * wy_e[None, :, None] * wz_e[None, None, :]
        A = jnp.einsum('qa,qb,qrs->abrs', Bx_e, Bx_e, W)
        Bm = jnp.einsum('rc,rd,abrs->abcds', By_e, By_e, A)
        C = jnp.einsum('se,sf,abcds->abcdef', Bz_e, Bz_e, Bm)
        return C

    block_fn = jax.vmap(jax.vmap(jax.vmap(
        element_block,
        in_axes=(None, None, 0, 0, None, None, 0)
    ), in_axes=(None, 0, None, 0, None, 0, None)),
        in_axes=(0, None, None, 0, 0, None, None))
    blocks = block_fn(Bx, By, Bz, J_split, wx, wy, wz)

    n_dof = n_dof_x * n_dof_y * n_dof_z

    ex = jnp.arange(ne_x).reshape(ne_x, 1, 1, 1, 1, 1, 1, 1, 1)
    ey = jnp.arange(ne_y).reshape(1, ne_y, 1, 1, 1, 1, 1, 1, 1)
    ez = jnp.arange(ne_z).reshape(1, 1, ne_z, 1, 1, 1, 1, 1, 1)
    k1 = jnp.arange(nlx).reshape(1, 1, 1, nlx, 1, 1, 1, 1, 1)
    l1 = jnp.arange(nlx).reshape(1, 1, 1, 1, nlx, 1, 1, 1, 1)
    k2 = jnp.arange(nly).reshape(1, 1, 1, 1, 1, nly, 1, 1, 1)
    l2 = jnp.arange(nly).reshape(1, 1, 1, 1, 1, 1, nly, 1, 1)
    k3 = jnp.arange(nlz).reshape(1, 1, 1, 1, 1, 1, 1, nlz, 1)
    l3 = jnp.arange(nlz).reshape(1, 1, 1, 1, 1, 1, 1, 1, nlz)

    gx_row = gx[ex, k1]
    gy_row = gy[ey, k2]
    gz_row = gz[ez, k3]
    gx_col = gx[ex, l1]
    gy_col = gy[ey, l2]
    gz_col = gz[ez, l3]

    row = (gx_row * n_dof_y * n_dof_z + gy_row * n_dof_z + gz_row).astype(jnp.int32)
    col = (gx_col * n_dof_y * n_dof_z + gy_col * n_dof_z + gz_col).astype(jnp.int32)

    row_b = jnp.broadcast_to(row, blocks.shape).reshape(-1)
    col_b = jnp.broadcast_to(col, blocks.shape).reshape(-1)
    vals = blocks.reshape(-1)

    M = jnp.zeros((n_dof, n_dof), dtype=vals.dtype).at[row_b, col_b].add(vals)
    return M


_m0_local_dense_pure_jit = jax.jit(
    _m0_local_dense_pure, static_argnums=(10, 11, 12)
)


def _prepare_inputs(seq):
    """Pull element-local arrays from `seq` (no JIT)."""
    bx, by, bz = seq.basis_0.Λ
    nx, ny, nz = seq.quad.nx, seq.quad.ny, seq.quad.nz

    qx = nx // (bx.n if bx.type == "periodic" else bx.n - bx.p)
    qy = ny // (by.n if by.type == "periodic" else by.n - by.p)
    qz = nz // (bz.n if bz.type == "periodic" else bz.n - bz.p)

    Bx, gx = evaluate_basis_local(bx, seq.quad.x_x, qx)
    By, gy = evaluate_basis_local(by, seq.quad.x_y, qy)
    Bz, gz = evaluate_basis_local(bz, seq.quad.x_z, qz)

    ne_x, ne_y, ne_z = Bx.shape[0], By.shape[0], Bz.shape[0]

    wx = seq.quad.w_x.reshape(ne_x, qx)
    wy = seq.quad.w_y.reshape(ne_y, qy)
    wz = seq.quad.w_z.reshape(ne_z, qz)

    J = seq.geometry.jacobian_j.reshape(ny, nx, nz).transpose(1, 0, 2)
    J = J.reshape(ne_x, qx, ne_y, qy, ne_z, qz).transpose(0, 2, 4, 1, 3, 5)

    return (Bx, By, Bz, gx, gy, gz, J, wx, wy, wz,
            int(bx.n), int(by.n), int(bz.n))


def assemble_m0_local_dense(seq):
    args = _prepare_inputs(seq)
    return _m0_local_dense_pure(*args)


def assemble_m0_local_dense_jit(seq):
    args = _prepare_inputs(seq)
    return _m0_local_dense_pure_jit(*args)


# %% Build small reference case and validate
def build_seq(n, p, polar=False):
    ns = (n, 2 * n, n)
    ps = (p, p, p)
    q = 2 * p
    types = ("clamped", "periodic", "periodic")
    F = toroid_map(epsilon=1 / 3)
    seq = DeRhamSequence(ns, ps, q, types, polar=polar)
    seq.set_map(F)
    seq.evaluate_1d()
    return seq


# Small validation: polar=False keeps M0 in raw tensor-product DOF space,
# which is what _assemble_mass_block produces. (Polar extraction is applied
# afterward as E M E.T -- it is post-multiplicative and orthogonal to the
# assembly itself.)
print("\n=== Validation: n=8, p=2, polar=False ===")
seq_small = build_seq(n=8, p=2, polar=False)
seq_small.assemble_mass_matrix(0)
M0_ref = jnp.asarray(seq_small.m0.todense())
M0_new = assemble_m0_local_dense(seq_small)
err = float(jnp.linalg.norm(M0_new - M0_ref) / jnp.linalg.norm(M0_ref))
print(f"  shape ref = {M0_ref.shape}, new = {M0_new.shape}")
print(f"  relative Frobenius error: {err:.3e}")
print(f"  max abs error: {float(jnp.max(jnp.abs(M0_new - M0_ref))):.3e}")
assert err < 1e-12, f"validation failed: rel err {err:.3e}"
print("  OK")


# %% Repeat validation at a couple of (n, p) combos
for n_v, p_v in [(8, 1), (8, 3), (12, 2), (16, 2)]:
    print(f"\n=== Validation: n={n_v}, p={p_v}, polar=False ===")
    seq_v = build_seq(n=n_v, p=p_v, polar=False)
    seq_v.assemble_mass_matrix(0)
    M0_ref = jnp.asarray(seq_v.m0.todense())
    M0_new = assemble_m0_local_dense(seq_v)
    err = float(jnp.linalg.norm(M0_new - M0_ref) / jnp.linalg.norm(M0_ref))
    max_abs = float(jnp.max(jnp.abs(M0_new - M0_ref)))
    print(f"  rel err = {err:.3e}, max abs err = {max_abs:.3e}")
    if err >= 1e-10:
        print("  *** MISMATCH ***")


# %% Timing comparison
def time_one(fn, n_warmup=1, n_repeat=3):
    for _ in range(n_warmup):
        out = fn()
        jax.block_until_ready(out)
    times = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        out = fn()
        jax.block_until_ready(out)
        times.append(time.perf_counter() - t0)
    return min(times), out


print("\n=== Timing comparison (assembly only; first call includes JIT) ===")
print(f"{'n':>4} {'p':>2}   {'global (s)':>12} {'local (s)':>12} {'speedup':>8}")
for n_v, p_v in [(8, 2), (12, 2), (16, 2), (8, 3), (12, 3), (16, 3), (8, 4), (12, 4)]:
    seq_v = build_seq(n=n_v, p=p_v, polar=False)

    def run_ref():
        seq_v.assemble_mass_matrix(0)
        return seq_v.m0.data

    def run_new():
        return assemble_m0_local_dense_jit(seq_v)

    t_ref, _ = time_one(run_ref, n_warmup=1, n_repeat=3)
    t_new, _ = time_one(run_new, n_warmup=1, n_repeat=3)
    print(f"{n_v:>4} {p_v:>2}   {t_ref:>12.4f} {t_new:>12.4f} {t_ref/t_new:>8.2f}x")


# %% Done
print("\nPhase 1 done. If validation passes and local timing scales as O(n^3),"
      " we can roll this into mrx.assembly as `assemble_scalar_local` and"
      " switch k=0 in _assemble_mass_block to use it.")
