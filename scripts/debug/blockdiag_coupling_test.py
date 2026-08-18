"""Is the core<->bulk coupling worth its cost?  k=0 stiffness AND the masses.

The production k=0 Laplacian preconditioner is a block factorization::

    y   = A_bb^-1 r_b
    z   = schur_inv (r_c - C0^T y)          schur = A_cc - C0^T A_bb^-1 C0
    x_b = y - A_bb^-1 (C0 z)                x_c = z

It carries two costs: ``core_coupling`` is a dense ``bulk x 3 n_z`` block --
``O(N n_z)``, 780 MB at 64x128x64, the same pathology that retired
``coupling_sb`` -- and the Schur needs ``3 n_z`` exact bulk CG solves at
assembly.

This tests dropping BOTH: a plain block-diagonal preconditioner

    x_c = A_cc^-1 r_c        (dense, 3 n_z square -- 0.3 MB, exact)
    x_b = A_bb^-1 r_b        (the atom)

against the full factorization, for each bulk atom. If block-diagonal is close,
``core_coupling`` and the Schur assembly both go away.

Full grid, **dbc only**: the free/Neumann K_0 is singular (constants) and plain
PCG diverges on it -- section 7.6.

The same question for the mass: dense exact inverse on the surgery (coupled)
rows + Kronecker on the bulk, coupling off, vs raw_kron which never splits.
"""
import argparse
import jax.numpy as jnp
import numpy as np

import jax
import mrx
from mrx.derham_sequence import DeRhamSequence
from mrx.local_assembly import build_matrixfree_mass_apply
from mrx.mappings import rotating_ellipse_map, toroid_map
from mrx.operators import (_apply_k0_tensor_hodge_bulk_inverse,
                           _apply_k0_tensor_hodge_surgery_to_bulk_coupling,
                           _apply_k0_tensor_hodge_core_block,
                           _assemble_dense_from_apply,
                           _assemble_k0_greville_bulk_factors,
                           _build_k0_tensor_hodge_preconditioner_factors,
                           _core_size, _symmetrize, apply_stiffness,
                           assemble_incidence_operators)
from mrx.solvers import solve_singular_cg
from mrx.preconditioners import (_extraction_gram_inverse, _symmetric_pseudoinverse,
                                 apply_mass_raw_kron_preconditioner,
                                 build_mass_jacobi_pair,
                                 build_mass_raw_kron_factors)

from mrx.experimental.modal_radial import (fd_harmonic_bulk_data,
                                          modal_perk_apply,
                                          modal_perk_bulk_data)

p = argparse.ArgumentParser()
p.add_argument("--ns", type=int, nargs=3, default=(8, 16, 16))
p.add_argument("--p", type=int, default=3)
p.add_argument("--tol", type=float, default=1e-10)
p.add_argument("--maxit", type=int, default=4000)
p.add_argument("--map-batch", type=int, default=256)
p.add_argument("--h5", default="data/W7X-vacuum.h5")
p.add_argument("--stride", type=int, default=2)
args = p.parse_args()
mrx.MAP_BATCH_SIZE_INNER = args.map_batch
NS, P = tuple(args.ns), args.p
TYPES = ("clamped", "periodic", "periodic")
print(f"block-diagonal / coupling test: ns={NS} p={P}\n", flush=True)


def pcg(A, b, Minv, tol, maxit):
    x = jnp.zeros_like(b); r = b - A(x); z = Minv(r); q = z
    rz = float(r @ z); nb = float(jnp.linalg.norm(b))
    for i in range(1, maxit + 1):
        Aq = A(q); den = float(q @ Aq)
        if den <= 0.0:
            return -i
        al = rz / den; x = x + al * q; r = r - al * Aq
        if float(jnp.linalg.norm(r)) / nb < tol:
            return i
        z = Minv(r); rzn = float(r @ z); q = z + (rzn / rz) * q; rz = rzn
    return maxit


# ------------------------------------------------------------------ k=0 stiffness
def stiffness_arms(seq, ops, dbc=True):
    cs = _core_size(seq)
    size = int(seq.n0_dbc if dbc else seq.n0)

    def K(x):
        return apply_stiffness(seq, ops, x, 0, dirichlet=dbc)
    K(jnp.zeros(size))                      # eager warmup before tracing

    A_cc = _symmetrize(_assemble_dense_from_apply(
        lambda rc: _apply_k0_tensor_hodge_core_block(seq, ops, cs, rc, dirichlet=dbc),
        cs, sequential=True))
    A_cc_inv = _symmetric_pseudoinverse(A_cc)

    fd = _assemble_k0_greville_bulk_factors(seq, dirichlet=dbc)
    f_fd = _build_k0_tensor_hodge_preconditioner_factors(
        core_size=cs, schur_inv=jnp.eye(cs), bulk_data=fd)
    fh = fd_harmonic_bulk_data(seq, dirichlet=dbc)
    f_fh = _build_k0_tensor_hodge_preconditioner_factors(
        core_size=cs, schur_inv=jnp.eye(cs), bulk_data=fh)
    pk = modal_perk_bulk_data(seq, dirichlet=dbc)
    atoms = {"fd": lambda rb: _apply_k0_tensor_hodge_bulk_inverse(f_fd, rb),
             "fd-harm": lambda rb: _apply_k0_tensor_hodge_bulk_inverse(f_fh, rb),
             "per-k": lambda rb: modal_perk_apply(pk, rb)}

    def blockdiag(atom):
        def M(r):
            return jnp.concatenate([A_cc_inv @ r[:cs], atom(r[cs:])])
        return M

    # WITH coupling: the production block factorization. schur_inv is built from
    # EXACT bulk solves, so it is atom-independent and may be reused verbatim
    # under a different bulk inverse (section 7.1).
    def build_coupled(atom):
        bulk_solve = jax.jit(lambda b: solve_singular_cg(
            bulk_operator, b, precond_matvec=atom, maxiter=1000, tol=1e-12)[0])
        C0 = _assemble_dense_from_apply(
            lambda rc: _apply_k0_tensor_hodge_surgery_to_bulk_coupling(
                seq, ops, cs, rc, dirichlet=dbc), cs, sequential=True)
        solves = jnp.stack([bulk_solve(C0[:, i]) for i in range(cs)], axis=1)
        schur_inv = _symmetric_pseudoinverse(_symmetrize(A_cc - C0.T @ solves))

        def M(r):
            y = atom(r[cs:])
            z = schur_inv @ (r[:cs] - C0.T @ y)
            return jnp.concatenate([z, y - atom(C0 @ z)])
        return M

    def bulk_operator(xb):
        full = jnp.zeros((size,)).at[cs:].set(xb)
        return apply_stiffness(seq, ops, full, 0, dirichlet=dbc)[cs:]

    return size, cs, K, atoms, blockdiag, build_coupled


# ------------------------------------------------------------------ mass
def mass_arms(seq, k, dbc):
    e = getattr(seq, f"e{k}_dbc" if dbc else f"e{k}")
    n = int(e.shape[0])
    ap = build_matrixfree_mass_apply(seq, k)

    def A(x):
        return e @ ap(e.T @ x)
    A(jnp.zeros(n))

    fac = build_mass_raw_kron_factors(seq, k, dirichlet=dbc)
    rk = lambda r: apply_mass_raw_kron_preconditioner(fac, e, r)

    coupled, _, _ = _extraction_gram_inverse(e)
    coupled = np.asarray(coupled) if coupled is not None else np.zeros(0, dtype=int)
    mask = np.zeros(n, dtype=bool); mask[coupled] = True
    bulk = np.flatnonzero(~mask)
    # dense exact A_ss over the surgery rows (|coupled| = O(n_z) applies)
    if coupled.size:
        cols = [np.asarray(A(jnp.zeros(n).at[int(i)].set(1.0)))[coupled] for i in coupled]
        A_ss = _symmetrize(jnp.asarray(np.stack(cols, axis=1)))
        A_ss_inv = _symmetric_pseudoinverse(A_ss)
    else:
        A_ss_inv = jnp.zeros((0, 0))
    ci = jnp.asarray(coupled); bi = jnp.asarray(bulk)

    def blockdiag(r):
        out = jnp.zeros_like(r)
        rb = jnp.zeros_like(r).at[bi].set(r[bi])
        out = out.at[bi].set(apply_mass_raw_kron_preconditioner(fac, e, rb)[bi])
        if ci.size:
            out = out.at[ci].set(A_ss_inv @ r[ci])
        return out
    jac = build_mass_jacobi_pair(seq, ap, k)
    dj = jac.dbc if dbc else jac.free
    return n, A, {"jacobi": lambda r: dj * r, "raw_kron": rk, "blockdiag": blockdiag}


def w7x_map(path, stride):
    import h5py
    from mrx.differential_forms import DiscreteFunction
    from mrx.projectors import _solve_tensor_collocation_axis
    with h5py.File(path, "r") as h:
        at = dict(h.attrs); nfp = int(at["nfp"])
        nr, nt, nz = (int(at[k]) for k in ("n_rho", "n_theta", "n_zeta"))
        R = np.asarray(h["R"]).reshape(nr, nt, nz)
        Z = np.asarray(h["Z"]).reshape(nr, nt, nz)
        ep = np.asarray(h["eval_points"]).reshape(nr, nt, nz, 3)
    sl = (slice(None, None, stride),) * 3
    R, Z, ep = R[sl], Z[sl], ep[sl]
    rho, th, ze = ep[:, 0, 0, 0], ep[0, :, 0, 1], ep[0, 0, :, 2]
    R0 = float(np.mean(R)); R, Z = R / R0, Z / R0
    fs = DeRhamSequence((len(rho), len(th), len(ze)), (P,) * 3, 2 * P, TYPES, polar=False)
    fs.evaluate_1d()
    br, bt, bz = fs.basis_0.Λ
    colls = (jnp.asarray(br.collocation_matrix(jnp.asarray(rho))),
             jnp.asarray(bt.collocation_matrix(jnp.asarray(th))),
             jnp.asarray(bz.collocation_matrix(jnp.asarray(ze))))
    def fit(v):
        c = jnp.asarray(v)
        for ax, cm in enumerate(colls):
            c = _solve_tensor_collocation_axis(cm, c, axis=ax)
        return c
    Rh = DiscreteFunction(fs.e0 @ fit(R).reshape(-1), fs.basis_0, fs.e0)
    Zh = DiscreteFunction(fs.e0 @ fit(Z).reshape(-1), fs.basis_0, fs.e0)
    an = 2 * jnp.pi / nfp
    def make(sg):
        def f(x):
            ang = an * x[2]; rr = Rh(x)[0]
            return jnp.array([rr * jnp.cos(ang), sg * rr * jnp.sin(ang), Zh(x)[0]])
        return f
    rng0 = np.random.default_rng(0)
    pts = jnp.asarray(rng0.uniform(0.05, 0.95, size=(256, 3)))
    d = jax.vmap(lambda x: jnp.linalg.det(jax.jacfwd(make(1.0))(x)))(pts)
    return make(1.0 if float(jnp.median(d)) > 0 else -1.0)


for gname, mk in (("toroid", lambda: toroid_map(epsilon=1 / 3, R0=1.0)),
                  ("rot-ellipse nfp3",
                   lambda: rotating_ellipse_map(eps=0.33, kappa=1.5, nfp=3)),
                  ("W7-X", lambda: w7x_map(args.h5, args.stride))):
    print(f"===== {gname} =====", flush=True)
    seq = DeRhamSequence(NS, (P,) * 3, 2 * P, TYPES, polar=True, tol=1e-12,
                         maxiter=1000, betti_numbers=(1, 1, 0, 0))
    seq.evaluate_1d(); seq.set_map(mk())
    _j = np.asarray(seq.geometry.jacobian_j)
    if not np.all(np.isfinite(_j)) or _j.min() <= 0.0:
        print("  SKIP: det(J) not positive/finite\n", flush=True); continue
    print(f"  det(J) in [{_j.min():.3e}, {_j.max():.3e}]", flush=True)
    ops = assemble_incidence_operators(seq); seq.set_operators(ops)

    size, cs, K, atoms, blockdiag, build_coupled = stiffness_arms(seq, ops, dbc=True)
    rng = np.random.default_rng(0)
    b = jnp.asarray(rng.standard_normal(size))
    print(f"  k=0 stiffness (FULL grid, dbc)  n={size}  core=3n_z={cs}", flush=True)
    for aname, atom in atoms.items():
        it_bd = pcg(K, b, blockdiag(atom), args.tol, args.maxit)
        it_cp = pcg(K, b, build_coupled(atom), args.tol, args.maxit)
        pen = (it_bd / it_cp) if it_cp > 0 else float("nan")
        print(f"    {aname:6s}: Schur+coupling {it_cp:>5}   block-diag {it_bd:>5}"
              f"   coupling worth {pen:.2f}x", flush=True)

    for k in (0, 1, 2, 3):
        for dbc in (False, True):
            n, A, arms = mass_arms(seq, k, dbc)
            bb = jnp.asarray(rng.standard_normal(n))
            res = {nm: pcg(A, bb, f, args.tol, args.maxit) for nm, f in arms.items()}
            print(f"  mass k={k} {'dbc ' if dbc else 'free'} n={n:>6}  "
                  + "  ".join(f"{nm}={v}" for nm, v in res.items()), flush=True)
    print(flush=True)
