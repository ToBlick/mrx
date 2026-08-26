"""Build-time benchmark of the extraction operators, the tensor-product
stencil assemblers and the nullspace helpers, with an optional check
against a reference checkout of the same functions.

Builds a polar p=3 sequence on ``mrx.mappings.toroid_map`` at the test
resolution (8, 16, 8) and at n=16, and times

* ``PolarExtractionOperator.build_extraction`` and
  ``BoundaryOperator.build_extraction`` (all operators of a sequence),
* ``ring1_control_points``,
* ``assemble_scalar`` / ``assemble_vectorial``,
* ``compute_nullspaces_iterative`` and ``get_stiffness_nullspace`` (n=8 only).

With ``--old-root PATH`` the same functions are loaded from another checkout
and every result is compared: identical COO pattern for the extraction
operators, dense agreement to roundoff for the assembled matrices, and
subspace agreement for the nullspace bases.

Usage (GPU job)::

    SCRIPT=scripts/benchmark/benchmark_assembly_build.py \
        ARGS="--old-root /path/to/reference/checkout" bash slurm/run.sh
"""
import argparse
import importlib.util
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

import mrx
from mrx import assembly, extraction_operators, nullspace
from mrx.derham_sequence import DeRhamSequence
from mrx.extraction_operators import get_xi
from mrx.geometry import grad_1d
from mrx.io import project_sampled_field
from mrx.mappings import toroid_map
from mrx.operators import (assemble_derivative_operators,
                           assemble_incidence_operators,
                           assemble_mass_jacobi_preconditioner,
                           assemble_projection_operators)

P = 3
TYPES = ("clamped", "periodic", "periodic")
BETTI = (1, 1, 0, 0)


def load_old(root, name):
    spec = importlib.util.spec_from_file_location(
        f"old_{name}", f"{root}/mrx/{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def timed(label, fn, repeat=1):
    """Run ``fn`` ``repeat`` times, block on the result, return the best time."""
    best = np.inf
    out = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn()
        jax.block_until_ready(out)
        best = min(best, time.perf_counter() - t0)
    print(f"  {label:<44s} {best:9.3f} s", flush=True)
    return out, best


def coo(mf):
    return (np.asarray(mf.rows), np.asarray(mf.cols), np.asarray(mf.vals))


def same_coo(a, b):
    return (a[0].shape == b[0].shape and np.array_equal(a[0], b[0])
            and np.array_equal(a[1], b[1]) and np.array_equal(a[2], b[2]))


def build_seq(ns):
    seq = DeRhamSequence(ns, (P,) * 3, 2 * P, TYPES, polar=True,
                         tol=1e-12, maxiter=1000, betti_numbers=BETTI)
    seq.evaluate_1d()
    return seq


def install_toroid(seq, n_sample=16):
    seq.assemble_reference_mass_matrix()
    F = toroid_map(epsilon=1 / 3, R0=1.0)
    ax = jnp.linspace(0.0, 1.0, n_sample)
    ri, ci, zi = jnp.meshgrid(ax, ax, ax, indexing="ij")
    pts = jnp.stack([ri.ravel(), ci.ravel(), zi.ravel()], axis=1)
    samples = jax.vmap(F)(pts)
    coeffs = jnp.stack([
        project_sampled_field((ax, ax, ax), samples[:, i], seq, k=0,
                              dirichlet=False, reference_domain=True)
        for i in range(3)], axis=0)
    seq.set_spline_map(coeffs)
    ops = assemble_incidence_operators(seq)
    ops = assemble_derivative_operators(seq, seq.geometry, operators=ops)
    ops = assemble_projection_operators(seq, operators=ops)
    ops = assemble_mass_jacobi_preconditioner(seq, ops, ks=(0, 1, 2, 3))
    seq.set_operators(ops)
    return seq


def extraction_operators_of(seq, mod):
    xi = get_xi(seq.ns[1])
    polar = [mod.PolarExtractionOperator(L, xi, bc)
             for L in (seq.basis_0, seq.basis_1, seq.basis_2, seq.basis_3)
             for bc in (False, True)]
    boundary = [mod.BoundaryOperator(L, t)
                for L in (seq.basis_0, seq.basis_1, seq.basis_2, seq.basis_3)
                for t in (("none",) * 3, ("dirichlet", "none", "none"),
                          ("dirichlet", "right", "left"))]
    return polar, boundary


def assembler_inputs(seq):
    quad_shape = (seq.quad.ny, seq.quad.nx, seq.quad.nz)
    R, T, Z = seq.basis_r_jk, seq.basis_t_jk, seq.basis_z_jk
    dR, dT, dZ = seq.d_basis_r_jk, seq.d_basis_t_jk, seq.d_basis_z_jk
    gr, gt, gz = (grad_1d(dR, TYPES[0]), grad_1d(dT, TYPES[1]),
                  grad_1d(dZ, TYPES[2]))
    w = seq.quad.w
    W_3x3 = w[:, None, None] * jnp.eye(3)
    scalar = dict(
        args=(R, T, Z, R, T, Z, w, quad_shape, seq.basis_0.shape[0],
              seq.basis_0.pr, seq.basis_0.pt, seq.basis_0.pz))
    # k=1 -> k=2 projection block (one identity term per component)
    row_terms = [[(0, R, dT, dZ, +1)], [(1, dR, T, dZ, +1)], [(2, dR, dT, Z, +1)]]
    col_terms = [[(0, dR, T, Z, +1)], [(1, R, dT, Z, +1)], [(2, R, T, dZ, +1)]]
    vectorial = dict(
        args=(row_terms, col_terms, W_3x3, quad_shape, list(seq.basis_2.shape),
              seq.basis_2.pr),
        kwargs=dict(col_comp_shapes=list(seq.basis_1.shape)))
    # curl-curl shaped input (two signed terms per component): exercises the
    # multi-term stacking of assemble_vectorial
    curl_terms = [[(1, dR, T, gz, +1), (2, dR, gt, Z, -1)],
                  [(0, R, dT, gz, -1), (2, gr, dT, Z, +1)],
                  [(0, R, gt, dZ, +1), (1, gr, T, dZ, -1)]]
    curlcurl = dict(
        args=(curl_terms, curl_terms, W_3x3, quad_shape, list(seq.basis_1.shape),
              seq.basis_1.pr))
    return scalar, vectorial, curlcurl


def dense(bcoo):
    return np.asarray(bcoo.todense())


def pattern(bcoo):
    idx = np.asarray(bcoo.indices)
    return idx[:, 0], idx[:, 1]


def check_close(label, a, b, rtol=1e-12):
    err = np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300)
    print(f"    {label:<40s} max rel diff {err:.2e}", flush=True)
    assert err < rtol, label


def run_resolution(ns, old, do_nullspace):
    print(f"\n=== ns={ns} p={P} polar toroid ===", flush=True)
    seq = build_seq(ns)

    print("extraction operators (8 polar + 12 boundary builds):")
    polar, boundary = extraction_operators_of(seq, extraction_operators)
    new_polar, _ = timed("PolarExtractionOperator x8",
                         lambda: [e.build_extraction() for e in polar])
    new_bnd, _ = timed("BoundaryOperator x12",
                       lambda: [e.build_extraction() for e in boundary])
    br, bt = seq.basis_0.Λ[0], seq.basis_0.Λ[1]
    F = toroid_map(epsilon=1 / 3, R0=1.0)

    def pol(r, t):
        X = F(jnp.array([r, t, 0.0]))
        return (jnp.hypot(X[0], X[1]), X[2])
    new_ring, _ = timed("ring1_control_points",
                        lambda: extraction_operators.ring1_control_points(pol, br, bt), repeat=2)
    if old is not None:
        old_polar, old_boundary = extraction_operators_of(seq, old["extraction_operators"])
        old_p, _ = timed("[old] PolarExtractionOperator x8",
                         lambda: [e.build_extraction() for e in old_polar])
        old_b, _ = timed("[old] BoundaryOperator x12",
                         lambda: [e.build_extraction() for e in old_boundary])
        old_ring, _ = timed("[old] ring1_control_points",
                            lambda: old["extraction_operators"].ring1_control_points(pol, br, bt), repeat=2)
        for a, b in zip(new_polar, old_p):
            assert same_coo(coo(a), coo(b)), "polar extraction pattern differs"
        for a, b in zip(new_bnd, old_b):
            assert same_coo(coo(a), coo(b)), "boundary extraction pattern differs"
        check_close("ring1_control_points", np.asarray(new_ring), np.asarray(old_ring))
        print("    extraction patterns identical (rows, cols, vals)")

    print("stencil assemblers:")
    scalar, vectorial, curlcurl = assembler_inputs(seq)
    new_s, _ = timed("assemble_scalar (M0 ref)",
                     lambda: assembly.assemble_scalar(*scalar["args"]), repeat=2)
    new_v, _ = timed("assemble_vectorial (P12 block)",
                     lambda: assembly.assemble_vectorial(*vectorial["args"], **vectorial["kwargs"]), repeat=2)
    new_c, _ = timed("assemble_vectorial (curl-curl)",
                     lambda: assembly.assemble_vectorial(*curlcurl["args"]), repeat=2)
    if old is not None:
        oa = old["assembly"]
        old_s, _ = timed("[old] assemble_scalar", lambda: oa.assemble_scalar(*scalar["args"]), repeat=2)
        old_v, _ = timed("[old] assemble_vectorial (P12)",
                         lambda: oa.assemble_vectorial(*vectorial["args"], **vectorial["kwargs"]), repeat=2)
        old_c, _ = timed("[old] assemble_vectorial (curl-curl)",
                         lambda: oa.assemble_vectorial(*curlcurl["args"]), repeat=2)
        for label, a, b in (("assemble_scalar", new_s, old_s),
                            ("assemble_vectorial P12", new_v, old_v),
                            ("assemble_vectorial curl-curl", new_c, old_c)):
            assert a.shape == b.shape and a.nse == b.nse, label
            pa, pb = pattern(a), pattern(b)
            assert np.array_equal(pa[0], pb[0]) and np.array_equal(pa[1], pb[1]), \
                f"{label}: COO pattern differs"
            check_close(label, dense(a), dense(b))

    if not do_nullspace:
        return
    print("nullspace helpers (toroid spline map installed):")
    install_toroid(seq)
    ops = seq.operators
    new_null, _ = timed("compute_nullspaces_iterative",
                        lambda: nullspace.compute_nullspaces_iterative(seq, ops)[0], repeat=2)
    seq.set_operators(new_null)
    new_stiff = {}
    for k, dbc in ((1, False), (2, True)):
        new_stiff[(k, dbc)], _ = timed(
            f"get_stiffness_nullspace k={k} dbc={dbc}",
            lambda k=k, dbc=dbc: nullspace.get_stiffness_nullspace(seq, new_null, k, dbc),
            repeat=2)
    if old is not None:
        on = old["nullspace"]
        old_null, _ = timed("[old] compute_nullspaces_iterative",
                            lambda: on.compute_nullspaces_iterative(seq, ops)[0], repeat=2)
        for k in range(4):
            for dbc in (False, True):
                a = np.asarray(nullspace.get_nullspace(new_null, k, dbc))
                b = np.asarray(nullspace.get_nullspace(old_null, k, dbc))
                for va, vb in zip(a, b):
                    s = np.sign(va @ vb)
                    err = np.max(np.abs(va - s * vb)) / np.max(np.abs(vb))
                    print(f"    harmonic k={k} dbc={dbc}: max rel diff {err:.2e}")
                    assert err < 1e-6, "harmonic vector differs"
        for k, dbc in ((1, False), (2, True)):
            old_b, _ = timed(f"[old] get_stiffness_nullspace k={k} dbc={dbc}",
                             lambda k=k, dbc=dbc: on.get_stiffness_nullspace(seq, old_null, k, dbc),
                             repeat=2)
            new_b = new_stiff[(k, dbc)]
            assert new_b.shape == old_b.shape, (new_b.shape, old_b.shape)
            # subspace agreement: every old vector is reproduced by the new basis
            M_old = jax.vmap(lambda v, k=k, dbc=dbc: seq.apply_mass_matrix(
                v, k, dirichlet=dbc, operators=new_null))(old_b)
            proj = (new_b @ M_old.T).T @ new_b          # (n_old, n)
            err = float(jnp.max(jnp.abs(proj - old_b)) / jnp.max(jnp.abs(old_b)))
            print(f"    stiffness nullspace k={k} dbc={dbc}: {new_b.shape[0]} vectors, "
                  f"subspace max rel diff {err:.2e}")
            assert err < 1e-8, "stiffness nullspace subspace differs"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--old-root", default=None,
                    help="reference checkout whose mrx/{assembly,extraction_operators,"
                         "nullspace}.py are timed and compared against")
    ap.add_argument("--n", type=int, nargs="*", default=[8, 16])
    args = ap.parse_args()
    print("mrx from:", mrx.__file__)
    old = None
    if args.old_root:
        old = {name: load_old(args.old_root, name)
               for name in ("assembly", "extraction_operators", "nullspace")}
        print("reference from:", args.old_root)
    for n in args.n:
        ns = (8, 16, 8) if n == 8 else (n, n, n)
        run_resolution(ns, old, do_nullspace=(n == 8))
    print("\nOK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
