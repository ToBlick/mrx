"""Benchmark deflated stiffness preconditioners for k=0/k=1/k=2 cases.

Builds the exact stiffness operators once, removes the image kernels through
``solve_singular_cg(..., vs=get_stiffness_nullspace(...))``, and compares the
production tensor stiffness preconditioner against Jacobi and Chebyshev.

The default cases match the harmonic-free standalone stiffness configurations:

- ``k0_dbc``  : grad-grad with Dirichlet data
- ``k1_dbc``  : curl-curl with Dirichlet data, deflating ``im(grad)``
- ``k2_free`` : div-div with free data, deflating ``im(curl)``
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp

import jax.experimental.sparse as jsparse

from mrx.assembly import assemble_vectorial, grad_1d
from mrx.derham_sequence import DeRhamSequence
from mrx.mappings import rotating_ellipse_map
from mrx.nullspace import get_nullspace, get_stiffness_nullspace
from mrx.operators import (
    _build_chebyshev_apply_preconditioner,
    _estimate_chebyshev_lanczos_bounds_apply,
    _mass_extraction,
    apply_laplacian_preconditioner,
    apply_mass_matrix,
    apply_stiffness,
    apply_stiffness_tensor_preconditioner,
    assemble_laplacian_operators,
    assemble_incidence_operators,
    assemble_mass_operators,
    assemble_tensor_mass_preconditioner,
    assemble_tensor_stiffness_preconditioner,
)
from mrx.preconditioners import MassPreconditionerSpec
from mrx.solvers import solve_singular_cg
from mrx.io import parse_int_list, parse_ns
from mrx.operators import diag_EAET


jax.config.update("jax_enable_x64", True)


TYPES = ("clamped", "periodic", "periodic")
BETTI = (1, 1, 0, 0)


@dataclass(frozen=True)
class Case:
    k: int
    dirichlet: bool

    @property
    def label(self) -> str:
        return f"k{self.k}_{'dbc' if self.dirichlet else 'free'}"

    @property
    def bc(self) -> str:
        return "dbc" if self.dirichlet else "free"

    @property
    def dof_attr(self) -> str:
        return f"n{self.k}{'_dbc' if self.dirichlet else ''}"


@dataclass(frozen=True)
class Row:
    case: str
    k: int
    bc: str
    strategy: str
    avg_iters: float
    max_iters: int
    avg_ms: float
    failures: int
    avg_residual: float = float('nan')
    rank: int = -1


def _parse_cases(text: str) -> tuple[Case, ...]:
    cases: list[Case] = []
    for raw in text.split(","):
        token = raw.strip().lower()
        if not token:
            continue
        if token == "k0_dbc":
            cases.append(Case(k=0, dirichlet=True))
        elif token == "k0_free":
            cases.append(Case(k=0, dirichlet=False))
        elif token == "k1_dbc":
            cases.append(Case(k=1, dirichlet=True))
        elif token == "k1_free":
            cases.append(Case(k=1, dirichlet=False))
        elif token == "k2_dbc":
            cases.append(Case(k=2, dirichlet=True))
        elif token == "k2_free":
            cases.append(Case(k=2, dirichlet=False))
        else:
            raise ValueError(
                f"Unknown case {raw!r}; expected comma-separated subset of "
                "k0_dbc,k0_free,k1_dbc,k1_free,k2_dbc,k2_free"
            )
    if not cases:
        raise ValueError("At least one stiffness benchmark case is required")
    return tuple(cases)


def build_sequence(args) -> DeRhamSequence:
    seq = DeRhamSequence(
        args.ns,
        (args.p, args.p, args.p),
        2 * args.p,
        TYPES,
        polar=True,
        tol=args.tol,
        maxiter=args.maxiter,
        betti_numbers=BETTI,
    )
    seq.evaluate_1d()
    seq.assemble_reference_mass_matrix()
    seq.set_map(
        rotating_ellipse_map(
            eps=args.rotating_eps,
            kappa=args.rotating_kappa,
            R0=args.rotating_r0,
            nfp=args.rotating_nfp,
        )
    )
    return seq


def build_base_operators(seq: DeRhamSequence, cases: tuple[Case, ...]):
    mass_ks = tuple(
        sorted(
            {
                degree
                for case in cases
                for degree in (case.k - 1, case.k, case.k + 1)
                if 0 <= degree <= 3
            }
        )
    )
    incidence_ks = tuple(
        sorted(
            {
                degree
                for case in cases
                for degree in (case.k - 1, case.k)
                if 0 <= degree <= 2
            }
        )
    )

    operators = assemble_mass_operators(seq, seq.geometry, ks=mass_ks)
    operators = assemble_incidence_operators(seq, operators=operators, ks=incidence_ks)
    if any(case.k == 0 for case in cases):
        operators = assemble_laplacian_operators(seq, seq.geometry, operators=operators, ks=(0,))
    seq.set_operators(operators)
    return operators


def build_sparse_stiffness_operator(seq: DeRhamSequence, case: Case):
    quad_shape = (seq.quad.ny, seq.quad.nx, seq.quad.nz)
    if case.k == 0:
        return None
    if case.k == 1:
        weight = (1.0 / seq.geometry.jacobian_j) * seq.quad.w
        w_3x3 = seq.geometry.metric_jkl * weight[:, None, None]
        d_r = seq.d_basis_r_jk
        d_t = seq.d_basis_t_jk
        d_z = seq.d_basis_z_jk
        r = seq.basis_r_jk
        t = seq.basis_t_jk
        z = seq.basis_z_jk
        curl_terms = [
            [(1, d_r, t, grad_1d(d_z, TYPES[2]), +1),
             (2, d_r, grad_1d(d_t, TYPES[1]), z, -1)],
            [(0, r, d_t, grad_1d(d_z, TYPES[2]), -1),
             (2, grad_1d(d_r, TYPES[0]), d_t, z, +1)],
            [(0, r, grad_1d(d_t, TYPES[1]), d_z, +1),
             (1, grad_1d(d_r, TYPES[0]), t, d_z, -1)],
        ]
        sp = assemble_vectorial(
            curl_terms,
            curl_terms,
            w_3x3,
            quad_shape,
            list(seq.basis_1.shape),
            seq.basis_1.pr,
        )
        return jsparse.BCSR.from_bcoo(sp)
    if case.k == 2:
        grad_r = grad_1d(seq.d_basis_r_jk, TYPES[0])
        grad_t = grad_1d(seq.d_basis_t_jk, TYPES[1])
        grad_z = grad_1d(seq.d_basis_z_jk, TYPES[2])
        w_scalar = (1.0 / seq.geometry.jacobian_j) * seq.quad.w
        w_3x3 = w_scalar[:, None, None] * jnp.ones((1, 3, 3), dtype=jnp.float64)
        div_terms = [
            [(0, grad_r, seq.d_basis_t_jk, seq.d_basis_z_jk, +1)],
            [(1, seq.d_basis_r_jk, grad_t, seq.d_basis_z_jk, +1)],
            [(2, seq.d_basis_r_jk, seq.d_basis_t_jk, grad_z, +1)],
        ]
        sp = assemble_vectorial(
            div_terms,
            div_terms,
            w_3x3,
            quad_shape,
            list(seq.basis_2.shape),
            seq.basis_2.pr,
        )
        return jsparse.BCSR.from_bcoo(sp)
    raise ValueError(f"Sparse stiffness assembly only supports k=0, k=1 or k=2 (got {case.k})")


def apply_sparse_stiffness_operator(seq: DeRhamSequence, operators, stiffness_sp, x, case: Case):
    if case.k == 0:
        return apply_stiffness(seq, operators, x, 0, dirichlet=case.dirichlet)
    e, e_T = _mass_extraction(operators, case.k, case.dirichlet)
    return e @ (stiffness_sp @ (e_T @ x))


def build_stiffness_jacobi_diaginv(operators, stiffness_sp, case: Case):
    if case.k == 0:
        return operators.dd0_diaginv_dbc if case.dirichlet else operators.dd0_diaginv
    e, e_T = _mass_extraction(operators, case.k, case.dirichlet)
    diagonal = diag_EAET(e, stiffness_sp, e_T)
    safe_diagonal = jnp.where(diagonal > 0.0, diagonal, 1.0)
    return jnp.where(diagonal > 0.0, 1.0 / safe_diagonal, 0.0)


def build_rhs_batch(seq: DeRhamSequence, operators, stiffness_sp, case: Case, *, n_rhs: int, seed: int):
    size = int(getattr(seq, case.dof_attr))
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, n_rhs)

    def K(x):
        return apply_sparse_stiffness_operator(seq, operators, stiffness_sp, x, case)

    return tuple(
        K(jax.random.normal(k, (size,), dtype=jnp.float64))
        for k in keys
    )


def time_solve(solve, rhs_batch) -> tuple[float, int, float, int, float]:
    x, info, res = solve(rhs_batch[0])
    jax.block_until_ready((x, info, res))

    iters: list[int] = []
    times_ms: list[float] = []
    residuals: list[float] = []
    failures = 0
    for rhs in rhs_batch:
        t0 = time.perf_counter()
        x, info, res = solve(rhs)
        jax.block_until_ready((x, info, res))
        info_int = int(info)
        failures += int(info_int >= 0)
        times_ms.append((time.perf_counter() - t0) * 1e3)
        iters.append(abs(info_int))
        residuals.append(float(res))
    return (
        float(jnp.mean(jnp.asarray(iters))),
        int(max(iters)),
        float(jnp.mean(jnp.asarray(times_ms))),
        failures,
        float(jnp.mean(jnp.asarray(residuals))),
    )


def benchmark_case(
    seq: DeRhamSequence,
    operators,
    rhs_batch,
    args,
    *,
    case: Case,
    strategy: str,
    jacobi_diaginv,
    stiffness_sp,
) -> Row:
    if case.k == 0:
        nullspace = get_nullspace(operators, case.k, case.dirichlet)
    else:
        nullspace = get_stiffness_nullspace(seq, operators, case.k, case.dirichlet)

    def operator_apply(x):
        return apply_sparse_stiffness_operator(seq, operators, stiffness_sp, x, case)

    def mass_apply(x):
        return apply_mass_matrix(seq, operators, x, case.k, dirichlet=case.dirichlet)

    if strategy == "tensor":
        @jax.jit
        def solve(rhs, vs):
            x, info = solve_singular_cg(
                operator_apply,
                rhs,
                mass_matvec=mass_apply,
                precond_matvec=(
                    (lambda x: apply_laplacian_preconditioner(
                        seq, operators, x, 0, dirichlet=case.dirichlet, kind="tensor",
                    )) if case.k == 0 else
                    (lambda x: apply_stiffness_tensor_preconditioner(
                        seq, operators, x, case.k, dirichlet=case.dirichlet,
                    ))
                ),
                vs=vs,
                tol=args.tol,
                maxiter=args.maxiter,
            )
            r = operator_apply(x) - rhs
            r_M = seq.l2_norm(r, case.k, dirichlet=case.dirichlet)
            b_M = seq.l2_norm(rhs, case.k, dirichlet=case.dirichlet)
            res = r_M / jnp.where(b_M > 0.0, b_M, 1.0)
            return x, info, res
    elif strategy == "jacobi":
        @jax.jit
        def solve(rhs, vs, diaginv):
            x, info = solve_singular_cg(
                operator_apply,
                rhs,
                mass_matvec=mass_apply,
                precond_matvec=lambda x: diaginv * x,
                vs=vs,
                tol=args.tol,
                maxiter=args.maxiter,
            )
            r = operator_apply(x) - rhs
            r_M = seq.l2_norm(r, case.k, dirichlet=case.dirichlet)
            b_M = seq.l2_norm(rhs, case.k, dirichlet=case.dirichlet)
            res = r_M / jnp.where(b_M > 0.0, b_M, 1.0)
            return x, info, res
    elif strategy.startswith("cheb"):
        jacobi_apply = lambda rhs, diaginv: diaginv * rhs
        spec = MassPreconditionerSpec(
            kind="chebyshev",
            steps=args.cheb_steps,
            power_iterations=8,
            min_eig_fraction=1e-3,
            smoother=MassPreconditionerSpec(kind="jacobi"),
        )
        min_eig, max_eig = _estimate_chebyshev_lanczos_bounds_apply(
            operator_apply,
            lambda rhs: jacobi_apply(rhs, jacobi_diaginv),
            int(getattr(seq, case.dof_attr)),
            spec=spec,
            seed=args.seed + 100 * case.k + int(case.dirichlet),
            orthogonal_vectors=nullspace,
        )
        chebyshev_apply = _build_chebyshev_apply_preconditioner(
            operator_apply,
            lambda rhs: jacobi_apply(rhs, jacobi_diaginv),
            steps=args.cheb_steps,
            min_eig=min_eig,
            max_eig=max_eig,
        )
        @jax.jit
        def solve(rhs, vs):
            x, info = solve_singular_cg(
                operator_apply,
                rhs,
                mass_matvec=mass_apply,
                precond_matvec=chebyshev_apply,
                vs=vs,
                tol=args.tol,
                maxiter=args.maxiter,
            )
            r = operator_apply(x) - rhs
            r_M = seq.l2_norm(r, case.k, dirichlet=case.dirichlet)
            b_M = seq.l2_norm(rhs, case.k, dirichlet=case.dirichlet)
            res = r_M / jnp.where(b_M > 0.0, b_M, 1.0)
            return x, info, res
    else:
        raise ValueError(f"Unknown strategy {strategy!r}")

    if strategy == "jacobi":
        solve_wrapped = lambda rhs: solve(rhs, nullspace, jacobi_diaginv)
    else:
        solve_wrapped = lambda rhs: solve(rhs, nullspace)

    avg_it, max_it, avg_ms, failures, avg_res = time_solve(solve_wrapped, rhs_batch)
    return Row(
        case=case.label,
        k=case.k,
        bc=case.bc,
        strategy=strategy,
        avg_iters=avg_it,
        max_iters=max_it,
        avg_ms=avg_ms,
        failures=failures,
        avg_residual=avg_res,
        rank=getattr(args, "_active_rank", -1) if strategy == "tensor" else -1,
    )


def print_table(rows: list[Row]) -> None:
    header = (
        f"{'case':>9} {'strategy':>14} {'avg_it':>7} {'max_it':>7} {'avg_ms':>9} {'fails':>6} {'avg_resM':>10}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        label = (
            f"tensor(r={row.rank})" if row.strategy == "tensor" and row.rank >= 0
            else row.strategy
        )
        print(
            f"{row.case:>9} {label:>14} {row.avg_iters:>7.1f} {row.max_iters:>7d}"
            f" {row.avg_ms:>9.2f} {row.failures:>6d} {row.avg_residual:>10.2e}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ns", type=parse_ns, default=(12, 16, 8))
    parser.add_argument("--p", type=int, default=3)
    parser.add_argument("--cases", type=_parse_cases, default=(Case(1, True), Case(2, False)))
    parser.add_argument(
        "--strategies",
        type=str,
        default="tensor,jacobi,chebyshev",
        help="Comma-separated subset of tensor,jacobi,chebyshev.",
    )
    parser.add_argument("--cheb-steps", type=int, default=4)
    parser.add_argument("--rank", type=int, default=None,
                        help="Single rank shortcut for --ranks (deprecated; use --ranks).")
    parser.add_argument(
        "--ranks",
        type=parse_int_list,
        default=(1,),
        help="Comma-separated tensor ranks to compare (>=1).",
    )
    parser.add_argument("--n-rhs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tol", type=float, default=1e-9)
    parser.add_argument("--maxiter", type=int, default=1000)
    parser.add_argument("--rotating-eps", type=float, default=1.0 / 7.0)
    parser.add_argument("--rotating-kappa", type=float, default=1.5)
    parser.add_argument("--rotating-r0", type=float, default=3.0)
    parser.add_argument("--rotating-nfp", type=int, default=3)
    parser.add_argument(
        "--no-inner-schur",
        dest="inner_schur",
        action="store_false",
        help="Disable the inner RT/zeta Schur in the k=1/k=2 bulk apply.",
    )
    parser.set_defaults(inner_schur=False)
    args = parser.parse_args()

    strategies = tuple(s.strip() for s in args.strategies.split(",") if s.strip())
    for strategy in strategies:
        if strategy not in ("tensor", "jacobi", "chebyshev"):
            raise ValueError(f"Unknown strategy {strategy!r}")
    if args.rank is not None:
        ranks = (args.rank,)
    else:
        ranks = args.ranks
    invalid_ranks = tuple(r for r in ranks if r < 1)
    if invalid_ranks:
        raise ValueError(f"--ranks must be >= 1; got {invalid_ranks}")

    print(f"Building sequence ns={args.ns} p={args.p} ...", flush=True)
    seq = build_sequence(args)
    base_operators = build_base_operators(seq, args.cases)
    seq.set_operators(base_operators)

    for case in args.cases:
        harmonic_count = int(get_nullspace(base_operators, case.k, case.dirichlet).shape[0])
        if case.k in (1, 2) and harmonic_count != 0:
            raise ValueError(
                f"Case {case.label} is not harmonic-free; got {harmonic_count} harmonic vectors"
            )

    stiffness_ops = {
        case.label: build_sparse_stiffness_operator(seq, case)
        for case in args.cases
    }
    rhs_batches = {
        case.label: build_rhs_batch(
            seq,
            base_operators,
            stiffness_ops[case.label],
            case,
            n_rhs=args.n_rhs,
            seed=args.seed + 1000 * case.k + 17 * int(case.dirichlet),
        )
        for case in args.cases
    }
    jacobi_diaginv = {
        case.label: build_stiffness_jacobi_diaginv(base_operators, stiffness_ops[case.label], case)
        for case in args.cases
    }

    rows: list[Row] = []
    non_tensor_strategies = tuple(s for s in strategies if s != "tensor")
    if non_tensor_strategies:
        for case in args.cases:
            rhs_batch = rhs_batches[case.label]
            for strategy in non_tensor_strategies:
                rows.append(
                    benchmark_case(
                        seq,
                        base_operators,
                        rhs_batch,
                        args,
                        case=case,
                        strategy=strategy,
                        jacobi_diaginv=jacobi_diaginv[case.label],
                        stiffness_sp=stiffness_ops[case.label],
                    )
                )

    if "tensor" in strategies:
        ks = tuple(sorted({case.k for case in args.cases}))
        for rank in ranks:
            print(f"Assembling tensor stiffness preconditioner rank={rank} ...", flush=True)
            operators = base_operators
            if 0 in ks:
                operators = assemble_tensor_mass_preconditioner(
                    seq,
                    operators=operators,
                    ks=(0,),
                    rank=rank,
                    cp_kwargs={"k0_rank": rank},
                )
                operators = assemble_laplacian_operators(
                    seq,
                    seq.geometry,
                    operators=operators,
                    ks=(0,),
                )
            if any(k in (1, 2) for k in ks):
                operators = assemble_tensor_stiffness_preconditioner(
                    seq,
                    operators=operators,
                    ks=tuple(k for k in ks if k in (1, 2)),
                    rank=rank,
                    cp_kwargs={
                        "k1_inner_schur": args.inner_schur,
                        "k2_inner_schur": args.inner_schur,
                    },
                )
            args._active_rank = rank
            for case in args.cases:
                rhs_batch = rhs_batches[case.label]
                rows.append(
                    benchmark_case(
                        seq,
                        operators,
                        rhs_batch,
                        args,
                        case=case,
                        strategy="tensor",
                        jacobi_diaginv=jacobi_diaginv[case.label],
                        stiffness_sp=stiffness_ops[case.label],
                    )
                )

    def _sort_key(row: Row):
        order = {"jacobi": 0, "chebyshev": 1, "tensor": 2}.get(row.strategy, 99)
        return (row.k, row.bc, order, row.rank)

    rows.sort(key=_sort_key)

    print()
    print_table(rows)


if __name__ == "__main__":
    main()