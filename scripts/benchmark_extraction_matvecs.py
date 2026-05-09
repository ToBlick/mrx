from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from mrx.derham_sequence import DeRhamSequence


jax.config.update("jax_enable_x64", True)


TYPES = ("clamped", "periodic", "periodic")
BETTI = (1, 1, 0, 0)


@dataclass(frozen=True)
class BenchmarkRow:
    k: int
    boundary: str
    operator: str
    n_out: int
    n_in: int
    nnz: int
    selector_like: bool
    row_nnz_min: int
    row_nnz_max: int
    max_abs_error: float
    bcsr_ms: float
    index_ms: float
    speedup: float


def _parse_ns(text: str) -> tuple[int, int, int]:
    parts = tuple(int(part.strip()) for part in text.split(","))
    if len(parts) != 3:
        raise ValueError(f"Expected ns as 'nr,nt,nz', got {text!r}")
    return parts


def _parse_k_list(text: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if not values:
        raise ValueError("Expected a non-empty comma-separated list of degrees")
    invalid = tuple(k for k in values if k not in (0, 1, 2, 3))
    if invalid:
        raise ValueError(f"Degrees must lie in 0,1,2,3; got {invalid}")
    return values


def _bcsr_to_coo_indices(mat):
    lengths = mat.indptr[1:] - mat.indptr[:-1]
    rows = jnp.repeat(jnp.arange(mat.shape[0]), lengths, total_repeat_length=mat.data.shape[0])
    return rows, mat.indices


def _flatten_data(data: jnp.ndarray) -> jnp.ndarray:
    if data.ndim == 1:
        return data
    flattened = data.reshape(data.shape[0], -1)
    if flattened.shape[1] != 1:
        raise ValueError("Expected scalar BCSR entries for extraction operators")
    return flattened[:, 0]


def _selector_kernels(extraction):
    row_lengths = extraction.indptr[1:] - extraction.indptr[:-1]
    selector_like = bool(jnp.all(row_lengths == 1))
    rows, cols = _bcsr_to_coo_indices(extraction)
    weights = _flatten_data(extraction.data)
    row_nnz_min = int(jnp.min(row_lengths))
    row_nnz_max = int(jnp.max(row_lengths))

    if selector_like:
        gather_cols = cols.astype(jnp.int32)
        gather_weights = weights

        def forward(x):
            return gather_weights * x[gather_cols]

        def transpose(x):
            return jnp.zeros((extraction.shape[1],), dtype=x.dtype).at[gather_cols].add(gather_weights * x)

        return selector_like, row_nnz_min, row_nnz_max, forward, transpose

    scatter_rows = rows.astype(jnp.int32)
    scatter_cols = cols.astype(jnp.int32)
    scatter_weights = weights

    def forward(x):
        return jnp.zeros((extraction.shape[0],), dtype=x.dtype).at[scatter_rows].add(scatter_weights * x[scatter_cols])

    def transpose(x):
        return jnp.zeros((extraction.shape[1],), dtype=x.dtype).at[scatter_cols].add(scatter_weights * x[scatter_rows])

    return selector_like, row_nnz_min, row_nnz_max, forward, transpose


def _time_apply(apply, x: jnp.ndarray, repeats: int) -> float:
    compiled = jax.jit(apply)
    out = compiled(x)
    jax.block_until_ready(out)

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = compiled(x)
        jax.block_until_ready(out)
        times.append((time.perf_counter() - t0) * 1e3)
    return float(sum(times) / len(times))


def _get_extraction(seq: DeRhamSequence, k: int, dirichlet: bool):
    suffix = "_dbc" if dirichlet else ""
    return getattr(seq, f"e{k}{suffix}"), getattr(seq, f"e{k}{suffix}_T")


def _benchmark_pair(seq: DeRhamSequence, *, k: int, dirichlet: bool, repeats: int, seed: int) -> tuple[BenchmarkRow, BenchmarkRow]:
    extraction, extraction_t = _get_extraction(seq, k, dirichlet)
    selector_like, row_nnz_min, row_nnz_max, index_forward, index_transpose = _selector_kernels(extraction)
    boundary = "dbc" if dirichlet else "free"

    key = jax.random.PRNGKey(seed)
    key_forward, key_transpose = jax.random.split(key)
    x_forward = jax.random.normal(key_forward, (extraction.shape[1],), dtype=jnp.float64)
    x_transpose = jax.random.normal(key_transpose, (extraction.shape[0],), dtype=jnp.float64)

    bcsr_forward = lambda x, mat=extraction: mat @ x
    bcsr_transpose = lambda x, mat=extraction_t: mat @ x

    forward_error = float(jnp.max(jnp.abs(bcsr_forward(x_forward) - index_forward(x_forward))))
    transpose_error = float(jnp.max(jnp.abs(bcsr_transpose(x_transpose) - index_transpose(x_transpose))))

    forward_bcsr_ms = _time_apply(bcsr_forward, x_forward, repeats)
    forward_index_ms = _time_apply(index_forward, x_forward, repeats)
    transpose_bcsr_ms = _time_apply(bcsr_transpose, x_transpose, repeats)
    transpose_index_ms = _time_apply(index_transpose, x_transpose, repeats)

    nnz = int(extraction.data.shape[0])
    return (
        BenchmarkRow(
            k=k,
            boundary=boundary,
            operator="E",
            n_out=int(extraction.shape[0]),
            n_in=int(extraction.shape[1]),
            nnz=nnz,
            selector_like=selector_like,
            row_nnz_min=row_nnz_min,
            row_nnz_max=row_nnz_max,
            max_abs_error=forward_error,
            bcsr_ms=forward_bcsr_ms,
            index_ms=forward_index_ms,
            speedup=forward_bcsr_ms / forward_index_ms if forward_index_ms > 0.0 else float("inf"),
        ),
        BenchmarkRow(
            k=k,
            boundary=boundary,
            operator="E_T",
            n_out=int(extraction_t.shape[0]),
            n_in=int(extraction_t.shape[1]),
            nnz=nnz,
            selector_like=selector_like,
            row_nnz_min=row_nnz_min,
            row_nnz_max=row_nnz_max,
            max_abs_error=transpose_error,
            bcsr_ms=transpose_bcsr_ms,
            index_ms=transpose_index_ms,
            speedup=transpose_bcsr_ms / transpose_index_ms if transpose_index_ms > 0.0 else float("inf"),
        ),
    )


def _print_rows(rows: list[BenchmarkRow]) -> None:
    header = (
        f"{'k':>2} {'bc':>5} {'op':>4} {'n_out':>8} {'n_in':>8} {'nnz':>8} "
        f"{'row_min':>7} {'row_max':>7} {'selector':>9} {'err_max':>12} {'bcsr_ms':>10} {'index_ms':>10} {'speedup':>8}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row.k:2d} {row.boundary:>5} {row.operator:>4} {row.n_out:8d} {row.n_in:8d} {row.nnz:8d} "
            f"{row.row_nnz_min:7d} {row.row_nnz_max:7d} {str(row.selector_like):>9} "
            f"{row.max_abs_error:12.3e} {row.bcsr_ms:10.3f} {row.index_ms:10.3f} {row.speedup:8.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Microbenchmark BCSR extraction matvecs against direct gather/scatter-style indexing."
    )
    parser.add_argument("--ns", type=_parse_ns, default=(16, 32, 16), help="Grid resolution as nr,nt,nz")
    parser.add_argument("--p", type=int, default=3, help="Spline degree in each dimension")
    parser.add_argument("--ks", type=_parse_k_list, default=(0, 1, 2, 3), help="Comma-separated list of degrees to benchmark")
    parser.add_argument("--boundary", choices=("free", "dbc", "both"), default="both", help="Which extraction operators to benchmark")
    parser.add_argument("--repeats", type=int, default=200, help="Timed repetitions after one warmup call")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for benchmark vectors")
    args = parser.parse_args()

    seq = DeRhamSequence(
        args.ns,
        (args.p, args.p, args.p),
        2 * args.p,
        TYPES,
        polar=True,
        betti_numbers=BETTI,
    )

    boundaries = (False, True) if args.boundary == "both" else ((args.boundary == "dbc"),)
    rows: list[BenchmarkRow] = []
    seed = args.seed
    for k in args.ks:
        for dirichlet in boundaries:
            forward_row, transpose_row = _benchmark_pair(
                seq,
                k=k,
                dirichlet=dirichlet,
                repeats=args.repeats,
                seed=seed,
            )
            rows.extend([forward_row, transpose_row])
            seed += 1

    print(
        f"ns={args.ns}, p={args.p}, ks={args.ks}, boundary={args.boundary}, repeats={args.repeats}"
    )
    _print_rows(rows)


if __name__ == "__main__":
    main()