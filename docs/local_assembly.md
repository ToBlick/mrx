# Element-local mass assembly

This note describes `mrx/local_assembly.py`: how the mass matrices `M0`, `M1`,
`M2`, `M3` are built, why it is fast, and how its memory footprint is bounded.

## Why a separate assembler

The reference path in `mrx/assembly.py` forms each mass matrix with a global
`O(n^6)` einsum over all degrees of freedom. That is simple but both slow and
memory-hungry as the mesh `n` grows.

`local_assembly.py` instead exploits the two structural properties of a
tensor-product spline space:

1. **Separability.** Each 3D basis function is a product of three 1D bases, so a
   3D integral factorizes into a sequence of 1D contractions
   (*sum factorization*).
2. **Local support.** A degree-`p` spline touches only `p+1` neighbours per
   axis, so each element couples only `(p+1)^3` row DOFs to `(p+1)^3` column
   DOFs.

Forming the contribution element by element and scattering it into a sparse
triplet list gives `O(n^3 p^6)` work that is **flat in `n`** on the GPU. The
result is the raw tensor-product mass matrix in the periodic/unextracted DOF
space, identical (to machine precision) to the global assembler. Polar/boundary
extraction `E M E^T` is applied afterwards, exactly as in the global path.

## Form weights

Each form degree uses a different geometry weight `W` at the quadrature points
(matching `mrx.operators._assemble_mass_block`). The quadrature weights `w` are
folded in per axis via the 1D Gauss weights.

| form | weight `W`        | basis on axis `c`        | rank      |
| ---- | ----------------- | ------------------------ | --------- |
| k=0  | `J`               | primal (all axes)        | scalar    |
| k=1  | `G^{-1} J`        | derivative on axis `c`   | 3×3 (vec) |
| k=2  | `G (1/J)`         | primal on axis `c`       | 3×3 (vec) |
| k=3  | `1/J`             | derivative (all axes)    | scalar    |

Here `J = det DF` is the mapping Jacobian determinant and `G` is the metric.
The 1-form and 2-form weights are full 3×3 metric tensors, so their assembly
loops over all nine component pairs `(cr, cc)`.

## Pipeline

```
assemble_mass_local(seq, k)            # dispatch by form degree
  -> assemble_m{0,1,2,3}_local(seq)    # pick bases + weight W
    -> _assemble_scalar_local          # k=0, k=3
       _assemble_vectorial_local       # k=1, k=2  (loops cr, cc in 0..2)
         -> evaluate_basis_local       # 1D basis values + global DOF ids
         -> _block_compute             # element-block contraction
         -> scatter into (vals, rows, cols) -> BCOO
```

* `evaluate_basis_local(basis, x_q, q)` returns, for every element, the
  `(p+1)` locally-active 1D basis values at the local Gauss points plus the
  global DOF index of each local basis. It handles both primal bases (`p+1`
  locals) and derivative bases (`p` locals).
* `_assemble_scalar_local` / `_assemble_vectorial_local` build the per-element
  dense blocks, compute the global row/column flat indices from the DOF ids,
  and concatenate everything into a single `BCOO`.

## Sum factorization of one element block

`_elem_block_mixed` contracts a single element block as three sequential 1D
einsums rather than one 6D einsum:

```python
W  = Wf * wx_e[:, None, None] * wy_e[None, :, None] * wz_e[None, None, :]
A  = einsum('qa,qb,qrs->abrs', Bxr, Bxc, W)        # contract x quad
Bm = einsum('rc,rd,abrs->abcds', Byr, Byc, A)      # contract y quad
C  = einsum('se,sf,abcds->abcdef', Bzr, Bzc, Bm)   # contract z quad
```

The result `C[a,b,c,d,e,f]` holds the `(p+1)^2` per-axis row/column couplings
for one element. `Bxr/Bxc` etc. are the row/column 1D basis values; for the
vectorial forms the row and column bases differ per component.

## Memory: where it goes, and the real limit

The element grid has three axes `(ex, ey, ez)`. `_block_compute` vectorizes all
three with nested `vmap`, which materializes the dense element-block array

```
shape (ne_x, ne_y, ne_z, (p+1)^6)   ~  O(n^3 p^6)
```

This array is not a reducible scratch buffer: its values **are** the assembled
matrix nonzeros, scattered straight into the `BCOO` `data`. So its size equals
`m{k}_nnz_stored`. At `n=32, p=4` the k=0 block is exactly
`28 * 64 * 32 * 5^6 = 8.96e8` float64 = **6.67 GiB**.

A `jax.lax.map` over the outer `ex` axis (inner `ey`/`ez` vectorized) lowers to
`scan` and caps the *intermediate* contraction to one `ex`-slice
(`O(n^2 p^6)`), but `scan` still stacks the per-slice results back into the same
full `(ne_x, ne_y, ne_z, (p+1)^6)` output. Because that output is the matrix
data, the outer/inner split does **not** lower the dominant allocation here, so
the assembler keeps the cleaner triple-`vmap`.

The honest bottleneck is the stored matrix size itself. With `BCOO` holding
`data` (float64, 8 B) plus `indices` (`(nnz, 2)` int32, 8 B), each stored
nonzero costs **16 bytes**:

| n (p=4) | `m1_nnz_stored` | M1 size      |
| ------- | --------------- | ------------ |
| 8       | 4.6e7           | ~0.74 GB     |
| 16      | 5.5e8           | ~8.85 GB     |
| 32      | ~5.2e9          | ~83 GB       |
| 64      | ~4.6e10         | ~740 GB      |

So the largest `(n, p)` corners (e.g. `n=32, 64` at `p=4`) exceed a single
80 GB H100 and are expected to OOM regardless of allocator settings; those
cases are simply accepted as out of reach on one device.

**Rule of thumb (still worth keeping in mind).** Nested `vmap`s over every axis
materialize the full batched output. When the large array is a genuine
*scratch* (not the result), prefer `jax.lax.map` with an outer (sequential) and
inner (vectorized) split so the largest live array stays of order `n^3`. Here
the array is the result itself, so the split does not help.

## Storage note

The stored `BCOO` for `M1`/`M2` is sparse, but ~3× denser than a strictly
orthonormal mapping would give, because the full 3×3 metric coupling is kept (no
orthonormality is assumed of the map). The stored nnz is a few times the
numerically nonzero count (`m1_nnz_stored` vs `m1_nnz_actual`); this is a
deliberate modeling choice and is left as-is.

## TODO: local polar interpolation and histopolation

`mrx/projectors.py` currently raises `NotImplementedError` for polar extraction
operators (`PolarExtractionOperator`) in `zeroform_interpolation`,
`zeroform_quasi_interpolation`, and `oneform_histopolation`. The dense-solve
implementations that existed before were removed because they formed an O(n²)
dense collocation matrix.

The local replacement for k=0 is a three-step Schur-complement solve:

1. **z-axis** (separable): for each of the `3 + (nr−2−o)·nt` distinct (r,θ)
   evaluation points, solve the banded 1D collocation system in ζ.
2. **r-θ coupling**: the full collocation matrix decomposes as
   `A = [[A_ii, A_io], [A_oi, A_oo]]` where `A_oo = C_r^{outer} ⊗ C_t` is
   solvable by two sequential 1D banded solves.  `A_ii` is 3×3 (per ζ-DOF)
   and `A_io` has O(p) nonzero radial rows (only basis functions whose support
   reaches the axis evaluation point `x_r[1]`).
3. Schur complement: solve the 3×3 system, back-substitute into the tensor
   solve for the outer DOFs.

The Greville points for polar DOFs are:
- **Inner** (p,m): `(x_r[1], x_t[p], x_z[m])` for p=0,1,2 — the same
  evaluation point in r, three distinct θ points to resolve constant/cos/sin.
- **Outer** (i,j,k): `(x_r[i+2], x_t[j], x_z[k])` — standard tensor Greville.

For k=1 histopolation the same Schur structure applies per component, using
histopolation intervals instead of collocation points in the derivative directions.

