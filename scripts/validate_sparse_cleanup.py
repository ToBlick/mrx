"""Short smoke test for the jax.sparse cleanup:
  - preconditioners.py: jsparse removed, _restrict_sparse_rows/cols are one-liners
  - operators.py: e* fields annotated MatrixFreeExtraction
  - derham_sequence.py: reference_m0 stored as MatrixFreeExtraction
  - utils.py: square_sparse deleted

Run (GPU recommended, CPU tolerable):
    JAX_ENABLE_X64=1 python scripts/validate_sparse_cleanup.py
"""
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from mrx.derham_sequence import DeRhamSequence
from mrx.extraction_operators import MatrixFreeExtraction
from mrx.mappings import toroid_map

print(f"JAX devices: {jax.devices()}")

# ── 1. Build a small-ish sequence ─────────────────────────────────────────────
ns, ps, q = (8, 16, 8), (2, 2, 2), 4
seq = DeRhamSequence(ns, ps, q, ("clamped", "periodic", "periodic"), polar=True)
seq.set_map(toroid_map(epsilon=1/3))
seq.evaluate_1d()
print("Sequence built.")

# ── 2. All extraction fields must be MatrixFreeExtraction ─────────────────────
for k in range(4):
    for tag in ("", "_dbc", "_bc"):
        e = getattr(seq, f"e{k}{tag}")
        assert isinstance(e, MatrixFreeExtraction), \
            f"e{k}{tag} is {type(e).__name__}, expected MatrixFreeExtraction"
print("2. All e* fields are MatrixFreeExtraction: OK")

# ── 3. reference_m0 is MatrixFreeExtraction ───────────────────────────────────
seq.assemble_reference_mass_matrix()
assert isinstance(seq.reference_m0, MatrixFreeExtraction), \
    f"reference_m0 is {type(seq.reference_m0).__name__}"
print("3. reference_m0 is MatrixFreeExtraction: OK")

# ── 4. reference mass matrix apply round-trips ────────────────────────────────
key = jax.random.PRNGKey(42)
v_dbc = jax.random.normal(key, (seq.n0_dbc,), dtype=jnp.float64)
v_full = jax.random.normal(key, (seq.n0,), dtype=jnp.float64)
out_dbc = seq._apply_reference_mass_matrix(v_dbc, dirichlet=True)
out_full = seq._apply_reference_mass_matrix(v_full, dirichlet=False)
assert out_dbc.shape == (seq.n0_dbc,), f"shape mismatch: {out_dbc.shape}"
assert out_full.shape == (seq.n0,), f"shape mismatch: {out_full.shape}"
print(f"4. reference mass apply shapes: DBC={out_dbc.shape} full={out_full.shape}: OK")

# ── 5. reference mass preconditioner (Jacobi) ─────────────────────────────────
prec_dbc = seq._apply_reference_mass_matrix_preconditioner(v_dbc, dirichlet=True)
prec_full = seq._apply_reference_mass_matrix_preconditioner(v_full, dirichlet=False)
assert prec_dbc.shape == (seq.n0_dbc,)
assert prec_full.shape == (seq.n0,)
print(f"5. reference mass preconditioner: OK")

# ── 6. _restrict_sparse_rows / _restrict_sparse_cols (from preconditioners) ───
from mrx.preconditioners import _restrict_sparse_rows, _restrict_sparse_cols
import numpy as np
rng = np.random.default_rng(0)
row_idx = jnp.asarray(rng.choice(seq.n0, size=seq.n0 // 2, replace=False).astype(np.int32))
col_idx = jnp.asarray(rng.choice(seq.e0.shape[1], size=seq.e0.shape[1] // 2, replace=False).astype(np.int32))

Er = _restrict_sparse_rows(seq.e0, row_idx)
assert isinstance(Er, MatrixFreeExtraction)
assert Er.shape[0] == len(row_idx)

Ec = _restrict_sparse_cols(seq.e0, col_idx)
assert isinstance(Ec, MatrixFreeExtraction)
assert Ec.shape[1] == len(col_idx)
print("6. _restrict_sparse_rows/_cols: OK")

# ── 7. SequenceOperators e* annotations are MatrixFreeExtraction ──────────────
from mrx.operators import SequenceOperators, _ensure_extraction_operators
ops = _ensure_extraction_operators(seq, None)
for k in range(4):
    for tag in ("", "_T", "_dbc", "_dbc_T", "_bc", "_bc_T"):
        e = getattr(ops, f"e{k}{tag}")
        assert isinstance(e, MatrixFreeExtraction), \
            f"ops.e{k}{tag} is {type(e).__name__}"
print("7. SequenceOperators e* are MatrixFreeExtraction: OK")

print("\nAll checks passed.")
