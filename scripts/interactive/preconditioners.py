# %%
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax.scipy.sparse.linalg import cg

import mrx
from mrx.derham_sequence import DeRhamSequence
from mrx.differential_forms import DiscreteFunction
from mrx.mappings import toroid_map

# Enable 64-bit precision for numerical stability
jax.config.update("jax_enable_x64", True)

types = ("clamped", "periodic", "periodic")  # Types
a = 1 / 3  # minor radius
π = jnp.pi
F = toroid_map(epsilon=a)

n = 16
p = 2

ns = (n, n, n)
ps = (p, p, p)
q = 2*p

    # Create DeRham sequence
seq = DeRhamSequence(ns, ps, q, types, F, polar=True,
                         dirichlet=True, tol=1e-9, maxiter=1000)
seq.evaluate_1d()
seq.assemble_all_sparse()
# %%

def get_schur_vector(M, blocks):
    """Build a block-Schur-complement preconditioner for a vector-valued form.

    The matrix M has dense blocks on the diagonal at specified positions.
    For each dense block, the Schur complement is formed against the rest.
    Between the dense blocks, diagonal (Jacobi) inversion is used.

    Parameters
    ----------
    M : array, shape (N, N)
        The full matrix.
    blocks : sequence of (position, length) tuples
        Each ``(pos, length)`` pair identifies a dense block on the diagonal
        of M.  Blocks must be non-overlapping and sorted by position.
        Regions of the diagonal not covered by any block are treated as
        diagonal (Jacobi) segments.

    Returns
    -------
    P : array, shape (N, N)
        The preconditioner matrix (to be applied as ``P @ M``).
    """
    N = M.shape[0]

    # ---- Build the list of segments: ("schur", pos, length) or ("diag", pos, length)
    segments = []
    cursor = 0
    for pos, length in blocks:
        if pos > cursor:
            segments.append(("diag", cursor, pos - cursor))
        if length > 0:
            segments.append(("schur", pos, length))
        cursor = pos + length
    if cursor < N:
        segments.append(("diag", cursor, N - cursor))

    # ---- For each segment, compute its preconditioner block
    # P is block-diagonal: for Schur blocks we compute the full Schur complement,
    # for diagonal blocks we invert the diagonal.
    P = jnp.zeros_like(M)
    for kind, pos, length in segments:
        if kind == "diag":
            d = jnp.diag(M)[pos:pos + length]
            P = P.at[pos:pos + length, pos:pos + length].set(jnp.diag(1.0 / d))
        else:  # "schur"
            # The block A is M[pos:pos+length, pos:pos+length].
            # The off-diagonal coupling C is M[pos:pos+length, complement].
            # D is the diagonal of M restricted to the complement.
            A = M[pos:pos + length, pos:pos + length]

            # Gather complement indices (everything not in this block)
            comp_mask = jnp.ones(N, dtype=bool).at[pos:pos + length].set(False)
            comp = jnp.where(comp_mask, size=N - length)[0]

            C = M[pos:pos + length][:, comp]
            d_comp = jnp.diag(M)[comp]
            D_inv_diag = 1.0 / d_comp

            # Schur complement: S = A - C @ diag(D_inv) @ C^T
            S = A - C * D_inv_diag[None, :] @ C.T
            P = P.at[pos:pos + length, pos:pos + length].set(jnp.linalg.inv(S))

    return P

def deflate_matrix(A):
    # 1. Get eigenvalues and eigenvectors. 
    # eigh sorts them in ascending order, so index 0 is your zero eigenvalue.
    eigvals, eigvecs = jnp.linalg.eigh(A)
    # 2. Extract the eigenvector corresponding to the zero eigenvalue
    v_null = eigvecs[:, 0]
    # 3. Apply the rank-1 update
    # Note: jnp.outer(v, v) is mathematically v @ v.T
    A_deflated = A + eigvals[1] * jnp.outer(v_null, v_null)
    return A_deflated

m0 = seq.e0 @ seq.m0_sp.todense() @ seq.e0.T
m1 = seq.e1 @ seq.m1_sp.todense() @ seq.e1.T
m2 = seq.e2 @ seq.m2_sp.todense() @ seq.e2.T
m3 = seq.e3 @ seq.m3_sp.todense() @ seq.e3.T

dd0 = seq.e0 @ seq.grad_grad_sp.todense() @ seq.e0.T
dd1 = seq.e1 @ seq.curl_curl_sp.todense() @ seq.e1.T + seq.e1 @ seq.d0_sp.todense() @ seq.e0.T @ jnp.linalg.solve(m0, seq.e0 @ seq.d0_sp.todense().T @ seq.e1.T)
dd2 = seq.e2 @ seq.div_div_sp.todense() @ seq.e2.T + seq.e2 @ seq.d1_sp.todense() @ seq.e1.T @ jnp.linalg.solve(m1, seq.e1 @ seq.d1_sp.todense().T @ seq.e2.T)
dd2 = deflate_matrix(dd2)
dd3 = seq.e3 @ seq.d2_sp.todense() @ seq.e2.T @ jnp.linalg.solve(m2, seq.e2 @ seq.d2_sp.todense().T @ seq.e3.T)
dd3 = deflate_matrix(dd3)

dd0_jacobi_precond = jnp.diag(seq.dd0_sp_diaginv) @ dd0
dd1_jacobi_precond = jnp.diag(seq.dd1_sp_diaginv) @ dd1
dd2_jacobi_precond = jnp.diag(seq.dd2_sp_diaginv) @ dd2
dd3_jacobi_precond = jnp.diag(seq.dd3_sp_diaginv) @ dd3

m0_jacobi_precond = jnp.diag(seq.m0_sp_diaginv) @ m0
m1_jacobi_precond = jnp.diag(seq.m1_sp_diaginv) @ m1
m2_jacobi_precond = jnp.diag(seq.m2_sp_diaginv) @ m2
m3_jacobi_precond = jnp.diag(seq.m3_sp_diaginv) @ m3
# Form the Schur complement:
#
# M = [ A  C]
#     [C^T D]
#
# S = A - C D^{-1} C^T where D^{-1} is the inverse diagonal
# A is 3ns[1] x 3ns[1].



# %%
m0_schur_precond = get_schur_vector(m0, ((0, 3*ns[1]),)) @ m0
dd0_schur_precond = get_schur_vector(dd0, ((0, 3*ns[1]),)) @ dd0
m3_schur_precond = get_schur_vector(m3, ()) @ m3
dd3_schur_precond = get_schur_vector(dd3, ((0, 3*ns[1]),)) @ dd3
m1_schur_precond = get_schur_vector(m1, ((seq.n1_1, 2*ns[1]), (seq.n1_2, 3*ns[1])) ) @ m1
dd1_schur_precond = get_schur_vector(dd1, ((seq.n1_1, 2*ns[1]), (seq.n1_2, 3*ns[1])) ) @ dd1
m2_schur_precond = get_schur_vector(m2, ((0, 2*ns[1]),) ) @ m2
dd2_schur_precond = get_schur_vector(dd2, ((0, 2*ns[1]),) ) @ dd2
# %%
# ── Collect all condition numbers ──────────────────────────────────────────
labels = [r"$k=0$", r"$k=1$", r"$k=2$", r"$k=3$"]
colors = {"None": "#4477AA", "Jacobi": "#EE6677", "Schur": "#228833"}
bar_width = 0.22


def _grouped_bar(ax, labels, c_raw, c_jac, c_sch):
    """Draw a grouped bar chart on *ax* and annotate each bar."""
    x = np.arange(len(labels))
    bars_raw = ax.bar(x - bar_width, c_raw, bar_width, label="None",
                      color=colors["None"], edgecolor="white", linewidth=0.5)
    bars_jac = ax.bar(x, c_jac, bar_width, label="Jacobi",
                      color=colors["Jacobi"], edgecolor="white", linewidth=0.5)
    bars_sch = ax.bar(x + bar_width, c_sch, bar_width, label="Schur",
                      color=colors["Schur"], edgecolor="white", linewidth=0.5)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=13)
    ax.set_ylabel(r"$\kappa$  (condition number)", fontsize=12)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(axis="y", which="both", linewidth=0.3, alpha=0.6)
    ax.set_axisbelow(True)
    for bars in [bars_raw, bars_jac, bars_sch]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.1e}",
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7, rotation=45)


# ── Figure 1: Mass matrices ───────────────────────────────────────────────
cond_mass_raw    = [float(jnp.linalg.cond(m0)),
                    float(jnp.linalg.cond(m1)),
                    float(jnp.linalg.cond(m2)),
                    float(jnp.linalg.cond(m3))]
cond_mass_jacobi = [float(jnp.linalg.cond(m0_jacobi_precond)),
                    float(jnp.linalg.cond(m1_jacobi_precond)),
                    float(jnp.linalg.cond(m2_jacobi_precond)),
                    float(jnp.linalg.cond(m3_jacobi_precond))]
cond_mass_schur  = [float(jnp.linalg.cond(m0_schur_precond)),
                    float(jnp.linalg.cond(m1_schur_precond)),
                    float(jnp.linalg.cond(m2_schur_precond)),
                    float(jnp.linalg.cond(m3_schur_precond))]

fig1, ax1 = plt.subplots(figsize=(7, 5))
_grouped_bar(ax1, labels, cond_mass_raw, cond_mass_jacobi, cond_mass_schur)
ax1.set_title(f"Mass matrices — $n={n}$, $p={p}$, polar+Dirichlet",
              fontsize=14, fontweight="bold")
fig1.tight_layout()
fig1.savefig("condition_numbers_mass.pdf", bbox_inches="tight", dpi=150)
plt.show()

# ── Figure 2: Hodge–Laplacians (dd0 – dd3) ────────────────────────────────
cond_dd_raw    = [float(jnp.linalg.cond(dd0)),
                  float(jnp.linalg.cond(dd1)),
                  float(jnp.linalg.cond(dd2)),
                  float(jnp.linalg.cond(dd3))]
cond_dd_jacobi = [float(jnp.linalg.cond(dd0_jacobi_precond)),
                  float(jnp.linalg.cond(dd1_jacobi_precond)),
                  float(jnp.linalg.cond(dd2_jacobi_precond)),
                  float(jnp.linalg.cond(dd3_jacobi_precond))]
cond_dd_schur  = [float(jnp.linalg.cond(dd0_schur_precond)),
                  float(jnp.linalg.cond(dd1_schur_precond)),
                  float(jnp.linalg.cond(dd2_schur_precond)),
                  float(jnp.linalg.cond(dd3_schur_precond))]

fig2, ax2 = plt.subplots(figsize=(7, 5))
_grouped_bar(ax2, labels, cond_dd_raw, cond_dd_jacobi, cond_dd_schur)
ax2.set_title(r"Hodge–Laplacians $\delta d^k$" + f" — $n={n}$, $p={p}$, polar+Dirichlet",
              fontsize=14, fontweight="bold")
fig2.tight_layout()
fig2.savefig("condition_numbers_dd.pdf", bbox_inches="tight", dpi=150)
plt.show()
# %%
