# %%
"""
Relaxation of the Siesta nfp=3 stellarator equilibrium.

Loads the GVEC equilibrium data from an HDF5 file (80³ regular grid in
Clebsch coordinates), interpolates the coordinate map R(ρ,θ,ζ) and
Z(ρ,θ,ζ) onto B-splines, builds a de Rham sequence on the resulting
stellarator geometry, L²-projects the sampled B-field via
``project_sampled_field``, and runs the MRX relaxation loop.

Usage
----- 
    python scripts/relax_siesta.py
"""
import time

import h5py
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax.scipy.interpolate import RegularGridInterpolator

import mrx
from mrx.assembly import (assemble_dense_hodge_laplacian,
                          assemble_dense_mass_matrix)
from mrx.derham_sequence import DeRhamSequence
from mrx.differential_forms import DiscreteFunction, Pushforward
from mrx.mappings import interpolate_map, rotating_ellipse_map
from mrx.plotting import get_1d_grids
from mrx.preconditioners import get_mass_jacobi_diaginv
from mrx.relaxation import (DescentMethod, IntegrationScheme, TimeStepChoice,
                            TimeStepper, apply_diffusion, compute_force,
                            relaxation_loop)
from mrx.utils import evaluate_at_xq, integrate_against

jax.config.update("jax_enable_x64", True)

NS = [10] * 3
PS = [3] * 3
QUAD_ORDER = 2 * PS[0]

map_func = rotating_ellipse_map()

seq = DeRhamSequence(
    NS, PS, QUAD_ORDER,
    ("clamped", "periodic", "periodic"),
    polar=True, tol=1e-6, maxiter=1000
)
seq.set_map(map_func)
seq.evaluate_1d()
seq.assemble_mass_matrix(0)
seq.assemble_hodge_laplacian(0)

# %%
m0_dense = seq.e0.todense() @ seq.m0.todense() @ seq.e0_T.todense()
# %%
plt.imshow(jnp.log10(m0_dense))
plt.colorbar()
plt.title("Mass matrix")
plt.show()
# %%
# %%
m11 = m0_dense[:3*NS[2], :3*NS[2]]
m12 = m0_dense[:3*NS[2], 3*NS[2]:]
m21 = m0_dense[3*NS[2]:, :3*NS[2]]
m22 = m0_dense[3*NS[2]:, 3*NS[2]:]

S = m11 - m12 @ jnp.linalg.solve(m22, m21)

P = jnp.block([[jnp.linalg.inv(S), jnp.zeros_like(m12)],
               [jnp.zeros_like(m21), jnp.diag(1/jnp.diag(m22))]])
P2 = jnp.block([[jnp.linalg.inv(m11), jnp.zeros_like(m12)],
                [jnp.zeros_like(m21), jnp.diag(1/jnp.diag(m22))]])


def preconditioned_matrix(P, M):
    """Compute sqrt(P) @ M @ sqrt(P), which is SPD if P and M are SPD,
    and has the same eigenvalues as P @ M."""
    sqrtP = np.linalg.cholesky(np.array(P))
    return sqrtP @ np.array(M) @ sqrtP.T


operators = seq.get_operators()
D_jac = jnp.sqrt(np.diag(get_mass_jacobi_diaginv(operators.mass_preconds, 0, False)))

preconditioned = {
    "Unpreconditioned": np.array(m0_dense),
    "Jacobi": D_jac @ np.array(m0_dense) @ D_jac,
    "Schur complement": preconditioned_matrix(P, m0_dense),
    "Block-diag (inv $M_{11}$)": preconditioned_matrix(P2, m0_dense),
}

fig, ax = plt.subplots(figsize=(8, 5))

colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
cond_numbers = {}

for (label, mat), color in zip(preconditioned.items(), colors):
    eigs = np.sort(np.linalg.eigvalsh(mat))
    cond = eigs[-1] / eigs[eigs > 0][0]
    cond_numbers[label] = cond
    ax.semilogy(np.arange(len(eigs)), eigs, "o-", markersize=3, linewidth=1,
                label=f"{label}  ($\\kappa={cond:.2e}$)", color=color)

ax.set_xlabel("Eigenvalue index (sorted)")
ax.set_ylabel("Eigenvalue (log scale)")
ax.set_title("Preconditioned eigenvalue spectra")
ax.legend(fontsize=8)

best_label = min(cond_numbers, key=cond_numbers.get)
fig.suptitle(
    f"Best for iterative solvers: \"{best_label}\"  "
    f"($\\kappa = {cond_numbers[best_label]:.2e}$)\n"
    "Tighter eigenvalue clustering → faster Krylov convergence",
    fontsize=10,
)
plt.tight_layout()
plt.show()
# %%
jnp.linalg.cond(m22)
# %%
