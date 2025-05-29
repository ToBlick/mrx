# %%
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from mrx.BoundaryConditions import LazyBoundaryOperator
from mrx.DifferentialForms import DifferentialForm
from mrx.LazyMatrices import LazyDerivativeMatrix, LazyMassMatrix, LazyStiffnessMatrix
from mrx.Quadrature import QuadratureRule

# Enable 64-bit precision for numerical stability
jax.config.update("jax_enable_x64", True)

# Get the absolute path to the script's directory
script_dir = Path(__file__).parent.absolute()
# Create the output directory in the same directory as the script
output_dir = script_dir / 'script_outputs'
os.makedirs(output_dir, exist_ok=True)

# Initialize differential forms and operators
ns = (12, 12, 1)  # Number of elements in each direction
ps = (3, 3, 0)  # Polynomial degree in each direction
types = ('clamped', 'clamped', 'constant')  # Boundary conditions
bcs = ('dirichlet', 'dirichlet', 'none')
# Define differential forms for different function spaces
Λ0 = DifferentialForm(0, ns, ps, types)  # H1 functions
Λ1 = DifferentialForm(1, ns, ps, types)  # H(curl) vector fields
Λ2 = DifferentialForm(2, ns, ps, types)  # H(div) vector fields
Λ3 = DifferentialForm(3, ns, ps, types)  # L2 densities

# Set up quadrature rule
Q = QuadratureRule(Λ0, 6)


def F(x): return x * jnp.pi


def generalized_eigh(A, B):
    Q = jnp.linalg.cholesky(B)
    # Q_inv = jnp.linalg.inv(Q)
    # C = Q_inv @ A @ Q_inv.T
    # Q B.T = A.T -> B Q.T = A
    # Q C = B
    C = jnp.linalg.solve(Q, jnp.linalg.solve(Q, A.T).T)
    eigenvalues, eigenvectors_transformed = jax.scipy.linalg.eigh(C)
    eigenvectors_original = jnp.linalg.solve(Q.T, eigenvectors_transformed)
    return eigenvalues, eigenvectors_original


_end = 26
# %%
for i in range(2):
    if i == 0:
        M0 = LazyMassMatrix(Λ0, Q, F=F).M
        L = LazyStiffnessMatrix(Λ0, Q, F=F).M
        evs, evecs = generalized_eigh(L, M0)
    else:
        B0, B1, B2, B3 = [LazyBoundaryOperator(
            Λ, bcs).M for Λ in (Λ0, Λ1, Λ2, Λ3)]
        M2 = LazyMassMatrix(Λ2, Q, F, B2).M
        M3 = LazyMassMatrix(Λ3, Q, F, B3).M
        D2 = LazyDerivativeMatrix(Λ2, Λ3, Q, F, B2, B3).M
        L = D2 @ jnp.linalg.solve(M2, D2.T)
        evs, evecs = generalized_eigh(L, M3)
    evs = evs[1:_end]

    true_evs = jnp.sort(jnp.array(
        [i**2 + j**2 for i in range(0, ns[0]+1) for j in range(0, ns[1]+1)]))[1:_end]
    fig, ax = plt.subplots()
    ax.set_yticks((true_evs[:]))
    ax.set_xticks(jnp.arange(1, _end + 1)[::2])
    ax.yaxis.grid(True, which='both')
    ax.xaxis.grid(True, which='both')
    ax.set_ylabel('λ')
    ax.legend()
    ax.plot(jnp.arange(1, _end), evs, marker='v', label='λ/ᴨ²')
    ax.plot(jnp.arange(1, _end), true_evs,
            marker='*', label='λ/ᴨ²', linestyle='')
    ax.set_xlabel('n')
    if i == 0:
        ax.set_title('Eigenvalues of the Laplace operator: k = 0')
        plt.savefig(
            "script_outputs/Helmholtz_eigenvalues_k=0.png", dpi=300, bbox_inches="tight"
        )
    else:
        ax.set_title('Eigenvalues of the Laplace operator: k = 3')
        plt.savefig(
            "script_outputs/Helmholtz_eigenvalues_k=3.png", dpi=300, bbox_inches="tight"
        )
# %%
l1_error = jnp.sum(jnp.abs(evs - true_evs)) / jnp.sum(true_evs)
