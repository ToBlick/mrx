"""
Vacuum Curl Eigenvalue Problem (β=0)

The curl eigenvalue problem is:
    ∇ × (∇ × B) = λ B  


Note: This implementation uses standard function spaces for now
"""
import jax
import jax.numpy as jnp
import numpy as np
# from mrx.relaxation import MRXDiagnostics
from mrx.derham_sequence import DeRhamSequence
from mrx.mappings import toroid_map

jax.config.update("jax_enable_x64", True)

# Tokamak example

def curl_eigenmodes(Seq, n_eigs: int = 10, tol: float = 1e-10):
    """
    Solve the curl eigenvalue problem: ∇ × (∇ × B) = λ B
    
    This solves the generalized eigenvalue problem:
        curl_curl @ B = λ M2 @ B
    
    where M2 is the mass matrix for 2-forms.
    
    Parameters
    ----------
    Seq : DeRhamSequence
    n_eigs : int, default=10 (Number of eigenvalues)
    tol : float, default=1e-10
    
    Returns
    -------
    eigvals  (ascending order by magnitude)
    eigvecs  (normalized)
    """
    # Generalized eigenvalue problem: curl_curl @ B = λ M2 @ B
    # Convert to standard eigenvalue problem: M2^{-1} @ curl_curl @ B = λ B
    curl_curl = Seq.strong_curl @ Seq.weak_curl
    M = Seq.M2
    
    A = jnp.linalg.solve(M, curl_curl)
    eigvals, eigvecs = jnp.linalg.eigh(A) #A = M2^{-1} @ curl_curl
    
    # Sort by eigenvalue magnitude
    idx = jnp.argsort(jnp.abs(eigvals))
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    # Return n_eigs number of eigenvalues
    n_eigs = min(n_eigs, int(eigvals.shape[0]))
    return eigvals[:n_eigs], eigvecs[:, :n_eigs]

# Parameters
eps = 0.33
kappa = 1.0
Phi = toroid_map(epsilon=eps, kappa=kappa, R0=1.0)

# Resolution
ns = (4, 4, 1)  # Axisymmetric 
ps = (3, 3, 0)
q = 5


# Set up DeRham sequence
Seq = DeRhamSequence(
    ns, ps, q,
    ("clamped", "periodic", "constant"),  
    Phi,
    polar=True,
    dirichlet=True
)

# Assemble operators
Seq.evaluate_1d()
Seq.assemble_all()
Seq.build_crossproduct_projections()
Seq.assemble_leray_projection()



# Solve curl eigenvalue problem with tolerance 1e-10
eigvals, eigvecs = curl_eigenmodes(Seq,n_eigs=10, tol=1e-10)

eigvals_np = np.asarray(eigvals)

print("\nFirst 10 curl eigenvalues:")
for i, λ in enumerate(eigvals_np):
    print(f"  λ_{i+1} = {λ:.6e}")

# Find curl-free fields
tol = 1e-10
# Convert to numpy 
eigvals_np = np.asarray(eigvals)
curl_free_np = np.abs(eigvals_np) < tol
n_curl_free = int(np.sum(curl_free_np))
# indices where curl_free is True
curl_free_indices = np.where(curl_free_np)[0]
# Extract columns 
if n_curl_free > 0:
    curl_free_B = jnp.take(eigvecs, curl_free_indices, axis=1)
else:
    curl_free_B = jnp.empty((eigvecs.shape[0], 0))

print(f"\nNumber of curl-free fields: {n_curl_free}")

# Apply Leray projection to enforce divergence-free 
curl_free_B_proj = Seq.P_Leray @ curl_free_B

# Verify curl-free and divergence-free properties
print("\nChecking curl-free fields:")
if n_curl_free > 0:
    for i in range(n_curl_free):
        B_curl_free = curl_free_B_proj[:, i]
        
        # Check that curl vanishes
        curl_B = Seq.weak_curl @ B_curl_free
        curl_norm = (curl_B @ Seq.M1 @ curl_B)**0.5
        print(f" For curl-free field {i+1}, ||∇ × B|| = {curl_norm:.2e}")
        
        # Check that divergence vanishes 
        div_B = Seq.strong_div @ B_curl_free
        div_norm = (div_B @ Seq.M3 @ div_B)**0.5
        print(f" For curl-free field {i+1}, ||∇ · B|| = {div_norm:.2e}")

else:
    print("No curl-free fields found.")