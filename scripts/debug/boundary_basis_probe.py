"""Which basis functions are non-zero at the clamped boundary, and how big?

face_operator scales the face term by ``e[-1]**2`` with ``e`` the derivative
spline evaluated at r=1. If the non-zero index is not the last one -- the
DerivativeSpline index convention is ``dLam_i = s(x, i+1)`` -- that factor is
spurious and the face operator is off by orders of magnitude.
"""
from __future__ import annotations
import os, sys
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mrx.derham_sequence import DeRhamSequence

seq = DeRhamSequence((8, 16, 8), (3,) * 3, 6, ("clamped", "periodic", "periodic"),
                     polar=True, tol=1e-12, maxiter=100, betti_numbers=(1, 1, 0, 0))
seq.evaluate_1d()
lam, dlam = seq.basis_0.Λ[0], seq.basis_0.dΛ[0]
print(f"primal: n={lam.n} p={lam.p} type={lam.type}")
print(f"deriv : n={dlam.n} p={dlam.p} type={dlam.type}\n")
for tag, basis, n in (("primal", lam, int(lam.n)), ("deriv", dlam, int(dlam.n))):
    for x in (1.0 - 1e-8, 1.0 - 1e-4, 1.0):
        v = np.asarray(jax.vmap(lambda i: jnp.sum(basis(x, i)))(jnp.arange(n)))
        nz = np.flatnonzero(np.abs(v) > 1e-12)
        print(f"{tag:7s} x={x:<12.8g} nonzero idx={list(nz[-4:])} "
              f"vals={np.array2string(v[nz[-4:]], precision=4)} "
              f"max|v|={np.abs(v).max():.4e} v[-1]={v[-1]:.4e}")
    print()
