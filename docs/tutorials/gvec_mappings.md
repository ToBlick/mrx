---
title: GVEC mappings
parent: Tutorials
layout: default
nav_order: 2
---

# GVEC mappings

In this tutorial, we explain how we can use the mappings defined by GVEC in MRX. This is a test in MRX.

### GVEC

GVEC (https://gvec.readthedocs.io) is an inverse-mapping type equilibrium code that operates under the assumption of nested flux surfaces. The main output of GVEC is the flux-coordinate mapping $\Psi : (\rho', \theta', \zeta') \mapsto (R, z)$ where $\rho'$ is a normalized flux-surface label, $\theta$ is a poloidal angle, and $\zeta'$ is a toroidal angle. By defining
$$
\begin{align}
\Phi_{\text{GVEC}}: \begin{pmatrix} \rho' \\ \theta' \\ \zeta' \end{pmatrix} &\mapsto \begin{pmatrix} R(\rho', \theta', \zeta') \cos(\zeta') \\ -R(\rho', \theta', \zeta') \sin(\zeta') \\ z(\rho', \theta', \zeta') \end{pmatrix} \\
\end{align}
$$
we obtain a map from the logical domain to the physical domain in $\mathbb{R}^3$.

GVEC allows the computation of so-called straight field line angle $\theta^*(\theta', \zeta') := \theta' + \lambda(\theta', \zeta')$. We chose to use these angles as poloidal angle in MRX. For more details on the definition of $\lambda$, we refer to the GVEC documentation. Let us call the GVEC coordinates $x' = (\rho', \theta', \zeta')$. The coordinates $\hat x = (r, \theta, \zeta)$ MRX mappings use remain on the unit cube and so we obtain the relations
$$
\begin{align}
r = \rho', \quad \theta = \frac{\theta^*(\theta', \zeta')}{2 \pi}, \quad \zeta = \frac{\zeta' n_{\text{fp}}}{2 \pi}
\end{align}
$$
with $n_{\text{fp}}$ the number of field periods.

### Input

We chose to re-construct $\Phi$ from $m$ sampled values of $X_1$ and $X_2$. This boils down to solving the following interpolation problem for both $X_1$ and $X_2$ separately:
$$
\begin{align}
    \min_{c} \left( \sum_{i = 0}^{n - 1} c_i^\alpha \, \Lambda^{0, \alpha}_i (\hat x_j) - X_\alpha(x'_j) \right)^2 \quad \forall \alpha = 1, 2 \text{ and } j = 1, \ldots, m .
\end{align}
$$
Here, $c_i^\alpha$ are the coefficients to be determined, $\Lambda^{0, \alpha}_i$ are the spline basis functions, and $X_\alpha(x'_j)$ are the sampled values. $\hat x_j$ are the coordinates of the sampled GVEC coordinates $x'_j$ mapped to MRX convention as described above.

GVEC provides these required sampled values in the form of an `xarray` dataset and supports setting our own $x'_j$ points. In this tutorial, we use pre-computed sampled values stored in `data/gvec_stellarator.h5`:

```python
p, n, nfp = 3, 8, 3
gvec_eq = xr.open_dataset("data/gvec_stellarator.h5", engine="h5netcdf")
θ_star = gvec_eq["thetastar"].values    # shape (mρ, mθ, mζ)
_ρ = gvec_eq["rho"].values              # shape (mρ,)
_θ = gvec_eq["theta"].values            # shape (mθ,)
_ζ = gvec_eq["zeta"].values             # shape (mζ,)
X1 = gvec_eq["X1"].values               # shape (mρ, mθ, mζ)
X2 = gvec_eq["X2"].values               # shape (mρ, mθ, mζ)
```

### Approximating the mapping in spline space

We now assemble the spline space that we use to approximate $X_1$ and $X_2$:
```python
mapSeq = DeRhamSequence((n, n, n), (p, p, p), p+2,("clamped", "periodic", "clamped"), lambda x: x, polar=False, dirichlet=False)
```
This mapping is periodic only in the polidal angle, since only a single field period is represented in the GVEC data. The `polar=False` flag might be surprising at first, but recall that this mapping is defined on the unit cube, not the polar spline space.

We can now build the interpolation matrix and solve for the spline coefficients using least-squares:
```python
ρ, θ, ζ = jnp.meshgrid(_ρ, _θ, _ζ, indexing="ij")    # evaluation grid, shape (mρ, mθ, mζ)
θ_star = jnp.asarray(θ_star)
pts = jnp.stack([ρ.ravel(), θ_star.ravel() / (2 * jnp.pi), ζ.ravel() / (2 * jnp.pi) * nfp], axis=1)  # x_hat_js, shape (mρ mθ mζ, 3)

M = jax.vmap(lambda i: jax.vmap(lambda x: mapSeq.Λ0[i](x)[0])(pts))(mapSeq.Λ0.ns).T # Λ0[i](x_hat_j)
y = jnp.stack([X1.ravel(), X2.ravel()], axis=1) # X_α(x'_j)
c, residuals, rank, s = jnp.linalg.lstsq(M, y, rcond=None)
```