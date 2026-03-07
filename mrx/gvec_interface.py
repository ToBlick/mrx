# %%
import jax
import jax.numpy as jnp

import mrx
from mrx.differential_forms import DiscreteFunction
from mrx.mappings import stellarator_map


def load_and_reshape_GVEC(gvec_eq, nfp):
    """
    Load GVEC equilibrium data and reshape it.

    Parameters
    ----------
    gvec_eq : xarray.Dataset
        GVEC equilibrium dataset.

    Returns
    -------
    x : jnp.ndarray
        evaluation points reshaped to (n_pts, 3).
    R : jnp.ndarray
        R coordinates reshaped to (n_pts,).
    Z : jnp.ndarray
        Z coordinates reshaped to (n_pts,).
    B : jnp.ndarray
        B field reshaped to (n_pts, 3).
    """
    _ρ = gvec_eq["rho"].values      # shape (mρ,)
    _θ = gvec_eq["theta"].values    # shape (mθ,)
    _ζ = gvec_eq["zeta"].values     # shape (mζ,)
    R = gvec_eq["X1"].values       # shape (mρ, mθ, mζ)
    Z = gvec_eq["X2"].values       # shape (mρ, mθ, mζ)
    B = gvec_eq["B"].values        # shape (mρ, mθ, mζ, 3)
    
    ρ, θ, ζ = jnp.meshgrid(_ρ, _θ, _ζ, indexing="ij")
    # θ_star = jnp.asarray(θ_star)
    pts = jnp.stack([ρ.ravel(),
                    θ.ravel() / (2 * jnp.pi),
                    ζ.ravel() / (2 * jnp.pi) * nfp], axis=1)  # x_hat_js, shape (mρ mθ mζ, 3)
    return (pts.reshape(-1, 3),
           R.ravel(),
           Z.ravel(),
           B.reshape(-1, 3))


def interpolate_map_from_GVEC(gvec_eq, nfp, mapSeq):
    """
    Interpolate the GVEC map onto a DeRham sequence.

    Parameters
    ----------
    gvec_eq : xarray.Dataset
        GVEC equilibrium dataset.
    nfp : int
        Number of field periods.
    mapSeq : DeRhamSequence
        DeRham sequence to interpolate the map onto.

    Returns
    -------
    gvec_stellarator_map : callable
        GVEC stellarator map.
    """
    _ρ = gvec_eq["rho"].values      # shape (mρ,)
    _θ = gvec_eq["theta"].values    # shape (mθ,)
    _ζ = gvec_eq["zeta"].values     # shape (mζ,)
    X1 = gvec_eq["X1"].values       # shape (mρ, mθ, mζ)
    X2 = gvec_eq["X2"].values       # shape (mρ, mθ, mζ)

    # Set up the interpolation problem:
    # ∑ c_ki Λ0[i](ρ,θ,ζ)_j ≈ Xk(ρ,θ,ζ)_j ∀j
    # evaluation grid, shape (mρ, mθ, mζ)
    ρ, θ, ζ = jnp.meshgrid(_ρ, _θ, _ζ, indexing="ij")
    # θ_star = jnp.asarray(θ_star)
    pts = jnp.stack([ρ.ravel(),
                    θ.ravel() / (2 * jnp.pi),
                    ζ.ravel() / (2 * jnp.pi) * nfp], axis=1)  # x_hat_js, shape (mρ mθ mζ, 3)

    M = mrx.double_map(
        lambda i, x: mapSeq.Lambda_0[i](x)[0],
        mapSeq.Lambda_0.ns, pts,
    ).T  # Λ0[i](x_hat_j)
    y = jnp.stack([X1.ravel(), X2.ravel()], axis=1)  # X_α(x'_j)
    c, _, _, _ = jnp.linalg.lstsq(M, y, rcond=None)

    X1_h = DiscreteFunction(c[:, 0], mapSeq.Lambda_0, mapSeq.E0)
    X2_h = DiscreteFunction(c[:, 1], mapSeq.Lambda_0, mapSeq.E0)

    return stellarator_map(X1_h, X2_h, nfp=nfp, flip_zeta=False), X1_h, X2_h


def interpolate_B_from_GVEC(gvec_eq, Seq, Phi, nfp, exclude_axis_tol=1e-3):
    """
    Interpolate GVEC B-field onto Seq.Lambda_2 basis.

    Parameters
    ----------
    gvec_eq : xarray.Dataset
        GVEC equilibrium dataset.
    Seq : DeRhamSequence
        DeRham sequence to interpolate the B-field onto.
    Phi : callable
        Mapping from logical coordinates to physical coords: (r,theta,zeta)->(x,y,z)
    nfp : int
        Number of field periods.
    exclude_axis_tol : float
        Tolerance for excluding points near the axis and exact boundary.

    Returns
    -------
    B_dof : jnp.ndarray
        B-field coefficients.
    residuals : jnp.ndarray
        Residuals of the interpolation.
    rank : int
        Rank of the interpolation.
    s : jnp.ndarray
        Singular values of the interpolation.
    """
    # build pts from gvec_eq if not provided (expects rho, theta, zeta coords)
    _ρ = jnp.array(gvec_eq.rho.values)
    _θ = jnp.array(gvec_eq.theta.values)
    _ζ = jnp.array(gvec_eq.zeta.values)
    ρ, θ, ζ = jnp.meshgrid(_ρ, _θ, _ζ, indexing="ij")
    # θ_star = jnp.asarray(θ_star)
    pts = jnp.stack([ρ.ravel(),
                    θ.ravel() / (2 * jnp.pi),
                    ζ.ravel() / (2 * jnp.pi) * nfp], axis=1)  # x_hat_js, shape (mρ mθ mζ, 3)
    # valid interpolation points (avoid axis and exact boundary)
    valid_pts = (pts[:, 0] > exclude_axis_tol) & (
        pts[:, 0] < 1 - exclude_axis_tol)

    def Λ2_phys(i, x):
        """
        Evaluate the physical 2-form basis function Phi*Λ2[i] at x.

        Parameters
        ----------
        i : int
            Index of the basis function.
        x : jnp.ndarray
            Point to evaluate the basis function at.

        Returns
        -------
        jnp.ndarray
            Value of the basis function at x.
        """
        # Pullback of basis function
        DPhix = jax.jacfwd(Phi)(x)  # Jacobian of Phi at x
        J = jnp.linalg.det(DPhix)
        return DPhix @ Seq.Lambda_2[i](x) / J

    valid_points = pts[valid_pts]

    M = mrx.double_map(Λ2_phys, Seq.Lambda_2.ns, valid_points)
    M = jnp.einsum('il,ljk->ijk', Seq.E2, M)        # Λ2[i](x_hat_j)_k
    y = gvec_eq.B.values.reshape(-1, 3)[valid_pts]  # B(x'_j)_k
    A = M.reshape(M.shape[0], -1).T
    b = y.ravel()
    # Solve least squares
    B_dof, residuals, _, _ = jnp.linalg.lstsq(A, b, rcond=None)
    return B_dof, residuals
