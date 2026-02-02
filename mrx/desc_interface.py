# desc_interface.py
"""
Interface for loading and projecting DESC equilibria onto MRX finite element spaces.
"""
from typing import Any

import desc
import desc.grid
import desc.io
import jax
import jax.numpy as jnp
from jax import Array

from mrx.derham_sequence import DeRhamSequence
from mrx.differential_forms import DiscreteFunction, Pushforward
from mrx.mappings import stellarator_map
from mrx.utils import integrate_against


class DESCWrapper:
    """
    Wrapper around a DESC equilibrium that evaluates R, Z, B at given points.

    Note: DESC uses (rho, theta, zeta) in [0,1] x [0,2π] x [0,2π/nfp] internally,
    but we normalize to [0,1]^3 for compatibility with MRX.
    """

    def __init__(self, eq: Any):
        """
        Initialize wrapper from a DESC equilibrium.

        Args:
            eq: DESC Equilibrium object
        """
        self.eq = eq
        self.nfp = eq.NFP

    def compute_at_points(self, points: Array) -> dict[str, Array]:
        """
        Compute R, Z, B at the given logical coordinates.

        Args:
            points: Array of shape (n_pts, 3) with coordinates in [0,1]^3
                    (rho, theta_normalized, zeta_normalized)

        Returns:
            Dictionary with 'R', 'Z', 'B' arrays evaluated at the points
        """
        # Convert from [0,1]^3 to DESC's coordinate convention
        pts_desc = points.copy()
        pts_desc = pts_desc.at[:, 1].set(
            points[:, 1] * 2 * jnp.pi)  # theta: [0,1] -> [0,2π]
        pts_desc = pts_desc.at[:, 2].set(
            points[:, 2] * 2 * jnp.pi / self.nfp)  # zeta: [0,1] -> [0,2π/nfp]

        # Create grid from the points
        grid = desc.grid.Grid(pts_desc, sort=False)
        vals = self.eq.compute(["R", "Z", "B"], grid=grid, basis="xyz")

        return {
            'R': jnp.array(vals["R"]),      # shape (n_pts,)
            'Z': jnp.array(vals["Z"]),      # shape (n_pts,)
            'B': jnp.array(vals["B"]),      # shape (n_pts, 3)
        }


def project_desc_equilibrium(
    desc_path: str,
    ns: tuple[int, int, int] = (4, 8, 4),
    ps: tuple[int, int, int] = (3, 3, 3),
) -> dict[str, Any]:
    """
    Load DESC equilibrium and project R, Z, B onto finite element spaces.

    Args:
        desc_path: Path to DESC equilibrium file
        ns: Number of basis functions in each direction
        ps: Polynomial degree

    Returns:
        Dictionary with projection results:
            - X1_h: DiscreteFunction for R
            - X2_h: DiscreteFunction for Z
            - F_h: Stellarator map from interpolated geometry
            - B_h: DiscreteFunction for B (2-form)
            - B_h_xyz: Pushforward of B_h to physical coordinates
            - map_seq: DeRhamSequence for geometry projection
            - seq: DeRhamSequence for B projection
            - wrapper: DESCWrapper instance
            - nfp: Number of field periods
    """
    # Load DESC equilibrium
    eq_fam = desc.io.load(desc_path)
    eq = eq_fam[-1]
    nfp = eq.NFP
    print(f"Loaded DESC equilibrium from {desc_path} (NFP={nfp})")

    # Create wrapper
    wrapper = DESCWrapper(eq)

    # ==========================================================================
    # Step 1: Project R and Z (geometry) onto 0-form space
    # ==========================================================================
    resolutions = ns
    spline_degrees = ps
    quad_order = ps[0]

    # Create sequence with identity map for geometry projection
    map_seq = DeRhamSequence(
        resolutions, spline_degrees, quad_order,
        ("clamped", "periodic", "periodic"),
        lambda x: x,  # Identity map initially
        polar=False,
        dirichlet=False
    )
    map_seq.evaluate_1d()
    map_seq.assemble_M0()

    # Evaluate R, Z at quadrature points
    quad_pts = map_seq.Q.x  # shape (n_q, 3)
    desc_vals = wrapper.compute_at_points(quad_pts)

    # Create callables that return the pre-computed values
    # (The projector will vmap over these, so we index by position)
    R_at_quad = desc_vals['R'][:, None]  # shape (n_q, 1) for 0-form
    Z_at_quad = desc_vals['Z'][:, None]  # shape (n_q, 1) for 0-form

    # For projection, we need callables - create simple indexing functions
    # But actually, we can compute the projection RHS directly since we have values at quad pts

    # RHS = ∫ f(x) * Λ_i(x) * J(x) * w(x) dx
    # Since map is identity, J = 1
    # We need to integrate manually here so we can use the batching
    # that DESC uses when evaluating stuff

    w_R = R_at_quad * (map_seq.Q.w * map_seq.J_j)[:, None]
    w_Z = Z_at_quad * (map_seq.Q.w * map_seq.J_j)[:, None]

    rhs_R = map_seq.E0 @ integrate_against(
        map_seq.get_Lambda_0_ijk, w_R, map_seq.Lambda_0.n)
    rhs_Z = map_seq.E0 @ integrate_against(
        map_seq.get_Lambda_0_ijk, w_Z, map_seq.Lambda_0.n)

    # Solve for coefficients: M @ c = rhs
    c_R = jnp.linalg.solve(map_seq.M0, rhs_R)
    c_Z = jnp.linalg.solve(map_seq.M0, rhs_Z)

    # Create discrete functions for R and Z
    X1_h = DiscreteFunction(c_R, map_seq.Lambda_0, map_seq.E0)  # R
    X2_h = DiscreteFunction(c_Z, map_seq.Lambda_0, map_seq.E0)  # Z

    # Compute R, Z projection errors
    R_exact = desc_vals['R']
    Z_exact = desc_vals['Z']
    R_interp = jax.vmap(X1_h)(map_seq.Q.x).ravel()
    Z_interp = jax.vmap(X2_h)(map_seq.Q.x).ravel()

    R_L2 = jnp.sqrt(jnp.sum((R_exact - R_interp)**2 * map_seq.Q.w) /
                    jnp.sum(R_exact**2 * map_seq.Q.w))
    Z_L2 = jnp.sqrt(jnp.sum((Z_exact - Z_interp)**2 * map_seq.Q.w) /
                    jnp.sum(Z_exact**2 * map_seq.Q.w))
    R_Linf = jnp.max(jnp.abs(R_exact - R_interp)) / jnp.max(jnp.abs(R_exact))
    Z_Linf = jnp.max(jnp.abs(Z_exact - Z_interp)) / jnp.max(jnp.abs(Z_exact))

    print("Geometry projection errors:")
    print(f"  R: L2={R_L2:.3e}, L∞={R_Linf:.3e}")
    print(f"  Z: L2={Z_L2:.3e}, L∞={Z_Linf:.3e}")

    # Create the stellarator map from interpolated geometry
    F_h = jax.jit(stellarator_map(X1_h, X2_h, nfp=nfp, flip_zeta=True))

    # ==========================================================================
    # Step 2: Project B onto 2-form space using the interpolated geometry
    # ==========================================================================

    # Create sequence with the interpolated map
    seq = DeRhamSequence(
        resolutions, spline_degrees, quad_order,
        ("clamped", "periodic", "periodic"),
        F_h,
        polar=True,
        dirichlet=True
    )
    seq.evaluate_1d()
    seq.assemble_M2()

    # Evaluate B at the new quadrature points (same logical coords, but need fresh eval)
    desc_vals_B = wrapper.compute_at_points(seq.Q.x)
    B_at_quad = desc_vals_B['B']  # shape (n_q, 3)

    # For 2-form projection: need to apply DF^T transformation
    # RHS = ∫ (DF^T @ B) · Λ_i * w dx
    DF = jax.jacfwd(seq.F)
    B_transformed = jax.vmap(lambda x, b: DF(x).T @ b)(seq.Q.x, B_at_quad)

    w_B = B_transformed * (seq.Q.w)[:, None]
    rhs_B = seq.E2 @ integrate_against(seq.get_Lambda_2_ijk,
                                       w_B, seq.Lambda_2.n)

    # Solve for B coefficients
    c_B = jnp.linalg.solve(seq.M2, rhs_B)

    # Create discrete function for B
    B_h = DiscreteFunction(c_B, seq.Lambda_2, seq.E2)
    B_h_xyz = Pushforward(B_h, seq.F, 2)

    # Compute B projection error
    B_exact = B_at_quad
    B_interp = jax.vmap(B_h_xyz)(seq.Q.x)
    J = seq.J_j

    B_diff_sq = jnp.sum((B_exact - B_interp)**2, axis=1)
    B_exact_sq = jnp.sum(B_exact**2, axis=1)
    B_L2 = jnp.sqrt(jnp.sum(B_diff_sq * J * seq.Q.w) /
                    jnp.sum(B_exact_sq * J * seq.Q.w))
    B_diff_norm = jnp.sqrt(B_diff_sq)
    B_exact_norm = jnp.sqrt(B_exact_sq)
    B_Linf = jnp.max(B_diff_norm) / jnp.max(B_exact_norm)

    print("B-field projection errors:")
    print(f"  B: L2={B_L2:.3e}, L∞={B_Linf:.3e}")

    return {
        'X1_h': X1_h,
        'X2_h': X2_h,
        'F_h': F_h,
        'B_h': B_h,
        'B_h_xyz': B_h_xyz,
        'map_seq': map_seq,
        'seq': seq,
        'wrapper': wrapper,
        'nfp': nfp,
    }
