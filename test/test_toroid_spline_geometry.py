import jax
import jax.numpy as jnp

from mrx.derham_sequence import DeRhamSequence
from mrx.differential_forms import DiscreteFunction
from mrx.io import project_sampled_field
from mrx.mappings import toroid_map


jax.config.update("jax_enable_x64", True)


types = ("clamped", "periodic", "periodic")


def _exact_u(x):
    r, chi, z = x
    pi = jnp.pi
    return 1 / 4 * (r ** 2 - r ** 4) * jnp.cos(2 * pi * z) * jnp.ones(1)


def _source_f(x):
    r, chi, z = x
    pi = jnp.pi
    a = 1 / 3
    radius = 1 + a * r * jnp.cos(2 * pi * chi)
    return (
        jnp.cos(2 * pi * z)
        * (
            -1 / a ** 2 * (1 - 4 * r ** 2)
            - 1 / (a * radius) * (r / 2 - r ** 3) * jnp.cos(2 * pi * chi)
            + 1 / 4 * (r ** 2 - r ** 4) / radius ** 2
        )
        * jnp.ones(1)
    )


def _solve_and_error_k0(seq):
    rhs = seq.p0_dbc(_source_f)
    u_hat = seq.apply_inverse_hodge_laplacian(rhs, k=0)
    u_h = DiscreteFunction(u_hat, seq.basis_0, seq.e0_dbc)

    diff_vals = jax.lax.map(
        lambda x: _exact_u(x) - u_h(x), seq.quad.x, batch_size=20_000
    )
    u_vals = jax.vmap(_exact_u)(seq.quad.x)
    l2_diff = jnp.einsum("ik,ik,i,i->", diff_vals,
                         diff_vals, seq.jacobian_j, seq.quad.w)
    l2_u = jnp.einsum("ik,ik,i,i->", u_vals, u_vals,
                      seq.jacobian_j, seq.quad.w)
    return float((l2_diff / l2_u) ** 0.5)


def test_poisson_k0_with_projected_toroid_spline_geometry():
    """Solve the torus k=0 Poisson problem after projecting the geometry map to splines."""
    p = 3
    n = 8
    ns = (n, n, n)
    ps = (p, p, p)
    a = 1 / 3
    torus_map = toroid_map(epsilon=a)

    print("Building reference sequence for torus spline-geometry Poisson test...")

    seq = DeRhamSequence(ns, ps, 2 * p, types, lambda x: x, polar=True,
                         tol=1e-12, maxiter=1000)
    seq.evaluate_1d()
    seq.assemble_reference_mass_matrix()

    n_sample = 40
    print(f"Projecting torus map to spline coefficients on a {n_sample}^3 logical grid...")
    r = jnp.linspace(0, 1, n_sample)
    chi = jnp.linspace(0, 1, n_sample)
    zeta = jnp.linspace(0, 1, n_sample)
    ri, chii, zetai = jnp.meshgrid(r, chi, zeta, indexing="ij")
    grid_pts = jnp.stack([ri.ravel(), chii.ravel(), zetai.ravel()], axis=1)
    map_grid = jax.vmap(torus_map)(grid_pts)

    coeffs = jnp.stack([
        project_sampled_field(
            (r, chi, zeta), map_grid[:, i], seq,
            k=0, dirichlet=False, reference_domain=True,
        )
        for i in range(3)
    ], axis=0)

    print("Switching sequence geometry to the spline map and assembling operators...")
    seq.set_spline_map(coeffs)
    seq.assemble_mass_matrix(0)
    seq.assemble_hodge_laplacian(0)
    seq.null_0_dbc = []

    print("Solving k=0 Poisson problem on the spline-projected torus geometry...")
    rel_error = _solve_and_error_k0(seq)
    print(f"Relative L2 error: {rel_error:.6e}")
    assert rel_error < 1e-2, (
        f"Relative L2 error too large after spline geometry projection: {rel_error:.2e}"
    )