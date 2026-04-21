# %%
"""Interactive demo for weak grad/curl/div with prescribed boundary dual terms.

This script is intentionally algebraic: ``boundary_dual`` is the dual-space
load that represents the boundary functional after you have assembled it.
The weak operators then apply

    weak_op(v; boundary_dual) = weak_op(v; 0) + M^{-1} boundary_dual.

So if you later assemble a surface integral like

    b_i = \int_{\partial\Omega} <trace term, test trace_i> dS,

you pass that assembled vector in as ``boundary_dual=b``.
"""

import jax
import jax.numpy as jnp
import numpy.testing as npt

from mrx.derham_sequence import DeRhamSequence
from mrx.differential_forms import DiscreteFunction, Pushforward
from mrx.mappings import rotating_ellipse_map
from mrx.projectors import BoundaryProjector
import mrx

jax.config.update("jax_enable_x64", True)



# %%
types = ("clamped", "periodic", "periodic")
BETTI = [1, 1, 0, 0]


def build_seq(n=4, p=2):
    seq = DeRhamSequence(
        (n, n, n),
        (p, p, p),
        2 * p,
        types,
        rotating_ellipse_map(nfp=3),
        polar=True,
        tol=1e-12,
        maxiter=1000,
    )
    seq.evaluate_1d()
    seq.assemble_all_sparse()
    seq._compute_nullspaces(BETTI)
    seq.assemble_projection_matrix(2,1)
    return seq


seq = build_seq()
print("Built sequence")
print(f"n0={seq.n0}, n1={seq.n1}, n2={seq.n2}, n3={seq.n3}")


# %%
def inspect_weak_operator(seq, operator_name, v, boundary_dual, out_degree):
    """Show how an explicit boundary dual enters a weak operator."""
    operator = getattr(seq, operator_name)

    base = operator(v, dirichlet_in=False, dirichlet_out=False)
    with_boundary = operator(
        v,
        dirichlet_in=False,
        dirichlet_out=False,
        boundary_dual=boundary_dual,
    )
    expected_shift = seq.apply_inverse_mass_matrix(
        boundary_dual, out_degree, dirichlet=False)
    delta = with_boundary - base

    print(f"\n{operator_name}")
    print(f"  input shape         : {v.shape}")
    print(f"  boundary_dual shape : {boundary_dual.shape}")
    print(f"  ||base||            : {float(jnp.linalg.norm(base)):.6e}")
    print(f"  ||delta||           : {float(jnp.linalg.norm(delta)):.6e}")
    print(f"  ||expected_shift||  : {float(jnp.linalg.norm(expected_shift)):.6e}")
    print(f"  ||delta-shift||     : {float(jnp.linalg.norm(delta - expected_shift)):.6e}")

    npt.assert_allclose(delta, expected_shift, atol=1e-12)
    return {
        "base": base,
        "with_boundary": with_boundary,
        "delta": delta,
        "expected_shift": expected_shift,
    }


def random_pair(key, n_in, n_dual):
    key_v, key_b = jax.random.split(key)
    v = jax.random.normal(key_v, (n_in,))
    boundary_dual = jax.random.normal(key_b, (n_dual,))
    return v, boundary_dual


def assemble_weak_grad_boundary_dual(seq, boundary_vec, dirichlet=True):
    """Assemble the weak-grad surface term in the 2-form test space.

    For a scalar field p and physical boundary vector field g = p n_hat, this
    returns the dual load

        b_i = \int_{r=1} g \cdot trace(phi_i^(2)) dS

    where the test basis phi_i^(2) is the same basis used by ``apply_weak_grad``.
    This is not the same object as ``seq.p2_bc(g)``, which only keeps the
    boundary trace DOFs.
    """
    projector = BoundaryProjector(seq, 2)
    nt, nz = projector._nt, projector._nz

    g_jk = jax.lax.map(
        boundary_vec,
        projector._xi_bdy,
        batch_size=mrx.MAP_BATCH_SIZE_INNER,
    ).reshape(nt, nz, -1)

    # Pull back the physical flux to the logical 2-form components used by the
    # reference-space basis functions.
    g_log = jnp.einsum(
        "tzij,tzi->tzj", projector._DF_bdy, g_jk
    ) / projector._J_bdy[:, :, None]

    wg0 = g_log[:, :, 0] * projector._w_surf
    part_r = jnp.einsum("tz,bz,ct->bc", wg0, seq.d_basis_z_jk, seq.d_basis_t_jk)
    b_r = jnp.einsum("a,bc->abc", projector._basis_r_1, part_r).ravel()

    wg1 = g_log[:, :, 1] * projector._w_surf
    part_t = jnp.einsum("tz,bz,ct->bc", wg1, seq.d_basis_z_jk, seq.basis_t_jk)
    b_t = jnp.einsum("a,bc->abc", projector._d_basis_r_1, part_t).ravel()

    wg2 = g_log[:, :, 2] * projector._w_surf
    part_z = jnp.einsum("tz,bz,ct->bc", wg2, seq.basis_z_jk, seq.d_basis_t_jk)
    b_z = jnp.einsum("a,bc->abc", projector._d_basis_r_1, part_z).ravel()

    b_full = jnp.concatenate([b_r, b_t, b_z])
    e2_test = seq.e2_dbc if dirichlet else seq.e2
    return e2_test @ b_full


print(
    "Pass boundary loads in the dual target space. The operator adds them before the mass solve."
)


# %%
v3, b2_dual = random_pair(jax.random.PRNGKey(10), seq.n3, seq.n2)
weak_grad_data = inspect_weak_operator(
    seq,
    "apply_weak_grad",
    v3,
    b2_dual,
    out_degree=2,
)


# %%
v2, b1_dual = random_pair(jax.random.PRNGKey(11), seq.n2, seq.n1)
weak_curl_data = inspect_weak_operator(
    seq,
    "apply_weak_curl",
    v2,
    b1_dual,
    out_degree=1,
)


# %%
v1, b0_dual = random_pair(jax.random.PRNGKey(12), seq.n1, seq.n0)
weak_div_data = inspect_weak_operator(
    seq,
    "apply_weak_div",
    v1,
    b0_dual,
    out_degree=0,
)


# %%
def field(xi):
    return seq.map(xi)

u1_full = seq.apply_inverse_mass_matrix(seq.p1(field), 1, dirichlet=False)
weak0_full = seq.apply_weak_div(u1_full, dirichlet_in=False, dirichlet_out=False)

g_bc = seq.e1_bc @ (seq.e1_T @ u1_full)
lift_spline = seq.bc_lift(g_bc, 1)
lift_full = seq.e1 @ lift_spline
interior_full = u1_full - lift_full
lift_dual = -seq.apply_derivative_matrix(lift_full, 0, dirichlet_in=False, dirichlet_out=False, transpose=True)
weak0_split = seq.apply_weak_div(interior_full, dirichlet_in=False, dirichlet_out=False, boundary_dual=lift_dual)

print('norm full-split', float(jnp.linalg.norm(weak0_full - weak0_split)))
print('boundary dofs norm', float(jnp.linalg.norm(g_bc)))
print('lift norm', float(jnp.linalg.norm(lift_full)))

weak0_fun = DiscreteFunction(weak0_split, seq.basis_0, seq.e0)
def exact(xi):
    return jnp.array([3.0])

diff = jax.lax.map(lambda x: weak0_fun(x) - exact(x), seq.quad.x, batch_size=0)
ref = jax.lax.map(exact, seq.quad.x, batch_size=0)
wJ = seq.jacobian_j * seq.quad.w
num = jnp.einsum('ik,ik,i->', diff, diff, wJ)
den = jnp.einsum('ik,ik,i->', ref, ref, wJ)
print('relerr to 3', float(jnp.sqrt(num / den)))
# %%
three0 = seq.apply_inverse_mass_matrix(
    seq.p0(lambda xi: jnp.array([3.0])),
    0,
    dirichlet=False,
)

print("coeff weak vs projected 3", float(jnp.linalg.norm(weak0_full - three0)))
print("l2 weak-3", float(seq.l2_norm(weak0_full - three0, 0, dirichlet=False)))
print("l2 3", float(seq.l2_norm(three0, 0, dirichlet=False)))
# %%
vals = jax.vmap(DiscreteFunction(weak0_full, seq.basis_0, seq.e0))(seq.quad.x)
wJ = seq.jacobian_j * seq.quad.w

print("min/max", float(vals.min()), float(vals.max()))
print("weighted mean", float(jnp.einsum("ik,i->", vals, wJ) / jnp.sum(wJ)))
print("first few", vals[:8, 0])
# %%
minus_three0 = seq.apply_inverse_mass_matrix(
    seq.p0(lambda xi: jnp.array([-3.0])),
    0,
    dirichlet=False,
)

print("coeff weak vs projected -3", float(jnp.linalg.norm(weak0_full - minus_three0)))
# %%
strong3 = seq.apply_strong_div(u1_full, dirichlet_in=False, dirichlet_out=False)
strong0 = seq.apply_projection_matrix(strong3, 3, 0, dirichlet_in=False, dirichlet_out=False)

print("weak vs strong-projected", float(jnp.linalg.norm(weak0_full - strong0)))
print("strong-projected vs 3", float(jnp.linalg.norm(strong0 - three0)))
# %%
