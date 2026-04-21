# """Integration tests for the de Rham complex: grad, curl, div.

# Project toy analytical functions defined in physical (x, y, z) coordinates
# onto discrete spaces, apply derivative operators, and compare against
# analytical gradients / curls / divergences.
# """

# import jax
# import jax.numpy as jnp
# import pytest

# from mrx.derham_sequence import DeRhamSequence
# from mrx.differential_forms import DiscreteFunction, Pushforward
# from mrx.mappings import toroid_map
# from mrx.utils import integrate_against, inv33

# jax.config.update("jax_enable_x64", True)

# types = ("clamped", "periodic", "periodic")
# NS = (4, 5, 6)


# # ---------------------------------------------------------------------------
# # Toy functions in physical coordinates and their analytical derivatives
# # ---------------------------------------------------------------------------

# def scalar_f(a, F):
#     """f(x,y,z) = x² + y² + z²"""
#     x, y, z = F(a)
#     return jnp.array([x**2 + y**2 + z**2])


# def grad_f(a, F):
#     """∇f = (2x, 2y, 2z)"""
#     x, y, z = F(a)
#     return jnp.array([2*x, 2*y, 2*z])


# def vector_g(a, F):
#     """G(x,y,z) = (y², z², x²)"""
#     x, y, z = F(a)
#     return jnp.array([y**2, z**2, x**2])


# def curl_g(a, F):
#     """curl G = (-2z, -2x, -2y)"""
#     x, y, z = F(a)
#     return jnp.array([-2*z, -2*x, -2*y])


# def vector_h(a, F):
#     """H(x,y,z) = (x², y², z²)"""
#     x, y, z = F(a)
#     return jnp.array([x**2, y**2, z**2])


# def div_h(a, F):
#     """div H = 2x + 2y + 2z"""
#     x, y, z = F(a)
#     return jnp.array([2*x + 2*y + 2*z])


# # ---------------------------------------------------------------------------
# # Helpers
# # ---------------------------------------------------------------------------

# def _build_seq(n, p=2):
#     a = 1 / 3
#     F = toroid_map(epsilon=a)
#     seq = DeRhamSequence((n, n, n), (p, p, p), 2*p, types, F,
#                          polar=True, tol=1e-12, maxiter=1000)
#     seq.evaluate_1d()
#     for k in range(4):
#         seq.assemble_mass_matrix(k)
#     for k in range(3):
#         seq.assemble_derivative_matrix(k)
#     return seq


# def _project(seq, k, f, dirichlet=False):
#     """L2-project a physical-space function *f(a)* as a k-form.

#     *f* takes a single parametric point a = (r, θ, ζ) and returns:
#       - shape (1,) for k=0 or k=3
#       - shape (3,) for k=1 or k=2
#     """
#     F = seq.map
#     DF = jax.jacfwd(F)
#     quad_shape = (seq.quad.ny, seq.quad.nx, seq.quad.nz)
#     comp_info, comp_shapes = seq._form_comp_info(k)
#     es = [seq.e0, seq.e1, seq.e2, seq.e3]
#     es_dbc = [seq.e0_dbc, seq.e1_dbc, seq.e2_dbc, seq.e3_dbc]
#     e = es_dbc[k] if dirichlet else es[k]

#     if k == 0:
#         f_jk = jax.lax.map(f, seq.quad.x, batch_size=0)
#         w_jk = f_jk * (seq.quad.w * seq.jacobian_j)[:, None]
#     elif k == 1:
#         def _v(x):
#             return inv33(DF(x)) @ f(x)
#         f_jk = jax.lax.map(_v, seq.quad.x, batch_size=0)
#         w_jk = f_jk * (seq.quad.w * seq.jacobian_j)[:, None]
#     elif k == 2:
#         def _v(x):
#             return DF(x).T @ f(x)
#         f_jk = jax.lax.map(_v, seq.quad.x, batch_size=0)
#         w_jk = f_jk * (seq.quad.w)[:, None]
#     elif k == 3:
#         f_jk = jax.lax.map(f, seq.quad.x, batch_size=0)
#         w_jk = f_jk * (seq.quad.w)[:, None]

#     rhs = integrate_against(w_jk, comp_info, comp_shapes, quad_shape)
#     b = e @ rhs
#     return seq.apply_inverse_mass_matrix(b, k, dirichlet=dirichlet)


# def _l2_error(seq, k, dofs, f_exact, dirichlet=False):
#     """Relative L2 error between discrete k-form *dofs* and *f_exact(a)*."""
#     bases = [seq.basis_0, seq.basis_1, seq.basis_2, seq.basis_3]
#     es = [seq.e0, seq.e1, seq.e2, seq.e3]
#     es_dbc = [seq.e0_dbc, seq.e1_dbc, seq.e2_dbc, seq.e3_dbc]
#     e = es_dbc[k] if dirichlet else es[k]
#     u_h = Pushforward(DiscreteFunction(dofs, bases[k], e), seq.map, k)

#     diff = jax.lax.map(
#         lambda a: f_exact(a) - u_h(a), seq.quad.x, batch_size=0)
#     ref = jax.lax.map(f_exact, seq.quad.x, batch_size=0)
#     L2_d = jnp.einsum("ik,ik,i,i->", diff, diff, seq.jacobian_j, seq.quad.w)
#     L2_r = jnp.einsum("ik,ik,i,i->", ref, ref, seq.jacobian_j, seq.quad.w)
#     return float((L2_d / L2_r) ** 0.5)


# # ---------------------------------------------------------------------------
# # Module-scoped fixtures — one sequence per resolution, reused everywhere
# # ---------------------------------------------------------------------------

# @pytest.fixture(scope="module")
# def seqs():
#     return {n: _build_seq(n) for n in NS}


# @pytest.fixture(scope="module")
# def projected(seqs):
#     """Pre-project all toy functions at every resolution.

#     Returns a dict keyed by (function_name, n) with DOF vectors.
#     """
#     data = {}
#     for n in NS:
#         seq = seqs[n]
#         F = seq.map
#         data[("f0", n)] = _project(seq, 0, lambda a: scalar_f(a, F))
#         data[("g1", n)] = _project(seq, 1, lambda a: vector_g(a, F))
#         data[("h2", n)] = _project(seq, 2, lambda a: vector_h(a, F))
#     return data


# @pytest.fixture(scope="module")
# def derivatives(seqs, projected):
#     """Apply strong derivative operators to projected functions.

#     Returns a dict keyed by (operator_name, n) with DOF vectors.
#     """
#     data = {}
#     for n in NS:
#         seq = seqs[n]
#         data[("grad_f", n)] = seq.apply_strong_grad(
#             projected[("f0", n)], dirichlet_in=False, dirichlet_out=False)
#         data[("curl_g", n)] = seq.apply_strong_curl(
#             projected[("g1", n)], dirichlet_in=False, dirichlet_out=False)
#         data[("div_h", n)] = seq.apply_strong_div(
#             projected[("h2", n)], dirichlet_in=False, dirichlet_out=False)
#         data[("curl_grad_f", n)] = seq.apply_strong_curl(
#             data[("grad_f", n)], dirichlet_in=False, dirichlet_out=False)
#         data[("div_curl_g", n)] = seq.apply_strong_div(
#             data[("curl_g", n)], dirichlet_in=False, dirichlet_out=False)
#     return data


# # ---------------------------------------------------------------------------
# # Tests
# # ---------------------------------------------------------------------------

# def _assert_convergence(errors, label, threshold=0.5, strict=True):
#     assert errors[0] < threshold, (
#         f"{label} error too large at n={NS[0]}: {errors[0]:.2e}")
#     if strict:
#         for i in range(1, len(errors)):
#             assert errors[i] < errors[i-1], (
#                 f"{label} error did not decrease: {errors}")
#     else:
#         assert errors[-1] < errors[0], (
#             f"{label} error did not decrease overall: {errors}")


# class TestGradient:
#     """grad(f) where f = x² + y² + z², expected ∇f = (2x, 2y, 2z)."""

#     def test_gradient_convergence(self, seqs, derivatives):
#         errors = []
#         for n in NS:
#             seq = seqs[n]
#             F = seq.map
#             err = _l2_error(seq, 1, derivatives[("grad_f", n)],
#                             lambda a: grad_f(a, F))
#             errors.append(err)
#             print(f"  grad n={n}: error={err:.4e}")
#         _assert_convergence(errors, "grad(f)")


# class TestCurl:
#     """curl(G) where G = (y², z², x²), expected curl = (-2z, -2x, -2y)."""

#     def test_curl_convergence(self, seqs, derivatives):
#         errors = []
#         for n in NS:
#             seq = seqs[n]
#             F = seq.map
#             err = _l2_error(seq, 2, derivatives[("curl_g", n)],
#                             lambda a: curl_g(a, F))
#             errors.append(err)
#             print(f"  curl n={n}: error={err:.4e}")
#         _assert_convergence(errors, "curl(G)", threshold=0.6)


# class TestDivergence:
#     """div(H) where H = (x², y², z²), expected div = 2(x+y+z)."""

#     def test_divergence_convergence(self, seqs, derivatives):
#         errors = []
#         for n in NS:
#             seq = seqs[n]
#             F = seq.map
#             err = _l2_error(seq, 3, derivatives[("div_h", n)],
#                             lambda a: div_h(a, F))
#             errors.append(err)
#             print(f"  div n={n}: error={err:.4e}")
#         _assert_convergence(errors, "div(H)")


# class TestExactness:
#     """De Rham exactness: curl(grad(f)) = 0, div(curl(G)) = 0."""

#     def test_curl_grad_zero(self, derivatives):
#         for n in NS:
#             dofs = derivatives[("curl_grad_f", n)]
#             assert jnp.linalg.norm(dofs) < 1e-8, (
#                 f"curl(grad(f)) not zero at n={n}: "
#                 f"norm={float(jnp.linalg.norm(dofs)):.2e}")

#     def test_div_curl_zero(self, derivatives):
#         for n in NS:
#             dofs = derivatives[("div_curl_g", n)]
#             assert jnp.linalg.norm(dofs) < 1e-8, (
#                 f"div(curl(G)) not zero at n={n}: "
#                 f"norm={float(jnp.linalg.norm(dofs)):.2e}")


# # ===========================================================================
# # Cylinder domain — DBC tests
# # ===========================================================================

# def _cylinder_map(x):
#     """F(r,θ,ζ) = (r cos 2πθ, r sin 2πθ, ζ)."""
#     r, theta, zeta = x
#     return jnp.array([
#         r * jnp.cos(2 * jnp.pi * theta),
#         r * jnp.sin(2 * jnp.pi * theta),
#         zeta,
#     ])


# # -- DBC-compatible toy functions on the cylinder ---------------------------
# # All vanish (in the appropriate form sense) at r=0 and r=1.

# def scalar_f_dbc(a, F):
#     """f = r²(1−r²) = (x²+y²)(1−(x²+y²)).  Vanishes at r=0,1."""
#     x, y, z = F(a)
#     s = x**2 + y**2
#     return jnp.array([s * (1 - s)])


# def grad_f_dbc(a, F):
#     """∇f = (2x(1−2r²), 2y(1−2r²), 0)."""
#     x, y, z = F(a)
#     s = x**2 + y**2
#     return jnp.array([2*x*(1 - 2*s), 2*y*(1 - 2*s), 0.0])


# def vector_g_dbc(a, F):
#     """G = (0, 0, r²(1−r²)).  All components vanish at r=0,1."""
#     x, y, z = F(a)
#     s = x**2 + y**2
#     return jnp.array([0.0, 0.0, s * (1 - s)])


# def curl_g_dbc(a, F):
#     """curl G = (2y(1−2r²), −2x(1−2r²), 0)."""
#     x, y, z = F(a)
#     s = x**2 + y**2
#     return jnp.array([2*y*(1 - 2*s), -2*x*(1 - 2*s), 0.0])


# def vector_h_dbc(a, F):
#     """H = ((1−r²)x, (1−r²)y, 0).  Normal component vanishes at r=0,1."""
#     x, y, z = F(a)
#     s = x**2 + y**2
#     return jnp.array([(1 - s)*x, (1 - s)*y, 0.0])


# def div_h_dbc(a, F):
#     """div H = 2(1−2r²)."""
#     x, y, z = F(a)
#     s = x**2 + y**2
#     return jnp.array([2*(1 - 2*s)])


# # Neumann-compatible scalar (∂_r f = 0 at r=1, regular at r=0)
# def scalar_f_neumann(a, F):
#     """f = 1 + r² − r⁴/2 = 1 + (x²+y²) − (x²+y²)²/2."""
#     x, y, z = F(a)
#     s = x**2 + y**2
#     return jnp.array([1 + s - s**2 / 2])


# # Analytical −Δ for the Dirichlet scalar r²(1−r²)
# def neg_lap_dirichlet(a, F):
#     """−Δ[r²(1−r²)] = 16(x²+y²) − 4."""
#     x, y, z = F(a)
#     return jnp.array([16*(x**2 + y**2) - 4])


# # Analytical −Δ for the Neumann scalar 1 + r² − r⁴/2
# def neg_lap_neumann(a, F):
#     """−Δ[1+r²−r⁴/2] = 8(x²+y²) − 4."""
#     x, y, z = F(a)
#     return jnp.array([8*(x**2 + y**2) - 4])


# # -- k=1 / k=2 Hodge-Laplacian toy functions on the cylinder ----------------

# def vector_g1_neumann(a, F):
#     """(0, 0, 1+r²−r⁴/2) — 1-form in free space (no tangential BC)."""
#     x, y, z = F(a)
#     s = x**2 + y**2
#     return jnp.array([0.0, 0.0, 1 + s - s**2 / 2])


# def vector_h2_neumann(a, F):
#     """((1−r²/2)x, (1−r²/2)y, 0) — 2-form in free space, div=0 at r=1."""
#     x, y, z = F(a)
#     s = x**2 + y**2
#     return jnp.array([(1 - s / 2) * x, (1 - s / 2) * y, 0.0])


# def neg_lap_g_dbc(a, F):
#     """−Δ(0,0,r²(1−r²)) = (0, 0, 16r²−4)."""
#     x, y, z = F(a)
#     return jnp.array([0.0, 0.0, 16*(x**2 + y**2) - 4])


# def neg_lap_g_neumann(a, F):
#     """−Δ(0,0,1+r²−r⁴/2) = (0, 0, 8r²−4)."""
#     x, y, z = F(a)
#     return jnp.array([0.0, 0.0, 8*(x**2 + y**2) - 4])


# def neg_lap_h_dbc(a, F):
#     """−Δ((1−r²)x,(1−r²)y,0) = (8x, 8y, 0)."""
#     x, y, z = F(a)
#     return jnp.array([8*x, 8*y, 0.0])


# def neg_lap_h_neumann(a, F):
#     """−Δ((1−r²/2)x,(1−r²/2)y,0) = (4x, 4y, 0)."""
#     x, y, z = F(a)
#     return jnp.array([4*x, 4*y, 0.0])


# # -- Cylinder fixtures ------------------------------------------------------

# def _build_cylinder_seq(n, p=2):
#     seq = DeRhamSequence((n, n, n), (p, p, p), 2*p, types, _cylinder_map,
#                          polar=True, tol=1e-12, maxiter=1000)
#     seq.evaluate_1d()
#     for k in range(4):
#         seq.assemble_mass_matrix(k)
#     for k in range(3):
#         seq.assemble_derivative_matrix(k)
#     for k in range(3):
#         seq.assemble_hodge_laplacian(k)
#     return seq


# @pytest.fixture(scope="module")
# def cyl_seqs():
#     return {n: _build_cylinder_seq(n) for n in NS}


# @pytest.fixture(scope="module")
# def cyl_projected(cyl_seqs):
#     data = {}
#     for n in NS:
#         seq = cyl_seqs[n]
#         F = seq.map
#         data[("f0", n)] = _project(
#             seq, 0, lambda a: scalar_f_dbc(a, F), dirichlet=True)
#         data[("g1", n)] = _project(
#             seq, 1, lambda a: vector_g_dbc(a, F), dirichlet=True)
#         data[("h2", n)] = _project(
#             seq, 2, lambda a: vector_h_dbc(a, F), dirichlet=True)
#         # Neumann 0-form (free space)
#         data[("f0_neu", n)] = _project(
#             seq, 0, lambda a: scalar_f_neumann(a, F), dirichlet=False)
#         # Dirichlet 3-form: scalar_f_dbc as 3-form (free space)
#         data[("f3_dir", n)] = _project(
#             seq, 3, lambda a: scalar_f_dbc(a, F), dirichlet=False)
#         # Neumann 3-form: scalar_f_neumann as 3-form (DBC space)
#         data[("f3_neu", n)] = _project(
#             seq, 3, lambda a: scalar_f_neumann(a, F), dirichlet=True)
#         # k=1 Neumann: (0,0,1+r²−r⁴/2) in free 1-form space
#         data[("g1_neu", n)] = _project(
#             seq, 1, lambda a: vector_g1_neumann(a, F), dirichlet=False)
#         # k=2 Neumann: (xr²,yr²,0) in free 2-form space
#         data[("h2_neu", n)] = _project(
#             seq, 2, lambda a: vector_h2_neumann(a, F), dirichlet=False)
#     return data


# @pytest.fixture(scope="module")
# def cyl_derivatives(cyl_seqs, cyl_projected):
#     data = {}
#     for n in NS:
#         seq = cyl_seqs[n]
#         # grad: DBC 0-form → DBC 1-form
#         data[("grad_f", n)] = seq.apply_strong_grad(
#             cyl_projected[("f0", n)],
#             dirichlet_in=True, dirichlet_out=True)
#         # curl: DBC 1-form → DBC 2-form
#         data[("curl_g", n)] = seq.apply_strong_curl(
#             cyl_projected[("g1", n)],
#             dirichlet_in=True, dirichlet_out=True)
#         # div: DBC 2-form → free 3-form
#         data[("div_h", n)] = seq.apply_strong_div(
#             cyl_projected[("h2", n)],
#             dirichlet_in=True, dirichlet_out=False)
#         # exactness: curl(grad(f)), div(curl(g))
#         data[("curl_grad_f", n)] = seq.apply_strong_curl(
#             data[("grad_f", n)],
#             dirichlet_in=True, dirichlet_out=True)
#         data[("div_curl_g", n)] = seq.apply_strong_div(
#             data[("curl_g", n)],
#             dirichlet_in=True, dirichlet_out=False)
#     return data


# @pytest.fixture(scope="module")
# def cyl_laplacians(cyl_seqs, cyl_projected):
#     """Apply Hodge Laplacian (k=0..3), then M⁻¹ to get strong form."""
#     data = {}
#     for n in NS:
#         seq = cyl_seqs[n]
#         # k=0 Dirichlet: HL₀(f_D), M₀⁻¹
#         hl = seq.apply_hodge_laplacian(
#             cyl_projected[("f0", n)], 0, dirichlet=True)
#         data[("hl0_dir", n)] = seq.apply_inverse_mass_matrix(
#             hl, 0, dirichlet=True)
#         # k=0 Neumann: HL₀(f_N), M₀⁻¹
#         hl = seq.apply_hodge_laplacian(
#             cyl_projected[("f0_neu", n)], 0, dirichlet=False)
#         data[("hl0_neu", n)] = seq.apply_inverse_mass_matrix(
#             hl, 0, dirichlet=False)
#         # k=1 Dirichlet: HL₁(g_D), M₁⁻¹
#         hl = seq.apply_hodge_laplacian(
#             cyl_projected[("g1", n)], 1, dirichlet=True)
#         data[("hl1_dir", n)] = seq.apply_inverse_mass_matrix(
#             hl, 1, dirichlet=True)
#         # k=1 Neumann: HL₁(g_N), M₁⁻¹
#         hl = seq.apply_hodge_laplacian(
#             cyl_projected[("g1_neu", n)], 1, dirichlet=False)
#         data[("hl1_neu", n)] = seq.apply_inverse_mass_matrix(
#             hl, 1, dirichlet=False)
#         # k=2 Dirichlet: HL₂(h_D), M₂⁻¹
#         hl = seq.apply_hodge_laplacian(
#             cyl_projected[("h2", n)], 2, dirichlet=True)
#         data[("hl2_dir", n)] = seq.apply_inverse_mass_matrix(
#             hl, 2, dirichlet=True)
#         # k=2 Neumann: HL₂(h_N), M₂⁻¹
#         hl = seq.apply_hodge_laplacian(
#             cyl_projected[("h2_neu", n)], 2, dirichlet=False)
#         data[("hl2_neu", n)] = seq.apply_inverse_mass_matrix(
#             hl, 2, dirichlet=False)
#         # k=3 Dirichlet: HL₃(f_D as 3-form), M₃⁻¹
#         hl = seq.apply_hodge_laplacian(
#             cyl_projected[("f3_dir", n)], 3, dirichlet=False)
#         data[("hl3_dir", n)] = seq.apply_inverse_mass_matrix(
#             hl, 3, dirichlet=False)
#         # k=3 Neumann: HL₃(f_N as 3-form), M₃⁻¹
#         hl = seq.apply_hodge_laplacian(
#             cyl_projected[("f3_neu", n)], 3, dirichlet=True)
#         data[("hl3_neu", n)] = seq.apply_inverse_mass_matrix(
#             hl, 3, dirichlet=True)
#     return data


# # -- Cylinder tests ---------------------------------------------------------

# class TestGradientDBC:
#     """DBC grad: f = r²(1−r²), expected ∇f = (2x(1−2r²), 2y(1−2r²), 0)."""

#     def test_gradient_convergence(self, cyl_seqs, cyl_derivatives):
#         errors = []
#         for n in NS:
#             seq = cyl_seqs[n]
#             F = seq.map
#             err = _l2_error(seq, 1, cyl_derivatives[("grad_f", n)],
#                             lambda a: grad_f_dbc(a, F), dirichlet=True)
#             errors.append(err)
#             print(f"  grad DBC n={n}: error={err:.4e}")
#         _assert_convergence(errors, "grad(f) DBC", threshold=0.6)


# class TestCurlDBC:
#     """DBC curl: G=(0,0,r²(1−r²)), expected curl G."""

#     def test_curl_convergence(self, cyl_seqs, cyl_derivatives):
#         errors = []
#         for n in NS:
#             seq = cyl_seqs[n]
#             F = seq.map
#             err = _l2_error(seq, 2, cyl_derivatives[("curl_g", n)],
#                             lambda a: curl_g_dbc(a, F), dirichlet=True)
#             errors.append(err)
#             print(f"  curl DBC n={n}: error={err:.4e}")
#         _assert_convergence(errors, "curl(G) DBC", threshold=0.6)


# class TestDivergenceDBC:
#     """DBC div: H=((1−r²)x,(1−r²)y,0), expected div H = 2(1−2r²)."""

#     def test_divergence_convergence(self, cyl_seqs, cyl_derivatives):
#         errors = []
#         for n in NS:
#             seq = cyl_seqs[n]
#             F = seq.map
#             err = _l2_error(seq, 3, cyl_derivatives[("div_h", n)],
#                             lambda a: div_h_dbc(a, F), dirichlet=False)
#             errors.append(err)
#             print(f"  div DBC n={n}: error={err:.4e}")
#         _assert_convergence(errors, "div(H) DBC")


# class TestExactnessDBC:
#     """De Rham exactness with DBC: curl(grad(f)) = 0, div(curl(G)) = 0."""

#     def test_curl_grad_zero(self, cyl_derivatives):
#         for n in NS:
#             dofs = cyl_derivatives[("curl_grad_f", n)]
#             assert jnp.linalg.norm(dofs) < 1e-8, (
#                 f"curl(grad(f)) DBC not zero at n={n}: "
#                 f"norm={float(jnp.linalg.norm(dofs)):.2e}")

#     def test_div_curl_zero(self, cyl_derivatives):
#         for n in NS:
#             dofs = cyl_derivatives[("div_curl_g", n)]
#             assert jnp.linalg.norm(dofs) < 1e-8, (
#                 f"div(curl(G)) DBC not zero at n={n}: "
#                 f"norm={float(jnp.linalg.norm(dofs)):.2e}")


# # -- Hodge Laplacian tests (k=0, k=3) on cylinder --------------------------

# class TestHodgeLaplacianK0Dirichlet:
#     """HL₀ with Dirichlet BCs: f = r²(1−r²), −Δf = 16r² − 4."""

#     def test_convergence(self, cyl_seqs, cyl_laplacians):
#         errors = []
#         for n in NS:
#             seq = cyl_seqs[n]
#             F = seq.map
#             err = _l2_error(seq, 0, cyl_laplacians[("hl0_dir", n)],
#                             lambda a: neg_lap_dirichlet(a, F),
#                             dirichlet=True)
#             errors.append(err)
#             print(f"  HL₀ Dirichlet n={n}: error={err:.4e}")
#         _assert_convergence(errors, "HL₀ Dirichlet", threshold=0.8)


# class TestHodgeLaplacianK0Neumann:
#     """HL₀ with Neumann BCs: f = 1+r²−r⁴/2, −Δf = 8r² − 4."""

#     def test_convergence(self, cyl_seqs, cyl_laplacians):
#         errors = []
#         for n in NS:
#             seq = cyl_seqs[n]
#             F = seq.map
#             err = _l2_error(seq, 0, cyl_laplacians[("hl0_neu", n)],
#                             lambda a: neg_lap_neumann(a, F),
#                             dirichlet=False)
#             errors.append(err)
#             print(f"  HL₀ Neumann n={n}: error={err:.4e}")
#         _assert_convergence(errors, "HL₀ Neumann")


# class TestHodgeLaplacianK3Dirichlet:
#     """HL₃ with Dirichlet BCs: f = r²(1−r²) as 3-form, −Δf = 16r² − 4."""

#     def test_convergence(self, cyl_seqs, cyl_laplacians):
#         errors = []
#         for n in NS:
#             seq = cyl_seqs[n]
#             F = seq.map
#             err = _l2_error(seq, 3, cyl_laplacians[("hl3_dir", n)],
#                             lambda a: neg_lap_dirichlet(a, F),
#                             dirichlet=False)
#             errors.append(err)
#             print(f"  HL₃ Dirichlet n={n}: error={err:.4e}")
#         _assert_convergence(errors, "HL₃ Dirichlet")


# class TestHodgeLaplacianK3Neumann:
#     """HL₃ with Neumann BCs: f = 1+r²−r⁴/2 as 3-form, −Δf = 8r² − 4."""

#     def test_convergence(self, cyl_seqs, cyl_laplacians):
#         errors = []
#         for n in NS:
#             seq = cyl_seqs[n]
#             F = seq.map
#             err = _l2_error(seq, 3, cyl_laplacians[("hl3_neu", n)],
#                             lambda a: neg_lap_neumann(a, F),
#                             dirichlet=True)
#             errors.append(err)
#             print(f"  HL₃ Neumann n={n}: error={err:.4e}")
#         _assert_convergence(errors, "HL₃ Neumann")


# # -- Hodge Laplacian tests (k=1, k=2) on cylinder --------------------------

# class TestHodgeLaplacianK1Dirichlet:
#     """HL₁ DBC: G = (0,0,r²(1−r²)), −ΔG = (0,0,16r²−4)."""

#     def test_convergence(self, cyl_seqs, cyl_laplacians):
#         errors = []
#         for n in NS:
#             seq = cyl_seqs[n]
#             F = seq.map
#             err = _l2_error(seq, 1, cyl_laplacians[("hl1_dir", n)],
#                             lambda a: neg_lap_g_dbc(a, F),
#                             dirichlet=True)
#             errors.append(err)
#             print(f"  HL₁ Dirichlet n={n}: error={err:.4e}")
#         _assert_convergence(errors, "HL₁ Dirichlet", threshold=0.8)


# class TestHodgeLaplacianK1Neumann:
#     """HL₁ free: G = (0,0,1+r²−r⁴/2), −ΔG = (0,0,8r²−4)."""

#     def test_convergence(self, cyl_seqs, cyl_laplacians):
#         errors = []
#         for n in NS:
#             seq = cyl_seqs[n]
#             F = seq.map
#             err = _l2_error(seq, 1, cyl_laplacians[("hl1_neu", n)],
#                             lambda a: neg_lap_g_neumann(a, F),
#                             dirichlet=False)
#             errors.append(err)
#             print(f"  HL₁ Neumann n={n}: error={err:.4e}")
#         _assert_convergence(errors, "HL₁ Neumann")


# class TestHodgeLaplacianK2Dirichlet:
#     """HL₂ DBC: H = ((1−r²)x,(1−r²)y,0), −ΔH = (8x,8y,0)."""

#     def test_convergence(self, cyl_seqs, cyl_laplacians):
#         errors = []
#         for n in NS:
#             seq = cyl_seqs[n]
#             F = seq.map
#             err = _l2_error(seq, 2, cyl_laplacians[("hl2_dir", n)],
#                             lambda a: neg_lap_h_dbc(a, F),
#                             dirichlet=True)
#             errors.append(err)
#             print(f"  HL₂ Dirichlet n={n}: error={err:.4e}")
#         _assert_convergence(errors, "HL₂ Dirichlet", threshold=0.8)


# class TestHodgeLaplacianK2Neumann:
#     """HL₂ free: H = (xr²,yr²,0), −ΔH = (−8x,−8y,0)."""

#     def test_convergence(self, cyl_seqs, cyl_laplacians):
#         errors = []
#         for n in NS:
#             seq = cyl_seqs[n]
#             F = seq.map
#             err = _l2_error(seq, 2, cyl_laplacians[("hl2_neu", n)],
#                             lambda a: neg_lap_h_neumann(a, F),
#                             dirichlet=False)
#             errors.append(err)
#             print(f"  HL₂ Neumann n={n}: error={err:.4e}")
#         _assert_convergence(errors, "HL₂ Neumann", strict=False)
