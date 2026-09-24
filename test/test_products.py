"""The quadratic loads against the mass and projection matrices, and against each other.

Every product load integrates a pointwise product against a basis, so
``a^T load(b, c)`` is the trilinear integral ``int a b c dV`` (with the
right products) and must not depend on which factor is the test function:
that ties the dot, scalar and scalar-times-vector loads of every degree
combination to each other without a solve. The constant 1 anchors them: as
a 0-form it is the L2 projection of 1 onto the natural 0-forms, as a
3-form the L2 projection of that through ``scalar_product_load_values`` and the
3-form mass; a product with either is a mass matrix or the ``P_12``
pairing, and the dot product integrated against either is the inner
product or the pairing. The cross-product loads are checked for
antisymmetry across the (m, k) pairs. All on li383 with ``J`` the weak curl
of the session field ``b0``.

On the half-period sequence every field lives on its parity view (``J``,
``B`` odd; ``1``, ``g = |B|^2``, ``rho``, ``tau`` even) and a product is
formed from the factors' quadrature values on the view of the product --
the ``_values`` loads; a product of two odd fields is even.
"""
import numpy as np
import pytest

from mrx.precision import eps


def close(x, y, seq, what, tol=0.0):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    scale = np.abs(y).max()
    assert scale > 0, what
    err = np.abs(x - y).max() / scale
    # The loads are assembled in the working dtype: 10 eps measured in
    # float32 (2026-09-05), on top of the solve tolerance of a product that
    # involves a solve.
    assert err < 1e2 * seq.tol + eps(1e2) + tol, f"{what}: relative error {err:.2e}"


@pytest.fixture(scope="module")
def fields(seq, b0):
    odd, even = seq.odd, seq.even
    J = odd.apply_weak_curl(b0, dirichlet=True)
    one = even.apply_inverse_mass_matrix(even.load(lambda x: 1.0, 0, dirichlet=False), 0, dirichlet=False)
    one_q = even.evaluate_at_quadrature(one, 0, False)
    one3 = even.apply_inverse_mass_matrix(
        even.scalar_product_load_values(one_q, one_q, 3, 0, 0, False), 3, dirichlet=False)
    # The 0-forms contain the constant exactly (partition of unity); the
    # 3-forms do not (the physical 1 is the density J), so one3 is 1 only to
    # the projection error, which Galerkin orthogonality measures as
    # ||1 - one3||^2 = int 1 dV - int one3^2 dV: two mass matrices, no product load.
    vol = one @ even.apply_mass_matrix(one, 0, False)
    err3 = np.sqrt(abs(vol - one3 @ even.apply_mass_matrix(one3, 3, False)) / vol)
    B_q = odd.evaluate_at_quadrature(b0, 2, True)
    g = even.apply_inverse_mass_matrix(even.dot_product_load_values(B_q, B_q, 0, 2, 2, False), 0, dirichlet=False)
    g_q = even.evaluate_at_quadrature(g, 0, False)
    # Two 3-forms of no particular meaning; any DoF vector is one.
    rho = even.scalar_product_load_values(g_q, g_q, 3, 0, 0, False)
    tau = even.scalar_product_load_values(one_q, g_q, 3, 0, 0, False)
    print(f"\n  constant 1 on the 3-forms: projection error {err3:.2e}")
    q = dict(J=odd.evaluate_at_quadrature(J, 1, True), B=B_q, one=one_q, g=g_q,
             one3=even.evaluate_at_quadrature(one3, 3, False), rho=even.evaluate_at_quadrature(rho, 3, False),
             tau=even.evaluate_at_quadrature(tau, 3, False))
    return dict(J=J, B=b0, one=one, one3=one3, err3=float(err3), g=g, rho=rho, tau=tau, q=q)


def test_cross_product_loads_are_antisymmetric(seq, fields):
    even, q = seq.even, fields["q"]
    for n in (1, 2):
        JxB = even.cross_product_load_values(q["J"], q["B"], n, 1, 2, True)
        BxJ = even.cross_product_load_values(q["B"], q["J"], n, 2, 1, True)
        close(JxB, -BxJ, seq, f"cross n={n}")


def test_products_with_one_are_the_mass_and_projection_matrices(seq, fields):
    odd, even, q = seq.odd, seq.even, fields["q"]
    J, B, one, one3, g, rho = (fields[k] for k in ("J", "B", "one", "one3", "g", "rho"))
    P21 = odd.apply_projection_matrix(B, 2, 1, True, True)
    P12 = odd.apply_projection_matrix(J, 1, 2, True, True)
    MJ, MB = odd.apply_mass_matrix(J, 1, True), odd.apply_mass_matrix(B, 2, True)
    # As a 3-form the constant carries its projection error (a wrong 3-form
    # convention would show as an O(1) error, the density J in place of 1).
    for m, unit, unit_q, tol in ((0, one, q["one"], 0.0), (3, one3, q["one3"], 10 * fields["err3"])):
        close(unit @ even.dot_product_load_values(q["J"], q["B"], m, 1, 2, False), J @ P21, seq, f"1 . (J . B) as a {m}-form", tol)
        close(unit @ even.dot_product_load_values(q["J"], q["J"], m, 1, 1, False), J @ MJ, seq, f"1 . (J . J) as a {m}-form", tol)
        close(unit @ even.dot_product_load_values(q["B"], q["B"], m, 2, 2, False), B @ MB, seq, f"1 . (B . B) as a {m}-form", tol)
        close(odd.scalar_vector_load_values(unit_q, q["J"], 1, m, 1, True), MJ, seq, f"1 J onto 1, 1 as a {m}-form", tol)
        close(odd.scalar_vector_load_values(unit_q, q["B"], 2, m, 2, True), MB, seq, f"1 B onto 2, 1 as a {m}-form", tol)
        close(odd.scalar_vector_load_values(unit_q, q["B"], 1, m, 2, True), P21, seq, f"1 B onto 1, 1 as a {m}-form", tol)
        close(odd.scalar_vector_load_values(unit_q, q["J"], 2, m, 1, True), P12, seq, f"1 J onto 2, 1 as a {m}-form", tol)
        close(even.scalar_product_load_values(unit_q, q["g"], 0, m, 0, False), even.apply_mass_matrix(g, 0, False), seq, f"1 g onto 0, 1 as a {m}-form", tol)
        close(even.scalar_product_load_values(unit_q, q["rho"], 3, m, 3, False), even.apply_mass_matrix(rho, 3, False), seq, f"1 rho onto 3, 1 as a {m}-form", tol)
    close(one @ even.dot_product_load_values(q["B"], q["B"], 0, 2, 2, False), B @ MB, seq, "|B|^2")
    # The scalar pairing P_03 / P_30 (the 0-form 1 is exact, so these are exact).
    close(even.scalar_product_load_values(q["one"], q["g"], 3, 0, 0, False),
          even.apply_projection_matrix(g, 0, 3, False, False), seq, "1 g onto 3 is P_03 g")
    close(even.scalar_product_load_values(q["one"], q["rho"], 0, 0, 3, False),
          even.apply_projection_matrix(rho, 3, 0, False, False), seq, "1 rho onto 0 is P_30 rho")


def test_trilinear_forms_do_not_depend_on_the_test_factor(seq, fields):
    """``a^T load(b, c)`` over the three rotations of ``(a, b, c)``, every
    degree combination once: the vector triple (scalar, vector, vector)
    through ``dot_product_load_values`` and ``scalar_vector_load_values``, the scalar
    triples through ``scalar_product_load_values``. The parity of the product picks
    the view it is assembled on."""
    odd, even, q = seq.odd, seq.even, fields["q"]
    J, B, g, rho, tau = (fields[k] for k in ("J", "B", "g", "rho", "tau"))
    vec = {1: (J, q["J"], True), 2: (B, q["B"], True)}          # odd
    sca = {0: (g, q["g"], False), 3: (rho, q["rho"], False)}   # even
    for kf, (f, f_q, df) in sca.items():
        for kv, (v, v_q, dv) in vec.items():
            for kw, (w, w_q, dw) in vec.items():
                a = f @ even.dot_product_load_values(v_q, w_q, kf, kv, kw, df)       # odd . odd: even
                b = v @ odd.scalar_vector_load_values(f_q, w_q, kv, kf, kw, dv)      # even * odd: odd
                c = w @ odd.scalar_vector_load_values(f_q, v_q, kw, kf, kv, dw)
                close([b, c], [a, a], seq, f"int f (v . w): degrees ({kf}, {kv}, {kw})")
    triples = [((0, g, q["g"]), (0, g, q["g"]), (3, rho, q["rho"])), ((3, rho, q["rho"]), (3, tau, q["tau"]), (0, g, q["g"])),
               ((3, rho, q["rho"]), (3, tau, q["tau"]), (3, rho, q["rho"]))]
    for (ka, a, a_q), (kb, b, b_q), (kc, c, c_q) in triples:
        x = a @ even.scalar_product_load_values(b_q, c_q, ka, kb, kc, False)
        y = b @ even.scalar_product_load_values(c_q, a_q, kb, kc, ka, False)
        z = c @ even.scalar_product_load_values(a_q, b_q, kc, ka, kb, False)
        close([y, z], [x, x], seq, f"int a b c: degrees ({ka}, {kb}, {kc})")
