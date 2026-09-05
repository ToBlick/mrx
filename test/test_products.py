"""The quadratic loads against the mass and projection matrices, and against each other.

Every product load integrates a pointwise product against a basis, so
``a^T load(b, c)`` is the trilinear integral ``int a b c dV`` (with the
right products) and must not depend on which factor is the test function:
that ties the dot, scalar and scalar-times-vector loads of every degree
combination to each other without a solve. The constant 1 anchors them: as
a 0-form it is the L2 projection of 1 onto the natural 0-forms, as a
3-form the L2 projection of that through ``scalar_product_load`` and the
3-form mass; a product with either is a mass matrix or the ``P_12``
pairing, and the dot product integrated against either is the inner
product or the pairing. The cross-product loads are checked for
antisymmetry across the (m, k) pairs. All on li383 with ``J`` the weak curl
of the session field ``b0``.
"""
import numpy as np
import pytest

from mrx.precision import eps


def close(x, y, seq, what, tol=0.0):
    # The loads are assembled in the working precision, so the comparison
    # cannot be resolved below roundoff however tight seq.tol is: at float32
    # refinement takes seq.tol to 1e-8 while the 0-form unit load lands 23 eps
    # off its mass matrix. The band is whichever floor is looser, and the
    # roundoff one is 2e-14 at float64 and inert.
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    scale = np.abs(y).max()
    assert scale > 0, what
    err = np.abs(x - y).max() / scale
    assert err < max(1e2 * seq.tol, eps(1e2)) + tol, f"{what}: relative error {err:.2e}"


@pytest.fixture(scope="module")
def fields(seq, b0):
    J = seq.apply_weak_curl(b0, dirichlet=True)
    one = seq.apply_inverse_mass_matrix(seq.load(lambda x: 1.0, 0, dirichlet=False), 0, dirichlet=False)
    one3 = seq.apply_inverse_mass_matrix(
        seq.scalar_product_load(one, one, 3, 0, 0, False, False, False), 3, dirichlet=False)
    # The 0-forms contain the constant exactly (partition of unity); the
    # 3-forms do not (the physical 1 is the density J), so one3 is 1 only to
    # the projection error, which Galerkin orthogonality measures as
    # ||1 - one3||^2 = int 1 dV - int one3^2 dV: two mass matrices, no product load.
    vol = one @ seq.apply_mass_matrix(one, 0, False)
    err3 = np.sqrt(abs(vol - one3 @ seq.apply_mass_matrix(one3, 3, False)) / vol)
    g = seq.apply_inverse_mass_matrix(seq.magnitude_squared_load(b0), 0, dirichlet=False)
    # Two 3-forms of no particular meaning; any DoF vector is one.
    rho = seq.scalar_product_load(g, g, 3, 0, 0, False, False, False)
    tau = seq.scalar_product_load(one, g, 3, 0, 0, False, False, False)
    print(f"\n  constant 1 on the 3-forms: projection error {err3:.2e}")
    return dict(J=J, B=b0, one=one, one3=one3, err3=float(err3), g=g, rho=rho, tau=tau)


def test_cross_product_loads_are_antisymmetric(seq, fields):
    J, B = fields["J"], fields["B"]
    for n in (1, 2):
        JxB = seq.cross_product_load(J, B, n, 1, 2, True, True, True)
        BxJ = seq.cross_product_load(B, J, n, 2, 1, True, True, True)
        close(JxB, -BxJ, seq, f"cross n={n}")


def test_products_with_one_are_the_mass_and_projection_matrices(seq, fields):
    J, B, one, one3, g, rho = (fields[k] for k in ("J", "B", "one", "one3", "g", "rho"))
    M = seq.apply_mass_matrix
    P21 = seq.apply_projection_matrix(B, 2, 1, True, True)
    P12 = seq.apply_projection_matrix(J, 1, 2, True, True)
    # As a 3-form the constant carries its projection error (a wrong 3-form
    # convention would show as an O(1) error, the density J in place of 1).
    for m, unit, tol in ((0, one, 0.0), (3, one3, 10 * fields["err3"])):
        close(unit @ seq.dot_product_load(J, B, m, 1, 2, False), J @ P21, seq, f"1 . (J . B) as a {m}-form", tol)
        close(unit @ seq.dot_product_load(J, J, m, 1, 1, False), J @ M(J, 1, True), seq, f"1 . (J . J) as a {m}-form", tol)
        close(unit @ seq.dot_product_load(B, B, m, 2, 2, False), B @ M(B, 2, True), seq, f"1 . (B . B) as a {m}-form", tol)
        close(seq.scalar_vector_load(unit, J, 1, m, 1, True, False, True), M(J, 1, True), seq, f"1 J onto 1, 1 as a {m}-form", tol)
        close(seq.scalar_vector_load(unit, B, 2, m, 2, True, False, True), M(B, 2, True), seq, f"1 B onto 2, 1 as a {m}-form", tol)
        close(seq.scalar_vector_load(unit, B, 1, m, 2, True, False, True), P21, seq, f"1 B onto 1, 1 as a {m}-form", tol)
        close(seq.scalar_vector_load(unit, J, 2, m, 1, True, False, True), P12, seq, f"1 J onto 2, 1 as a {m}-form", tol)
        close(seq.scalar_product_load(unit, g, 0, m, 0, False, False, False), M(g, 0, False), seq, f"1 g onto 0, 1 as a {m}-form", tol)
        close(seq.scalar_product_load(unit, rho, 3, m, 3, False, False, False), M(rho, 3, False), seq, f"1 rho onto 3, 1 as a {m}-form", tol)
    close(one @ seq.magnitude_squared_load(B), B @ M(B, 2, True), seq, "|B|^2")
    # The scalar pairing P_03 / P_30 (the 0-form 1 is exact, so these are exact).
    close(seq.scalar_product_load(one, g, 3, 0, 0, False, False, False),
          seq.apply_projection_matrix(g, 0, 3, False, False), seq, "1 g onto 3 is P_03 g")
    close(seq.scalar_product_load(one, rho, 0, 0, 3, False, False, False),
          seq.apply_projection_matrix(rho, 3, 0, False, False), seq, "1 rho onto 0 is P_30 rho")


def test_trilinear_forms_do_not_depend_on_the_test_factor(seq, fields):
    """``a^T load(b, c)`` over the three rotations of ``(a, b, c)``, every
    degree combination once: the vector triple (scalar, vector, vector)
    through ``dot_product_load`` and ``scalar_vector_load``, the scalar
    triples through ``scalar_product_load``."""
    J, B, g, rho, tau = (fields[k] for k in ("J", "B", "g", "rho", "tau"))
    vec = {1: (J, True), 2: (B, True)}
    sca = {0: (g, False), 3: (rho, False)}
    for kf, (f, df) in sca.items():
        for kv, (v, dv) in vec.items():
            for kw, (w, dw) in vec.items():
                a = f @ seq.dot_product_load(v, w, kf, kv, kw, df, dv, dw)
                b = v @ seq.scalar_vector_load(f, w, kv, kf, kw, dv, df, dw)
                c = w @ seq.scalar_vector_load(f, v, kw, kf, kv, dw, df, dv)
                close([b, c], [a, a], seq, f"int f (v . w): degrees ({kf}, {kv}, {kw})")
    triples = [((0, g), (0, g), (3, rho)), ((3, rho), (3, tau), (0, g)), ((3, rho), (3, tau), (3, rho))]
    for (ka, a), (kb, b), (kc, c) in triples:
        x = a @ seq.scalar_product_load(b, c, ka, kb, kc, False, False, False)
        y = b @ seq.scalar_product_load(c, a, kb, kc, ka, False, False, False)
        z = c @ seq.scalar_product_load(a, b, kc, ka, kb, False, False, False)
        close([y, z], [x, x], seq, f"int a b c: degrees ({ka}, {kb}, {kc})")
