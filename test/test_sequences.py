# test_derham_sequence.py
import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest
from mrx.derham_sequence import DeRhamSequence
from mrx.mappings import rotating_ellipse_map
jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("p", [1, 2, 3])
@pytest.mark.parametrize("dirichlet, expected_nulls", [
    (True,  [False, False, True, True]),   # clamped
    (False, [True,  True,  False, False])  # free
])
def test_derham_sequence_eigen_and_exactness(p, dirichlet, expected_nulls):
    """
    Check dd operator spectra and exactness identities for rotating ellipse map.
    
    Parameters
    ----------
    p : int
        The degree of the spline in each direction.
    dirichlet : bool
        Whether the problem has dirichlet boundary conditions.
    expected_nulls : list
        The expected nullspace pattern of the dd operator.
    """
    Seq = DeRhamSequence(
        (4, 4, 4),
        (p, p, p),
        2*p,
        ("clamped", "periodic", "periodic"),
        rotating_ellipse_map(),
        polar=True,
        dirichlet=dirichlet
    )

    Seq.evaluate_1d()
    Seq.assemble_all()

    eigs = [
        jnp.linalg.eigvalsh(Seq.M0 @ Seq.dd0),
        jnp.linalg.eigvalsh(Seq.M1 @ Seq.dd1),
        jnp.linalg.eigvalsh(Seq.M2 @ Seq.dd2),
        jnp.linalg.eigvalsh(Seq.M3 @ Seq.dd3),
    ]

    # Check expected nullspace pattern and positive definiteness
    for i, (vals, should_be_zero) in enumerate(zip(eigs, expected_nulls)):
        # --- all eigenvalues should be nonnegative ---
        min_eig = jnp.min(vals)
        assert min_eig > -1e-10, (
            f"p={p} dirichlet={dirichlet} → dd{i} has negative eigenvalue {min_eig}"
        )

        # --- check smallest eigenvalue matches expected nullspace pattern ---
        λ0 = float(vals[0])
        if should_be_zero:
            assert abs(λ0) < 1e-10, (
                f"p={p} dirichlet={dirichlet} → dd{i} should have zero eigenvalue (got {λ0})"
            )
        else:
            assert abs(λ0) > 1e-6, (
                f"p={p} dirichlet={dirichlet} → dd{i} should NOT have zero eigenvalue (got {λ0})"
            )

    # Check exactness identities
    curl_grad = jnp.max(jnp.abs(Seq.strong_curl @ Seq.strong_grad))
    div_curl = jnp.max(jnp.abs(Seq.strong_div @ Seq.strong_curl))

    npt.assert_allclose(curl_grad, 0.0, atol=1e-12, err_msg=f"curl∘grad ≠ 0 for p={p}")
    npt.assert_allclose(div_curl, 0.0, atol=1e-12, err_msg=f"div∘curl ≠ 0 for p={p}")
