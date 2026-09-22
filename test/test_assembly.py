"""Matrix-free assembly against a quadrature oracle, on the session geometry.

``mrx.mass`` applies every mass by sum factorisation with the metric weight
formed from ``DF`` inside the kernel. The oracle evaluates the same k-form at
the quadrature points through the basis tables, multiplies by the metric
weight of the space, and integrates back against the basis -- two tensor
contractions that share nothing with the fused kernel. One random vector per
degree: the identity is linear, so one vector tests the whole operator.
"""
import os
import subprocess
import sys
import textwrap

import jax.numpy as jnp
import numpy as np
import pytest

import mrx
from mrx.quadrature import integrate_against

# Roundoff identity relative to the size of the result: 1e3 eps
# (2.2e-13 f64 / 1.2e-4 f32).
IDENT = mrx.eps(1e3)


def _weight(seq, k):
    """The mass weight of k-forms at the quadrature points."""
    J = seq.jacobian_j
    if k == 0:
        return J
    if k == 3:
        return 1.0 / J
    if k == 1:
        return seq.metric_inv_jkl * J[:, None, None]
    return seq.metric_jkl / J[:, None, None]


def _oracle_mass(seq, x, k, dirichlet):
    """``M_k x`` by evaluate -> weight -> integrate, through the extraction."""
    u_q = seq.evaluate_at_quadrature(x, k, dirichlet)               # (n_q, d)
    w = _weight(seq, k)
    wu = u_q * w[:, None] if w.ndim == 1 else jnp.einsum('qij,qj->qi', w, u_q)
    comp_info, comp_shapes = seq._form_comp_info(k)
    raw = integrate_against(wu * seq.quad.w[:, None], comp_info, comp_shapes, seq.quad.shape)
    return seq.E(k, dirichlet) @ raw


@pytest.mark.parametrize("k", (0, 1, 2, 3))
def test_mass_apply_matches_quadrature_oracle(seq, k):
    dirichlet = True
    x = jnp.asarray(np.random.default_rng(k).standard_normal(seq.n(k, dirichlet)),
                    dtype=mrx.DTYPE)
    got = seq.apply_mass_matrix(x, k, dirichlet)
    want = _oracle_mass(seq, x, k, dirichlet)
    err = float(jnp.max(jnp.abs(got - want)) / jnp.max(jnp.abs(want)))
    assert err < IDENT, f"k={k}: kernel vs oracle off by {err:.2e}"


@pytest.mark.parametrize(("pair", "partner"), (((2, 1), (1, 2)), ((0, 3), (3, 0))))
def test_projection_pairs_are_transposes(seq, pair, partner):
    """``<P_12 x, y> = <x, P_21 y>`` on the extracted Dirichlet spaces."""
    k_in, k_out = pair
    rng = np.random.default_rng(7)
    x = jnp.asarray(rng.standard_normal(seq.n(k_in, True)), dtype=mrx.DTYPE)
    y = jnp.asarray(rng.standard_normal(seq.n(k_out, True)), dtype=mrx.DTYPE)
    lhs = float(y @ seq.apply_projection_matrix(x, *pair, True, dirichlet_out=True))
    rhs = float(x @ seq.apply_projection_matrix(y, *partner, True, dirichlet_out=True))
    assert abs(lhs - rhs) < IDENT * abs(lhs), f"{pair}: {lhs:.6e} vs {rhs:.6e}"


def test_shift_plan_accepts_both_axis_kinds_and_names_a_bad_one():
    """``_shift_plan`` is the kernel's one precondition, and it has no oracle.

    Everything else in ``mrx.mass`` is covered by the apply above. This is the
    gate that decides whether the indexless gather and assembly are usable at
    all, so the two shapes it must accept -- a periodic axis, which wraps, and
    a clamped one, which does not -- and the rejection are checked directly.
    Pure index arithmetic, no sequence and no solves.
    """
    from mrx.mass import _shift_plan

    ne, nloc = 4, 3
    e = np.arange(ne)[:, None]
    lo = np.arange(nloc)[None, :]

    S_per = ne                      # periodic: e + l wraps
    S_clamp = ne + nloc - 1         # clamped: e + l never reaches S
    g_per = (e + lo) % S_per
    g_clamp = e + lo

    plan = _shift_plan(g_clamp, g_per, g_per, (S_clamp, S_per, S_per))
    assert plan == ((ne, nloc, S_clamp), (ne, nloc, S_per), (ne, nloc, S_per))

    # Permuting two DoFs is still a bijection, so nothing but the shift itself
    # can catch it. The message has to name the axis, since the caller passes
    # three and a plan is rebuilt per component.
    g_bad = g_per.copy()
    g_bad[0, 0], g_bad[0, 1] = g_bad[0, 1], g_bad[0, 0]
    with pytest.raises(ValueError, match="axis y"):
        _shift_plan(g_clamp, g_bad, g_per, (S_clamp, S_per, S_per))


def test_indexed_assembly_matches_the_shift_form(seq):
    """The indexed gather and assembly agree with the shifted form.

    The gather is the same integer map, so it agrees exactly. The assembly
    is the same sum in a different order, so it agrees to float32 roundoff.
    The full kernel, both static branches of it, agrees on a real plan: this
    is what keeps ``MRX_ASSEMBLY=indexed`` from being a different operator.
    """
    from mrx.mass import (
        _indexed_accumulate,
        _indexed_gather,
        _structured_accumulate,
        _structured_gather,
        _sumfact_kernel,
    )

    for k in (0, 1, 2):
        plan = seq.mass_plan[k]
        seen = []
        for component in (*plan.gather_plans, *plan.shift_plans):
            if component in seen:
                continue
            seen.append(component)
            (_, _, sx), (_, _, sy), (_, _, sz) = component
            rng = np.random.default_rng(sx * sy + sz + k)
            x = jnp.asarray(rng.standard_normal(sx * sy * sz), dtype=mrx.DTYPE)
            gathered = _structured_gather(x, component)
            indexed = _indexed_gather(x, component)
            assert jnp.array_equal(gathered, indexed), f"k={k} gather differs"

            y = jnp.asarray(rng.standard_normal(gathered.shape), dtype=mrx.DTYPE)
            shifted = _structured_accumulate(y, component)
            summed = _indexed_accumulate(y, component)
            err = float(jnp.max(jnp.abs(shifted - summed))
                        / jnp.max(jnp.abs(shifted)))
            assert err < IDENT, f"k={k} assembly off by {err:.2e}"

        n = int(seq.E(k, False).forward_shape[1])
        x = jnp.asarray(np.random.default_rng(50 + k).standard_normal(n),
                        dtype=mrx.DTYPE)
        weights = seq.geometry.mass_weights[k]
        common = dict(pairs=plan.pairs, cols=plan.cols, starts_c=plan.starts_c,
                      shift_plans=plan.shift_plans, gather_plans=plan.gather_plans)
        shift = _sumfact_kernel(x, plan.Bvals_r, plan.Bvals_c, weights,
                                assembly="shift", **common)
        indexed = _sumfact_kernel(x, plan.Bvals_r, plan.Bvals_c, weights,
                                  assembly="indexed", **common)
        err = float(jnp.max(jnp.abs(shift - indexed)) / jnp.max(jnp.abs(shift)))
        assert err < IDENT, f"k={k} kernel off by {err:.2e}"


def test_assembly_mode_follows_the_backend_and_rejects_garbage(monkeypatch):
    """Unset, Metal gets the indexed form and every other backend the shifts.

    The override is read at the call, so a bad value fails there rather than
    at import, and a test can check both without a subprocess.
    """
    import jax

    from mrx.mass import _assembly_mode

    monkeypatch.delenv("MRX_ASSEMBLY", raising=False)
    expect = "indexed" if jax.default_backend() == "mps" else "shift"
    assert _assembly_mode() == expect

    monkeypatch.setenv("MRX_ASSEMBLY", "shift")
    assert _assembly_mode() == "shift"
    monkeypatch.setenv("MRX_ASSEMBLY", "indexed")
    assert _assembly_mode() == "indexed"
    monkeypatch.setenv("MRX_ASSEMBLY", "tf32")
    with pytest.raises(ValueError, match="MRX_ASSEMBLY"):
        _assembly_mode()

    # The Newton matvec pins the shift assembly for its own trace, whatever
    # the backend default is. The pin has to win over the environment too.
    from mrx.mass import _assembly_override
    monkeypatch.setenv("MRX_ASSEMBLY", "indexed")
    token = _assembly_override.set("shift")
    try:
        assert _assembly_mode() == "shift"
    finally:
        _assembly_override.reset(token)
    assert _assembly_mode() == "indexed"


def _assembly_mode_under(**env: str) -> tuple[str, str]:
    """``(backend, mode)`` from a fresh interpreter with ``env`` set.

    ``JAX_PLATFORMS`` is read when the backend initializes, so the parent
    process cannot change it, and a shell that exported ``MRX_ASSEMBLY``
    must not leak into the child.
    """
    script = textwrap.dedent("""
        import jax
        from mrx.mass import _assembly_mode
        print(jax.default_backend(), _assembly_mode())
    """)
    child = {k: v for k, v in os.environ.items()
             if k not in ("MRX_ASSEMBLY", "JAX_PLATFORMS")}
    child.update(env)
    done = subprocess.run([sys.executable, "-c", script],
                          capture_output=True, text=True, env=child)
    assert done.returncode == 0, done.stderr
    backend, mode = done.stdout.split()
    return backend, mode


def test_indexed_assembly_is_selected_only_on_metal():
    """The indexed kernel runs only when Metal is the initialized backend.

    ``JAX_PLATFORMS=cpu`` in an environment where jax-mps is installed still
    selects the shift assembly: installing the plugin must not change a CPU
    run. ``MRX_ASSEMBLY`` is a separate override, checked on the CPU so it
    cannot be mistaken for detection. The Metal case is skipped where the
    plugin is not what this process initialized.
    """
    import jax

    assert _assembly_mode_under(JAX_PLATFORMS="cpu") == ("cpu", "shift")
    assert _assembly_mode_under(JAX_PLATFORMS="cpu", MRX_ASSEMBLY="indexed") == (
        "cpu", "indexed")
    if jax.default_backend() != "mps":
        pytest.skip("this process did not initialize Metal")
    assert _assembly_mode_under(JAX_PLATFORMS="mps") == ("mps", "indexed")
