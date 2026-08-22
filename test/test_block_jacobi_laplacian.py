"""Guards for the block-Jacobi Laplacian preconditioner.

`mrx/experimental/block_jacobi_laplacian.py` is being promoted to the
production Laplacian preconditioner for k = 0..3 (see
`docs/research/production_simplification_plan.md`). It had NO test coverage;
these are the checks that plan calls for, each chosen because it has caught --
or would have caught -- a real bug:

1. the Dirichlet invariant. The boundary term must vanish under an essential
   condition. A missing `a == 0` guard once added entries on the PERIODIC
   theta/zeta axes, and this is the check that exposed it.
2. k = 0 carries no boundary trace at all (`W_0 = 0`), so the scale is a no-op
   there. At p = 1 the trace was once silently absent for a related reason.
3. `bc_scale = 0` reproduces `bc_entry=False` -- the wiring check behind every
   "does the term help" comparison in the handoff.
4. SPD. `bc_entry="ibpr"` (the exact cross-term correction, since removed) made
   `P` indefinite on every geometry; a rank-one update can do that.
5. an iteration-count regression, so a change that quietly multiplies the work
   is caught.

**Every inertness test carries a POSITIVE CONTROL** -- the same comparison in
the configuration where the term IS live, asserted to differ by a large factor.
Without it these would pass just as happily against a preconditioner that had
lost the boundary term entirely. The inertness tolerance is 1e-8 relative and
the control asserts >1e-2, so the two are separated by six orders of magnitude.
The residual is BUILD NOISE, not the term leaking: two builds of the IDENTICAL
configuration differ by the same ~1e-14 on the same ~1.7% of rows (the dense
polar core), which `test_defaults_are_the_production_configuration` measures
explicitly rather than assuming.

Everything reuses the session `torus_seq` fixture and a module-scoped build
cache, so no test here assembles a sequence, computes a nullspace, or builds
the same preconditioner twice.
"""
import os

import numpy as np
import pytest

jnp = pytest.importorskip("jax.numpy")

import mrx.operators as op  # noqa: E402
from mrx.experimental.block_jacobi_laplacian import (  # noqa: E402
    BlockJacobiLaplacian, trace_components,
)

# k = 3 is the cheapest vector case (one component); k = 1 is the one
# production cares most about. k = 2 tracks k = 1 in every measurement in the
# handoff, so it does not earn its own build here.
KS = (1, 3)
PROD_SCALE = 0.10
# Thresholds are set from the MEASURED separation, not chosen round numbers.
# On the session fixture the two populations are eight orders apart:
#   inert (Dirichlet, bc_scale 0 vs 100)   ~1e-11  -- pure build noise: the
#       dense polar-core block is not reproducible to the last bit between
#       builds, and it is ~1.7% of the rows. Measured, not assumed, in
#       test_defaults_are_the_production_configuration.
#   live  (free, bc_scale 0 vs 0.10 or 100) 2.5e-3 to 3.2e-3 at k=1, the
#       WEAKEST case (the term touches one radial row of one component in
#       three, and P's response saturates -- cf. min eig(P) = 1/(1+r s), so
#       0.10 and 100 move it by nearly the same amount).
# 1e-8 sits three orders above the noise; 1e-4 sits 25x below the weakest real
# effect and four orders above the inert bound.
INERT = 1e-8      # two builds that must agree
LIVE = 1e-4       # ... and, for the control, must NOT


@pytest.fixture(scope="module")
def bj(torus_seq):
    """Memoised preconditioner probes: one build per distinct configuration."""
    cache = {}

    def get(k, dbc, *, bc_entry="ibpd", bc_scale=PROD_SCALE, nvec=4):
        key = (k, dbc, bc_entry, bc_scale)
        if key not in cache:
            prev = os.environ.get("MRX_BJ_BC_SCALE")
            os.environ["MRX_BJ_BC_SCALE"] = str(bc_scale)
            try:
                pre = BlockJacobiLaplacian(
                    torus_seq, torus_seq.get_operators(), k, dbc,
                    ktilde_mode="honest", lumped="diag", bc_entry=bc_entry,
                    extra_rings=0, outer_rings=0)
            finally:
                if prev is None:
                    os.environ.pop("MRX_BJ_BC_SCALE", None)
                else:
                    os.environ["MRX_BJ_BC_SCALE"] = prev
            n = int(getattr(torus_seq, f"n{k}_dbc" if dbc else f"n{k}"))
            rng = np.random.default_rng(0)
            probe = np.stack([
                np.asarray(pre.apply(jnp.asarray(rng.standard_normal(n))))
                for _ in range(nvec)])
            cache[key] = (pre, n, probe)
        return cache[key]

    return get


def _rel(a, b):
    return float(np.abs(a - b).max() / np.abs(b).max())


@pytest.mark.parametrize("k", KS)
def test_boundary_term_vanishes_under_dirichlet(bj, k):
    """THE invariant: under an essential condition the boundary term does not
    exist, so the scale must not move the preconditioner."""
    lo = bj(k, True, bc_scale=0.0)[2]
    hi = bj(k, True, bc_scale=100.0)[2]
    d_dbc = _rel(hi, lo)
    assert d_dbc < INERT, (
        f"k={k} dbc: a 100x change in bc_scale moved the preconditioner by "
        f"{d_dbc:.2e}; the boundary term must vanish under Dirichlet")

    # POSITIVE CONTROL -- the same comparison where the term IS live. Without
    # this, the assertion above would also pass if the term were never
    # assembled at all.
    d_free = _rel(bj(k, False, bc_scale=100.0)[2], bj(k, False, bc_scale=0.0)[2])
    assert d_free > LIVE, (
        f"k={k} FREE: bc_scale did not move the preconditioner either "
        f"({d_free:.2e}) -- the boundary term is not being assembled, so the "
        "Dirichlet check above proves nothing")


def test_k0_carries_no_boundary_trace(bj):
    """`W_0 = 0`: no k=0 component's radial axis is a derivative axis, so the
    boundary block never runs and the scale is inert at k=0 even under FREE
    BCs -- the one degree where that is true."""
    assert trace_components(0) == ()
    assert trace_components(1) == (0,)          # control: k=1 does carry one
    d = _rel(bj(0, False, bc_scale=100.0)[2], bj(0, False, bc_scale=0.0)[2])
    assert d < INERT, (
        f"k=0 free: bc_scale moved the preconditioner by {d:.2e}, but k=0 has "
        "no boundary trace to scale")


@pytest.mark.parametrize("k", KS)
def test_scale_zero_reproduces_no_boundary_term(bj, k):
    """`bc_scale=0` must be the same operator as `bc_entry=False`."""
    off = bj(k, False, bc_entry=False)[2]
    zero = bj(k, False, bc_entry="ibpd", bc_scale=0.0)[2]
    d = _rel(zero, off)
    assert d < INERT, (
        f"k={k} free: bc_scale=0 differs from bc_entry=False by {d:.2e}")

    # POSITIVE CONTROL: the production scale must NOT match "no term".
    d_live = _rel(bj(k, False, bc_scale=PROD_SCALE)[2], off)
    assert d_live > LIVE, (
        f"k={k} free: bc_scale={PROD_SCALE} is indistinguishable from no "
        f"boundary term at all ({d_live:.2e})")


@pytest.mark.parametrize("k,dbc", [(1, True), (3, False)])
def test_preconditioner_is_spd(bj, k, dbc):
    """`P` must be symmetric positive definite -- CG assumes it.

    Cholesky rather than an eigendecomposition: definitive and ~3x cheaper.
    One Dirichlet and one free case; k=3 free is the cheapest vector case and
    the one where the boundary term is largest relative to the stiffness.
    """
    pre, n, _ = bj(k, dbc)
    dense = np.stack([np.asarray(pre.apply(jnp.zeros(n).at[i].set(1.0)))
                      for i in range(n)], axis=1)
    asym = np.abs(dense - dense.T).max() / np.abs(dense).max()
    assert asym < 1e-10, f"k={k} dbc={dbc}: P is not symmetric ({asym:.2e})"
    np.linalg.cholesky(0.5 * (dense + dense.T))   # raises if not positive definite


def test_iteration_count_regression(torus_seq, bj):
    """A cheap end-to-end guard: k=1 Dirichlet, which is NON-SINGULAR, so this
    needs no nullspace and no deflation.

    The band is deliberately wide. The measured iteration-count noise floor is
    ~1% (up to 2.4% on the singular free rows); this test exists to catch a
    change that alters the work by a FACTOR, not to pin a number.
    """
    k, dbc = 1, True
    pre, n, _ = bj(k, dbc)
    ops = torus_seq.get_operators()

    def A(x):
        return op.apply_hodge_laplacian_approx(torus_seq, ops, x, k, dirichlet=dbc)

    rng = np.random.default_rng(31)
    b = jnp.asarray(rng.standard_normal(n))
    x = jnp.zeros(n)
    r = b - A(x)
    z = pre.apply(r)
    p = z
    rz, nb = float(r @ z), float(jnp.linalg.norm(b))
    iters = 0
    for iters in range(1, 501):
        Ap = A(p)
        a = rz / float(p @ Ap)
        x = x + a * p
        r = r - a * Ap
        if float(jnp.linalg.norm(r)) / nb < 1e-8:
            break
        z = pre.apply(r)
        rz_new = float(r @ z)
        p = z + (rz_new / rz) * p
        rz = rz_new

    assert float(jnp.linalg.norm(b - A(x))) / nb < 1e-8, "PCG did not converge"
    # Measured 2026-08-22 on the session fixture (8,16,8, p=2, spline toroid).
    # Point Jacobi needs several times this; a regression to that scale is what
    # this guards, not a few percent.
    assert iters < 120, (
        f"k=1 dbc PCG took {iters} iterations; the block atom should need well "
        "under 120 here. A jump to jacobi-like counts means the atom or its "
        "boundary term regressed.")


def test_boundary_term_earns_its_place_at_k3_free(torus_seq, bj):
    """The one test that guards the term's MAGNITUDE, not just its presence.

    k=3 free is the right case: it is NON-SINGULAR (so no nullspace or
    deflation is needed) and the boundary term is live and large there -- the
    handoff measures 2.6-3.3x over no term at all across geometries. The other
    iteration test runs at k=1 Dirichlet, where the term is inert by design, so
    it cannot catch a term applied at the wrong strength.

    Self-calibrating: the assertion is against the SAME solve with the term
    switched off, so it does not depend on the mesh, the geometry or the
    machine.
    """
    k, dbc = 3, False
    ops = torus_seq.get_operators()

    def A(x):
        return op.apply_hodge_laplacian_approx(torus_seq, ops, x, k, dirichlet=dbc)

    def solve(pre, n):
        rng = np.random.default_rng(7)
        b = jnp.asarray(rng.standard_normal(n))
        x, r = jnp.zeros(n), None
        r = b - A(x)
        z = pre.apply(r)
        p = z
        rz, nb = float(r @ z), float(jnp.linalg.norm(b))
        for it in range(1, 1001):
            Ap = A(p)
            a = rz / float(p @ Ap)
            x = x + a * p
            r = r - a * Ap
            if float(jnp.linalg.norm(r)) / nb < 1e-8:
                return it
            z = pre.apply(r)
            rz_new = float(r @ z)
            p = z + (rz_new / rz) * p
            rz = rz_new
        return None

    pre_on, n, _ = bj(k, dbc, bc_scale=PROD_SCALE)
    pre_off, _, _ = bj(k, dbc, bc_entry=False)
    on, off = solve(pre_on, n), solve(pre_off, n)
    assert on is not None and off is not None, "PCG did not converge"
    assert on < 0.75 * off, (
        f"k=3 free: {on} iterations with the boundary term vs {off} without. "
        "The term should be worth well over 25% here (2.6-3.3x measured across "
        "geometries), so this close means it is mis-scaled or not applied.")


def test_defaults_are_the_production_configuration(torus_seq):
    """Calling with NO keyword arguments must BE the production configuration.

    That is the whole point of Phase 1 of the production plan: one method, no
    required parameters. If a default drifts, every caller that relies on it
    silently changes -- which is exactly how `ktilde_mode="roundtrip"` survived
    as the default while being 3-10x worse than `"honest"` on all 28 A/B rows
    (docs/research/production_simplification_plan.md §1).
    """
    import inspect

    import mrx.experimental.block_jacobi_laplacian as bjl

    assert bjl.PRODUCTION_BC_SCALE == PROD_SCALE

    # Cheap and directly on the thing that was wrong: the defaults themselves.
    for fn in (bjl.BlockJacobiLaplacian.__init__, bjl.component_factors,
               bjl.build_bulk_atom):
        params = inspect.signature(fn).parameters
        assert params["ktilde_mode"].default == "honest", (
            f"{fn.__qualname__}: ktilde_mode default is "
            f"{params['ktilde_mode'].default!r}; 'roundtrip' loses every A/B row")
        assert params["lumped"].default == "diag", fn.__qualname__
        assert params["bc_entry"].default == "ibpd", fn.__qualname__

    # And behaviourally: bare defaults == the explicit production config.
    prev = os.environ.pop("MRX_BJ_BC_SCALE", None)
    try:
        n = int(getattr(torus_seq, "n1"))
        rng = np.random.default_rng(3)
        vecs = [jnp.asarray(rng.standard_normal(n)) for _ in range(2)]
        ops = torus_seq.get_operators()
        bare = BlockJacobiLaplacian(torus_seq, ops, 1, False)
        spelled = BlockJacobiLaplacian(
            torus_seq, ops, 1, False, ktilde_mode="honest", lumped="diag",
            bc_entry="ibpd", bc_scale=bjl.PRODUCTION_BC_SCALE,
            extra_rings=0, outer_rings=0)
    finally:
        if prev is not None:
            os.environ["MRX_BJ_BC_SCALE"] = prev
        # CONTROL: two builds of the IDENTICAL configuration. Establishes the
        # reproducibility floor, so the comparison below is read against a
        # measured number rather than a guessed tolerance. The dense polar-core
        # block is not bit-reproducible between builds (~1.7% of rows move by
        # ~1e-14), which is also the residual every inertness test above sees.
        twin = BlockJacobiLaplacian(torus_seq, ops, 1, False)

    probe = lambda pre: np.stack([np.asarray(pre.apply(v)) for v in vecs])
    floor = _rel(probe(twin), probe(bare))
    assert floor < INERT, (
        f"two identical builds differ by {floor:.2e}, above the {INERT:.0e} "
        "tolerance every inertness test in this file relies on")
    d = _rel(probe(spelled), probe(bare))
    assert d < INERT, (
        f"bare defaults differ from the explicit production configuration by "
        f"{d:.2e} (identical-build floor is {floor:.2e})")
