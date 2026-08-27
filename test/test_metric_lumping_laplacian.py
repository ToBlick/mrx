"""Guards for the block-Jacobi Laplacian preconditioner.

`mrx/metric_lumping_laplacian.py` IS the production Laplacian
preconditioner for k = 0..3 (see `docs/source/concepts/PRODUCTION.md`). It had no test
coverage when it was promoted;
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
lost the boundary term entirely. The residual of an inert comparison is BUILD
NOISE, not the term leaking: two builds of the IDENTICAL configuration differ
on the dense polar-core rows, which
`test_defaults_are_the_production_configuration` measures explicitly rather
than assuming. Measured on the (4,6,4) tiny fixture, 2026-08-26: inert
comparisons <= 1.6e-13 in float64 (700 eps) and <= 2.2e-5 in float32 (180
eps); live ones >= 4.4e-3 in either precision. INERT is therefore 1e4 eps
(2.2e-12 / 1.2e-3) and LIVE a fixed 1e-4: eight orders of separation in
float64, one in float32 -- single precision keeps both assertions but loses
the wide band between them.

The tests reuse the session `tiny_seq` fixture and a module-scoped build
cache, so no test here assembles a sequence, computes a nullspace, or builds
the same preconditioner twice. The iteration-count regression of the
production k=1 Dirichlet solve is the band in `test/test_poisson.py`.
"""
import os

import numpy as np
import pytest

jnp = pytest.importorskip("jax.numpy")

import mrx  # noqa: E402
import mrx.operators as op  # noqa: E402
from mrx.metric_lumping_laplacian import (  # noqa: E402
    MetricLumpingLaplacian, MetricLumpingMass, trace_components,
)

# k = 3 is the cheapest vector case (one component); k = 1 is the one
# production cares most about. k = 2 tracks k = 1 in every measurement in the
# handoff, so it does not earn its own build here.
KS = (1, 3)
PROD_SCALE = 3.0
# Thresholds are set from the MEASURED separation, not chosen round numbers.
#   inert (Dirichlet, bc_scale 0 vs 100)  <= 1.6e-13 f64 / 2.2e-5 f32 -- pure
#       build noise: the dense polar-core block is not reproducible to the
#       last bit between builds. Measured, not assumed, in
#       test_defaults_are_the_production_configuration.
#   live  (free, bc_scale 0 vs 3.0 or 100)  >= 4.4e-3 at k=1, the WEAKEST
#       case (the term touches one radial row of one component in three, and
#       P's response saturates -- cf. min eig(P) = 1/(1+r s), so 3.0 and 100
#       move it by nearly the same amount).
# The noise is a few hundred eps in either precision, so INERT is eps-scaled:
# 1e4 eps = 2.2e-12 f64 / 1.2e-3 f32, a decade above the noise. LIVE is a
# measured physical effect, 40x below the weakest one, and does not scale.
INERT = mrx.eps(1e4)   # two builds that must agree
LIVE = 1e-4            # ... and, for the control, must NOT
# The hand-written PCG loops below stop at the residual an iterative solve
# can reach in the working dtype.
CG_TOL = mrx.sqrt_eps()


def _probe_cache(seq):
    """Memoised preconditioner probes on ``seq``: one build per configuration."""
    cache = {}

    def get(k, dbc, *, bc_entry="ibpd", bc_scale=PROD_SCALE, nvec=4):
        key = (k, dbc, bc_entry, bc_scale)
        if key not in cache:
            prev = os.environ.get("MRX_BJ_BC_SCALE")
            os.environ["MRX_BJ_BC_SCALE"] = str(bc_scale)
            try:
                pre = MetricLumpingLaplacian(
                    seq, seq.get_operators(), k, dbc,
                    ktilde_mode="honest", lumped="diag", bc_entry=bc_entry,
                    extra_rings=0, outer_rings=0)
            finally:
                if prev is None:
                    os.environ.pop("MRX_BJ_BC_SCALE", None)
                else:
                    os.environ["MRX_BJ_BC_SCALE"] = prev
            n = int(getattr(seq, f"n{k}_dbc" if dbc else f"n{k}"))
            rng = np.random.default_rng(0)
            probe = np.stack([
                np.asarray(pre.apply(jnp.asarray(rng.standard_normal(n))))
                for _ in range(nvec)])
            cache[key] = (pre, n, probe)
        return cache[key]

    return get


@pytest.fixture(scope="module")
def bj(tiny_seq):
    return _probe_cache(tiny_seq)


def _rel(a, b):
    return float(np.abs(a - b).max() / np.abs(b).max())


@pytest.mark.parametrize("k", KS)
def test_boundary_term_vanishes_under_dirichlet(bj, k):
    """THE invariant: under an essential condition the boundary term does not
    exist, so the scale must not move the preconditioner."""
    lo = bj(k, True, bc_scale=0.0)[2]
    hi = bj(k, True, bc_scale=100.0)[2]
    d_dbc = _rel(hi, lo)
    print(f"\n  k={k} dbc inert d={d_dbc:.2e}")
    assert d_dbc < INERT, (
        f"k={k} dbc: a 100x change in bc_scale moved the preconditioner by "
        f"{d_dbc:.2e}; the boundary term must vanish under Dirichlet")

    # POSITIVE CONTROL -- the same comparison where the term IS live. Without
    # this, the assertion above would also pass if the term were never
    # assembled at all.
    d_free = _rel(bj(k, False, bc_scale=100.0)[2], bj(k, False, bc_scale=0.0)[2])
    print(f"  k={k} free live d={d_free:.2e}")
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
    print(f"\n  k=0 free inert d={d:.2e}")
    assert d < INERT, (
        f"k=0 free: bc_scale moved the preconditioner by {d:.2e}, but k=0 has "
        "no boundary trace to scale")


@pytest.mark.parametrize("k", KS)
def test_scale_zero_reproduces_no_boundary_term(bj, k):
    """`bc_scale=0` must be the same operator as `bc_entry=False`."""
    off = bj(k, False, bc_entry=False)[2]
    zero = bj(k, False, bc_entry="ibpd", bc_scale=0.0)[2]
    d = _rel(zero, off)
    print(f"\n  k={k} scale-zero inert d={d:.2e}")
    assert d < INERT, (
        f"k={k} free: bc_scale=0 differs from bc_entry=False by {d:.2e}")

    # POSITIVE CONTROL: the production scale must NOT match "no term".
    d_live = _rel(bj(k, False, bc_scale=PROD_SCALE)[2], off)
    print(f"  k={k} prod-scale live d={d_live:.2e}")
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
    print(f"\n  k={k} dbc={dbc} asym={asym:.2e}")
    # a roundoff identity through the dense polar-core solve
    assert asym < INERT, f"k={k} dbc={dbc}: P is not symmetric ({asym:.2e})"
    np.linalg.cholesky(0.5 * (dense + dense.T))   # raises if not positive definite


def test_boundary_term_earns_its_place_at_k3_free(tiny_seq, bj):
    """The one test that guards the term's MAGNITUDE, not just its presence.

    k=3 free is the right case: it is NON-SINGULAR (so no nullspace or
    deflation is needed) and the boundary term is live and large there -- the
    handoff measures 2.6-3.3x over no term at all across geometries at
    production resolution. The k=1 Dirichlet band in test_poisson.py runs
    where the term is inert by design, so it cannot catch a term applied at
    the wrong strength.

    Self-calibrating: the assertion is against the SAME solve with the term
    switched off, so it does not depend on the geometry or the machine --
    only the ratio is resolution-dependent, and it is measured on tiny_seq.
    """
    k, dbc = 3, False
    ops = tiny_seq.get_operators()

    def A(x):
        return op.apply_laplacian_approx(tiny_seq, ops, x, k, dirichlet=dbc)

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
            if float(jnp.linalg.norm(r)) / nb < CG_TOL:
                return it
            z = pre.apply(r)
            rz_new = float(r @ z)
            p = z + (rz_new / rz) * p
            rz = rz_new
        return None

    pre_on, n, _ = bj(k, dbc, bc_scale=PROD_SCALE)
    pre_off, _, _ = bj(k, dbc, bc_entry=False)
    on, off = solve(pre_on, n), solve(pre_off, n)
    print(f"\n  k=3 free PCG: {on} iterations with the boundary term, {off} without")
    assert on is not None and off is not None, "PCG did not converge"
    # Measured 2026-08-26 on tiny_seq ((4, 6, 4) p=2), float64: 24 iterations
    # with the term against 30 without (see the print) -- 20% at four radial
    # cells, where most rows ARE boundary rows, against 2.6-3.3x at
    # production resolution. The band is halfway to "no term": a term that
    # is not applied gives 30, a mis-scaled one lands in between.
    assert on <= 0.9 * off, (
        f"k=3 free: {on} iterations with the boundary term vs {off} without "
        "(24 vs 30 measured); the term is mis-scaled or not applied.")


def test_defaults_are_the_production_configuration(tiny_seq):
    """Calling with NO keyword arguments must BE the production configuration.

    That is the whole point of Phase 1 of the production plan: one method, no
    required parameters. If a default drifts, every caller that relies on it
    silently changes -- which is exactly how `ktilde_mode="roundtrip"` survived
    as the default while being 3-10x worse than `"honest"` on all 28 A/B rows
    (docs/research/production_simplification_plan.md §1).
    """
    import inspect

    import mrx.metric_lumping_laplacian as bjl

    assert bjl.PRODUCTION_BC_SCALE == PROD_SCALE

    # Cheap and directly on the thing that was wrong: the defaults themselves.
    for fn in (bjl.MetricLumpingLaplacian.__init__, bjl.component_factors,
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
        n = int(getattr(tiny_seq, "n1"))
        rng = np.random.default_rng(3)
        vecs = [jnp.asarray(rng.standard_normal(n)) for _ in range(2)]
        ops = tiny_seq.get_operators()
        bare = MetricLumpingLaplacian(tiny_seq, ops, 1, False)
        spelled = MetricLumpingLaplacian(
            tiny_seq, ops, 1, False, ktilde_mode="honest", lumped="diag",
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
        twin = MetricLumpingLaplacian(tiny_seq, ops, 1, False)

    def probe(pre):
        return np.stack([np.asarray(pre.apply(v)) for v in vecs])

    floor = _rel(probe(twin), probe(bare))
    print(f"\n  identical-build floor={floor:.2e}")
    assert floor < INERT, (
        f"two identical builds differ by {floor:.2e}, above the {INERT:.0e} "
        "tolerance every inertness test in this file relies on")
    d = _rel(probe(spelled), probe(bare))
    assert d < INERT, (
        f"bare defaults differ from the explicit production configuration by "
        f"{d:.2e} (identical-build floor is {floor:.2e})")


def test_production_dispatch_wiring(tiny_seq, laplacian_jacobi_diag):
    """`kind='metric_lumping'` reaches the atom, and `kind='auto'` prefers it once it is
    assembled -- the Phase 1b wiring.

    `'auto'` used to resolve to `'jacobi'` unconditionally while its docstring
    claimed it preferred `'tensor'` at k=0, so this pins the new behaviour on
    both sides: jacobi before assembly, block after. The jacobi diagonals
    come from the session fixture rather than a rebuild here.
    """
    from mrx.operators import (
        METRIC_LUMPING_CACHE_ATTR, assemble_metric_lumping_laplacian_preconditioner,
    )

    k, dbc = 3, False          # non-singular, single component: the cheap case
    n = int(getattr(tiny_seq, f"n{k}"))
    rng = np.random.default_rng(11)
    v = jnp.asarray(rng.standard_normal(n))
    ops = tiny_seq.get_operators()

    prev = getattr(tiny_seq, METRIC_LUMPING_CACHE_ATTR, None)
    try:
        # Before assembly: 'auto' must fall back, and 'metric_lumping' must say so
        # clearly rather than silently doing something else.
        setattr(tiny_seq, METRIC_LUMPING_CACHE_ATTR, None)
        auto_before = tiny_seq.apply_laplacian_preconditioner(
            v, k, dirichlet=dbc, kind='auto')
        jac = tiny_seq.apply_laplacian_preconditioner(
            v, k, dirichlet=dbc, kind='jacobi')
        # `_rel`, not exact equality: two calls down the SAME path differ by
        # ~1 ULP (jax re-tracing), which is the third time in this file that
        # bit-identity turned out to be the wrong assertion. INERT is four
        # orders below the LIVE separation asserted at the end of this test.
        assert _rel(np.asarray(auto_before), np.asarray(jac)) < INERT, (
            "kind='auto' did not fall back to jacobi before assembly")
        with pytest.raises(ValueError, match="not assembled"):
            tiny_seq.apply_laplacian_preconditioner(
                v, k, dirichlet=dbc, kind='metric_lumping')

        # After assembly: 'metric_lumping' is the atom, and 'auto' now picks it.
        assemble_metric_lumping_laplacian_preconditioner(
            tiny_seq, ops, ks=(k,), dirichlets=(dbc,))
        blk = tiny_seq.apply_laplacian_preconditioner(
            v, k, dirichlet=dbc, kind='metric_lumping')
        auto_after = tiny_seq.apply_laplacian_preconditioner(
            v, k, dirichlet=dbc, kind='auto')
        assert _rel(np.asarray(auto_after), np.asarray(blk)) < INERT, (
            "kind='auto' did not prefer the block atom after assembly")
        assert _rel(np.asarray(blk), np.asarray(jac)) > LIVE, (
            "kind='metric_lumping' returned essentially the jacobi diagonal; the "
            "dispatch is not reaching the atom")
    finally:
        setattr(tiny_seq, METRIC_LUMPING_CACHE_ATTR, prev)


def test_first_apply_inside_a_trace_does_not_poison_the_instance(tiny_seq):
    """The payload is built at CONSTRUCTION, not memoised on the first apply.

    docs/research/OPEN.md 1.1: the flattened payload used to be built lazily
    on the first ``apply`` and stored on the instance. Instances are
    long-lived (a dict on ``seq``, session-scoped in this suite), so a first
    apply inside a ``lax`` body stashed TRACERS on the object and the failure
    surfaced as an ``UnexpectedTracerError`` in whatever applied it next --
    an unrelated test. Here the very first apply of two fresh instances is
    inside a ``lax.scan`` body, and the eager apply afterwards must still
    work and agree with it.
    """
    import jax

    k, dbc = 3, False          # single component: the cheapest pair to build
    ops = tiny_seq.get_operators()
    lap = MetricLumpingLaplacian(tiny_seq, ops, k, dbc)
    mass = MetricLumpingMass(tiny_seq, ops, k, dbc)
    n = int(getattr(tiny_seq, f"n{k}"))
    v = jnp.asarray(np.random.default_rng(23).standard_normal(n))

    def step(x):
        return lap.apply(x) + mass.apply(x)

    inside, _ = jax.lax.scan(lambda x, _: (step(x), None), v, None, length=2)
    outside = step(step(v))            # the eager apply, AFTER the traced one
    assert np.all(np.isfinite(np.asarray(outside)))
    assert _rel(np.asarray(outside), np.asarray(inside)) < INERT, (
        "eager apply after a traced first apply disagrees with the trace")


def test_metric_lumping_mass_is_the_default_and_jit_safe(tiny_seq):
    """`metric_lumping` IS the production mass preconditioner, and works in a trace.

    The load-bearing part: the build is host-side
    numpy, so an apply that touches the host dies with
    TracerArrayConversionError inside `solve_singular_cg`'s `jax.lax.while_loop`
    while working perfectly on a concrete array. That is exactly how an earlier
    version of this test passed against a mass preconditioner that could not be
    used in production at all.
    """
    import jax

    from mrx.operators import _build_operator_preconditioner_apply
    from mrx.preconditioners import (
        MassPreconditionerSpec, default_mass_preconditioner,
    )

    assert default_mass_preconditioner().kind == 'metric_lumping'

    k, dbc = 3, False       # single component: the cheapest mass to build
    n = int(getattr(tiny_seq, f"n{k}"))
    ops = tiny_seq.get_operators()
    rng = np.random.default_rng(5)
    v = jnp.asarray(rng.standard_normal(n))

    spec = MassPreconditionerSpec(kind='metric_lumping')
    out = _build_operator_preconditioner_apply(
        tiny_seq, ops, k=k, dirichlet=dbc, operator_apply=None,
        preconditioner=spec)(v)
    assert np.all(np.isfinite(np.asarray(out)))

    # THE CHECK THAT MATTERS: usable under jit, not merely callable.
    fn = _build_operator_preconditioner_apply(
        tiny_seq, ops, k=k, dirichlet=dbc, operator_apply=None,
        preconditioner=spec)
    assert np.all(np.isfinite(np.asarray(jax.jit(fn)(v)))), (
        "metric_lumping is not usable under jit")

    # ... and a spec naming a kind that does not exist fails loudly rather
    # than quietly doing something else, at every degree. It raises before
    # anything is built, so this costs nothing.
    for bad_k in (0, 1, 2, 3):
        with pytest.raises(ValueError, match="preconditioner kind"):
            _build_operator_preconditioner_apply(
                tiny_seq, ops, k=bad_k, dirichlet=dbc, operator_apply=None,
                preconditioner=MassPreconditionerSpec(kind='no_such_kind'))


def test_probed_diagonal_is_the_honest_reference(tiny_seq):
    """`_probed_laplacian_diaginv` is the exact diagonal of `L_k` as applied.

    `kind='jacobi'` is NOT that for k >= 1: its weak half is a closed form
    under the Kronecker mass model, i.e. a model of `D M^-1 D^T` rather than
    the operator's own. Any gap between the two is the mass model's error, and
    a preconditioner measured against `jacobi` inherits it. This pins the
    distinction so the reference cannot silently drift back to the model.
    """
    # Not a preconditioner KIND any more -- the kinds are none/jacobi/metric_lumping --
    # but still the reference the jacobi diagonal has to be checked against.
    from mrx.operators import (
        PROBED_DIAG_CACHE_ATTR, _laplacian_diaginv, _probed_laplacian_diaginv,
    )

    ops = tiny_seq.get_operators()
    prev = getattr(tiny_seq, PROBED_DIAG_CACHE_ATTR, None)
    try:
        for k, dbc in ((3, False), (0, False)):
            n = int(getattr(tiny_seq, f"n{k}"))
            rng = np.random.default_rng(17)
            v = jnp.asarray(rng.standard_normal(n))
            probed = _probed_laplacian_diaginv(
                tiny_seq, ops, k, dbc) * v
            modelled = np.asarray(_laplacian_diaginv(tiny_seq, ops, k, dbc)) * np.asarray(v)
            assert np.all(np.isfinite(np.asarray(probed)))
            d = _rel(np.asarray(probed), modelled)
            print(f"\n  k={k} probed-vs-closed-form d={d:.2e}")
            if k == 0:
                # L_0 = S_0 has NO weak term, so there is no mass model to be
                # wrong: the closed form is exact and the two must agree --
                # to the roundoff of an O(N)-apply probe.
                assert d < INERT, (
                    f"k=0: probed and closed-form diagonals differ by {d:.2e}, "
                    "but L_0 has no weak term for the model to approximate")
            else:
                # k >= 1: they may differ (that IS the mass model's error), but
                # both must be genuine diagonals of the same operator, so they
                # cannot disagree wildly.
                assert d < 1.0, (
                    f"k={k}: probed vs modelled diagonal differ by {d:.2e}; "
                    "one of them is not diag(L_k)")
    finally:
        setattr(tiny_seq, PROBED_DIAG_CACHE_ATTR, prev)
