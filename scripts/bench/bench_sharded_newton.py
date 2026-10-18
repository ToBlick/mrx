"""Shard ONE equilibrium's Newton solve over the chips of a multi-chip host: does it get faster?

``scripts/bench/bench_multichip.py`` runs several equilibria side by side, one per chip. This benchmark splits a
single one. The operators are matrix-free: every Hessian apply evaluates fields at the quadrature points, multiplies
them with the geometry there and integrates back. The large arrays are the ones on the quadrature grid
``Q = (Q_r, Q_theta, Q_zeta)`` (``(p + 1)(n_r - p)`` radial points for a uniform radial axis), stored r-major and
flat, while the DoF vectors are small. Sharding the matvec therefore means splitting the quadrature grid into radial
slabs, one per chip. Three ways of running the same work are timed:

``single``
    Everything on one chip, as ``scripts/relax.py`` runs. The baseline.
``gspmd_r``
    The arrays on the quadrature grid (the metric, its inverse, the Jacobian, the mass and reference weights, the
    quadrature points and weights and the CFL weights: every array whose leading axis is the flat point count or
    whose first three axes are ``Q``) are placed with ``NamedSharding(mesh, P("r"))`` on a 1-D ``Auto`` mesh of the
    chips, so each chip holds ``Q_r / n_dev`` radial slabs. All other arrays (DoF vectors, basis tables, extraction
    indices, preconditioners) are replicated. ``jax.jit`` partitions the program (GSPMD/Shardy): the products at
    the quadrature points run on the local slab, an integration back to the DoFs ends with an all-reduce, and all
    work on DoF vectors (extraction gathers, MINRES vector updates, FFT preconditioner) runs replicated on every chip.
``gspmd_r_constrained``
    As ``gspmd_r``, with ``with_sharding_constraint(P("r"))`` on the quadrature-grid intermediates: the output of
    every evaluation at the quadrature points, the input of every integration against a basis, and the two grid
    intermediates of the dense mass apply. The constraints are installed by this script, by wrapping the functions
    that ``mrx.derham_sequence``, ``mrx.projectors`` and ``mrx.operators`` imported, for the duration of the trace
    (:func:`constrain_quadrature`); the library itself is not changed. The wrapper of the mass apply repeats the
    dense einsum path of :func:`mrx.mass.sumfact_apply` (the double-single path of float32 solves is left alone).

Two programs are timed per mode: one :func:`mrx.relaxation.newton.newton_direction` with ``--maxiter`` MINRES
iterations (``--tol`` is tiny by default, so every solve runs the full ``--maxiter``), and one full
``TimeStepper.relaxation_step`` with the same Newton settings. For each the script records the lower and compile
time, the first call, and the median of ``--repeats`` steady-state calls, the compiled memory analysis, the
per-device ``memory_stats()`` (when the backend has them), and the collectives in the optimised HLO: their total
count and the count inside each ``while`` body (the MINRES iteration is one of them, so that count is the number
of collectives per MINRES iteration). The sharded result is compared with the single-chip one.

The radial quadrature count must divide by the number of chips; otherwise the script stops and suggests ``n_r``.

    MRX_DTYPE=float32 python -u scripts/bench/bench_sharded_newton.py --geometry data/wout_li383_1.4m.nc \\
        --resolution 26 44 20 --maxiter 20 --out /gcs/outputs/bench/sharded_newton

On CPU, four fake devices check the sharding (timings there mean nothing):

    JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 MRX_DTYPE=float64 \\
        python -u scripts/bench/bench_sharded_newton.py --resolution 10 12 12 --repeats 2

Output: ``<out>/sharded_newton_results.json``, rewritten after every mode.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import re
import statistics
import sys
import time
import traceback

MODES = ("single", "gspmd_r", "gspmd_r_constrained")
PROGRAMS = ("matvec", "direction", "step")
COLLECTIVES = ("all-reduce", "all-gather", "reduce-scatter", "collective-permute", "all-to-all")
TRACEBACK_CHARS = 4000
DTYPE_BYTES = dict(f64=8, s64=8, u64=8, c64=8, c128=16, f32=4, s32=4, u32=4, bf16=2, f16=2, s8=1, u8=1, pred=1)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--geometry", default="data/wout_li383_1.4m.nc",
                    help="geometry and initial field: a VMEC wout (.nc), GVEC state (.dat) or DESC output (.h5)")
    ap.add_argument("--resolution", type=int, nargs=3, default=(10, 12, 12),
                    help="number of splines in (r, theta, zeta)")
    ap.add_argument("--degree", type=int, default=2, help="spline degree")
    ap.add_argument("--devices", type=int, default=0, help="chips to shard over, 0 = all")
    ap.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES), help="modes to time (see above)")
    ap.add_argument("--programs", nargs="+", choices=PROGRAMS, default=list(PROGRAMS),
                    help="matvec: one apply of the Newton operator (one MINRES iteration's work); direction: one "
                         "newton_direction; step: one full relaxation step")
    ap.add_argument("--maxiter", type=int, default=20, help="MINRES iterations of the Newton solve")
    ap.add_argument("--tol", type=float, default=1e-12,
                    help="Newton solve tolerance; tiny so that every solve runs --maxiter iterations")
    ap.add_argument("--repeats", type=int, default=5, help="steady-state calls per program (the median is reported)")
    ap.add_argument("--max-rel-diff", type=float, default=None,
                    help="largest accepted relative difference to single of the matvec (default 1e-5 in float32, "
                         "1e-12 in float64)")
    ap.add_argument("--max-rel-diff-krylov", type=float, default=None,
                    help="the same for direction and step, whose Krylov solves amplify the changed order of the "
                         "sums (default 1e-2 in float32, 1e-5 in float64; for direction at least 10 times the "
                         "roundoff floor measured on one chip)")
    ap.add_argument("--dump-hlo", action="store_true", help="also write the optimised HLO of every program")
    ap.add_argument("--out", default="outputs/bench/sharded_newton",
                    help="output directory, for sharded_newton_results.json")
    return ap.parse_args()


def fresh(fn):
    """A new function object calling ``fn``, so ``jax.jit`` traces it again instead of reusing the trace of an
    earlier mode (whose wrapped library functions were different)."""
    return lambda *a: fn(*a)


def radial_points(n_r, p):
    """The radial quadrature count of a uniform clamped radial axis: ``p + 1`` Gauss points on each of the
    ``n_r - p`` cells."""
    return (p + 1) * (n_r - p)


def suggest_n_r(n_r, p, n_dev):
    """The nearest ``n_r`` below and above ``n_r`` whose radial quadrature count divides by ``n_dev``."""
    below = next((n for n in range(n_r, p, -1) if radial_points(n, p) % n_dev == 0), None)
    above = next(n for n in range(n_r, n_r + 4 * n_dev + 1) if radial_points(n, p) % n_dev == 0)
    return below, above


def hlo_collectives(text):
    """Count the collectives of an optimised HLO module: in total, and inside each ``while`` body including the
    computations it calls (one iteration of that loop). Returns ``(total, loops)`` with ``loops`` a list of
    ``{body, per_iteration: {op: n}}`` with at least one collective, largest first. The key ``bytes`` holds the
    summed result size of the collectives (per device)."""
    comps, calls, own = {}, {}, {}
    current = None
    op_re = re.compile(r"\b(" + "|".join(COLLECTIVES) + r")(-start)?\(")
    call_re = re.compile(r"\b(?:body|condition|to_apply|calls|true_computation|false_computation)=%?([\w.\-]+)")
    branch_re = re.compile(r"branch_computations=\{([^}]*)\}")
    while_re = re.compile(r"\bwhile\(.*\bbody=%?([\w.\-]+)")
    shape_re = re.compile(r"\b([a-z]+\d*)\[([\d,]*)\]")
    whiles = []
    for line in text.splitlines():
        if line and not line[0].isspace() and line.rstrip().endswith("{"):
            m = re.match(r"(?:ENTRY\s+)?%?([\w.\-]+)", line)
            current = m.group(1) if m else None
            comps[current] = True
            calls[current], own[current] = set(), {}
            continue
        if current is None:
            continue
        for m in op_re.finditer(line):
            own[current][m.group(1)] = own[current].get(m.group(1), 0) + 1
            # the payload: the result shape(s) between '=' and the op name, e.g. 'f32[3,10,12,12]{3,2,1,0}'
            result = line[line.find("=") + 1:m.start()] if "=" in line[:m.start()] else ""
            nbytes = sum(DTYPE_BYTES.get(dt, 4) * math.prod(int(d) for d in dims.split(",") if d)
                         for dt, dims in shape_re.findall(result))
            own[current]["bytes"] = own[current].get("bytes", 0) + nbytes
        for m in call_re.finditer(line):
            calls[current].add(m.group(1))
        for m in branch_re.finditer(line):
            calls[current].update(s.strip().lstrip("%") for s in m.group(1).split(","))
        m = while_re.search(line)
        if m:
            whiles.append(m.group(1))

    memo = {}

    def transitive(name, stack=()):
        if name in memo:
            return memo[name]
        acc = dict(own.get(name, {}))
        for c in calls.get(name, ()):
            if c in stack or c == name:
                continue
            for k, v in transitive(c, stack + (name,)).items():
                acc[k] = acc.get(k, 0) + v
        memo[name] = acc
        return acc

    total = {}
    for counts in own.values():
        for k, v in counts.items():
            total[k] = total.get(k, 0) + v
    loops = [dict(body=b, per_iteration=transitive(b)) for b in dict.fromkeys(whiles)]
    loops = [lp for lp in loops if lp["per_iteration"]]
    loops.sort(key=lambda lp: -sum(v for k, v in lp["per_iteration"].items() if k != "bytes"))
    return total, loops


@contextlib.contextmanager
def constrain_quadrature(mesh):
    """Put ``with_sharding_constraint(P("r"))`` on the quadrature-grid intermediates while a program is traced, by
    wrapping the names that the library modules imported. Restores them on exit."""
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415
    import mrx.derham_sequence as dr  # noqa: PLC0415
    import mrx.double_single as ds  # noqa: PLC0415
    import mrx.operators as ops  # noqa: PLC0415
    import mrx.projectors as proj  # noqa: PLC0415
    from jax.sharding import NamedSharding  # noqa: PLC0415
    from jax.sharding import PartitionSpec as P  # noqa: PLC0415

    def wsc(x, *spec):
        return jax.lax.with_sharding_constraint(x, NamedSharding(mesh, P(*spec)))

    evaluate, integrate_dr, integrate_proj, sumfact = (dr.evaluate_at_xq, dr.integrate_against,
                                                       proj.integrate_against, ops.sumfact_apply)

    def evaluate_c(*a, **k):
        return wsc(evaluate(*a, **k), "r")                         # (n_q, d), r-major flat

    def integrate_c(orig):
        return lambda f_jk, comp_info: orig(wsc(f_jk, "r"), comp_info)

    def sumfact_c(plan, weights, x):
        # The dense path of mrx.mass.sumfact_apply with the two grid intermediates constrained.
        if isinstance(x, ds.DS):
            return sumfact(plan, weights, x)
        R, T, Z = plan.tables_c
        n_max = tuple(t.shape[1] for t in plan.tables_c)
        parts, offset = [], 0
        for s in plan.shapes_c:
            n = s[0] * s[1] * s[2]
            parts.append(jnp.pad(x[offset:offset + n].reshape(s), [(0, n_max[a] - s[a]) for a in range(3)]))
            offset += n
        u = wsc(jnp.einsum('cijk,cia,cjb,ckd->abdc', jnp.stack(parts), R, T, Z), "r")
        v = wsc(jnp.einsum('abdrc,abdc->rabd', weights, u), None, "r")
        R, T, Z = plan.tables_r
        y = jnp.einsum('ria,rjb,rkd,rabd->rijk', R, T, Z, v)
        return jnp.concatenate([y[c, :s[0], :s[1], :s[2]].reshape(-1) for c, s in enumerate(plan.shapes_r)])

    dr.evaluate_at_xq, dr.integrate_against = evaluate_c, integrate_c(integrate_dr)
    proj.integrate_against, ops.sumfact_apply = integrate_c(integrate_proj), sumfact_c
    try:
        yield
    finally:
        dr.evaluate_at_xq, dr.integrate_against = evaluate, integrate_dr
        proj.integrate_against, ops.sumfact_apply = integrate_proj, sumfact


def main():
    args = parse_args()
    os.environ.setdefault("MRX_DTYPE", "float32")
    t_process = time.perf_counter()

    import equinox as eqx  # noqa: PLC0415
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    from jax.sharding import AxisType, NamedSharding  # noqa: PLC0415
    from jax.sharding import PartitionSpec as P  # noqa: PLC0415

    # MRX_DTYPE must be in the environment before mrx is imported: mrx/precision.py reads it once, at import.
    import mrx  # noqa: PLC0415
    from mrx.nullspace import compute_nullspaces  # noqa: PLC0415
    from mrx.relaxation.config import Budget, Descent, Geometry, Method, Newton, RelaxConfig  # noqa: PLC0415
    from mrx.relaxation.initial_conditions import initial_field  # noqa: PLC0415
    from mrx.relaxation.loop import initial_state  # noqa: PLC0415
    # _newton_system is private: it is the operator newton_direction hands to MINRES, timed alone as `matvec`.
    from mrx.relaxation.newton import (_newton_system, harmonic_atom_profiles, newton_direction,  # noqa: PLC0415
                                       parallel_penalty_profile)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from bench_multichip import CompileMetrics, _version, cache_settings  # noqa: PLC0415
    metrics = CompileMetrics().install()

    devices = jax.devices()
    n_dev = args.devices or len(devices)
    devices = devices[:n_dev]
    n_r, p = args.resolution[0], args.degree
    if radial_points(n_r, p) % n_dev:
        below, above = suggest_n_r(n_r, p, n_dev)
        raise SystemExit(f"the radial quadrature count (p + 1)(n_r - p) = {radial_points(n_r, p)} at n_r = {n_r}, "
                         f"p = {p} does not divide by {n_dev} chips: use n_r = {below} or {above}")
    f32 = mrx.DTYPE.name == "float32"
    max_rel = dict(matvec=args.max_rel_diff if args.max_rel_diff is not None else (1e-5 if f32 else 1e-12))
    max_rel["direction"] = max_rel["step"] = (args.max_rel_diff_krylov if args.max_rel_diff_krylov is not None
                                              else (1e-2 if f32 else 1e-5))

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "sharded_newton_results.json")
    results = dict(
        jax=jax.__version__, jaxlib=_version("jaxlib"), libtpu=_version("libtpu"), python=sys.version.split()[0],
        image_name=os.environ.get("IMAGE_NAME"), mrx_file=mrx.__file__,
        devices=n_dev, device_kind=devices[0].device_kind, platform=devices[0].platform,
        dtype=mrx.DTYPE.name, geometry=args.geometry, resolution=list(args.resolution), spline_degree=p,
        maxiter=args.maxiter, tol=args.tol, repeats=args.repeats, max_rel_diff=max_rel,
        cache=cache_settings(), modes={})

    def write():
        with open(out_path, "w") as fh:
            json.dump(results, fh, indent=1)

    print(f"[env] jax {jax.__version__}  mrx {mrx.DTYPE.name} from {mrx.__file__}  {n_dev} x "
          f"{devices[0].device_kind}", flush=True)

    # --- one-time setup ------------------------------------------------------------------------------
    t0 = time.perf_counter()
    geom = Geometry(path=args.geometry, resolution=tuple(args.resolution), spline_degree=p,
                    precision=mrx.DTYPE.name)
    mrx.MAP_BATCH_SIZE_INNER = geom.max_batch
    seq, _ = geom.build()
    compute_nullspaces(seq)
    cfg = RelaxConfig(geometry=geom, descent=Descent(method=Method.NEWTON),
                      newton=Newton(tol=args.tol, maxiter=args.maxiter), budget=Budget(steps=1, chunk=1))
    ts = cfg.stepper(seq)
    B0, _ = initial_field(seq)
    state0 = initial_state(B0, ts)
    J0, load0 = ts._lorentz(B0, None)
    a0 = jnp.zeros(seq.even.n(1))
    # a fixed, smooth-ish test potential for the matvec: the force potential of the start state
    a_test = state0.warm.a_F
    jax.block_until_ready((state0, J0, load0))
    q_shape = tuple(int(n) for n in seq.quad.shape)
    n_q = math.prod(q_shape)
    if q_shape[0] % n_dev:   # non-uniform radial knots
        raise SystemExit(f"Q_r = {q_shape[0]} does not divide by {n_dev} chips; change n_r or --knots")
    results.update(t_setup_s=time.perf_counter() - t0, quad_shape=list(q_shape), n_q=n_q,
                   dofs=dict(B=int(seq.odd.n(2)), a=int(seq.even.n(1)), u=int(seq.even.n(2))),
                   mass_chebyshev_steps=None if ts.mass_inverse is None else ts.mass_inverse.steps)
    print(f"[setup] {time.perf_counter() - t0:.1f}s  Q = {q_shape} ({n_q} points, {q_shape[0] // n_dev} radial "
          f"points per chip)  DoFs: B {seq.odd.n(2)}, a {seq.even.n(1)}  mass Chebyshev degree "
          f"{results['mass_chebyshev_steps']}", flush=True)
    write()

    ts_dyn, ts_static = eqx.partition(ts, eqx.is_array)

    def is_quadrature(x):
        return x.ndim >= 1 and (x.shape[0] == n_q or (x.ndim >= 3 and tuple(x.shape[:3]) == q_shape))

    def place(tree, sharded, replicated):
        """device_put every array leaf, with ``sharded`` for the quadrature-grid arrays and ``replicated`` for the
        rest. A leaf shared between the sequence and its parity views is placed once and stays shared."""
        memo, names = {}, []

        def put(path, x):
            if id(x) not in memo:
                quad = sharded is not None and is_quadrature(x)
                memo[id(x)] = jax.device_put(x, sharded if quad else replicated)
                if quad:
                    names.append((jax.tree_util.keystr(path), tuple(x.shape)))
            return memo[id(x)]
        return jax.tree_util.tree_map_with_path(put, tree), names

    def matvec(ts_dyn_, B, J, a):
        """One apply of the Newton operator ``a -> curl^T H curl a`` (the work of one MINRES iteration), at a
        fixed ``a``; it is exact arithmetic up to the order of the sums, so it checks the partitioning."""
        ts_ = eqx.combine(ts_dyn_, ts_static)
        seq_ = ts_.seq
        penalty = parallel_penalty_profile(harmonic_atom_profiles(seq_, B), ts_.newton_penalty)
        return (_newton_system(seq_, B, J, penalty, ts_.mass_inverse)[2](a),)

    def direction(ts_dyn_, B, J, load, a_guess):
        ts_ = eqx.combine(ts_dyn_, ts_static)
        u, a, info = newton_direction(ts_.seq, B, J, load, a_guess, ts_.newton_penalty, ts_.newton_tol,
                                      ts_.newton_maxiter, ts_.mass_inverse)
        return u, info

    def step(ts_dyn_, state):
        ts_ = eqx.combine(ts_dyn_, ts_static)
        st = ts_.relaxation_step(state)
        return st.B_nplus1, st.dt, st.last.newton_it

    def probe(ts_dyn_, B, u):
        """Two quadrature-grid intermediates of the Hessian apply, returned so their sharding can be printed."""
        ts_ = eqx.combine(ts_dyn_, ts_static)
        odd, even = ts_.seq.odd, ts_.seq.even
        B_jk = odd.evaluate_at_quadrature(B, 2)
        u_jk = even.evaluate_at_quadrature(u, 2)
        return B_jk, odd.cross_product_load_values(u_jk, B_jk, 1, 2, 2)

    mesh = jax.make_mesh((n_dev,), ("r",), axis_types=(AxisType.Auto,), devices=devices)
    reference = {}
    for mode in args.modes:
        print(f"\n================ {mode} ================", flush=True)
        rec = results["modes"][mode] = {}
        try:
            if mode == "single":
                rep, shd = devices[0], None
            else:
                rep, shd = NamedSharding(mesh, P()), NamedSharding(mesh, P("r"))
            ts_m, sharded_leaves = place(ts_dyn, shd, rep)
            (B_m, J_m, load_m, a_m, at_m), _ = place((B0, J0, load0, a0, a_test), None, rep)
            state_m, _ = place(state0, None, rep)
            if shd is not None:
                geo = ts_m.seq.geometry
                rec["sharded_leaves"] = [dict(path=n, shape=list(s)) for n, s in sharded_leaves]
                rec["geometry_sharding"] = {k: str(getattr(geo, k).sharding)
                                            for k in ("metric_jkl", "metric_inv_jkl", "jacobian_j")}
                rec["geometry_sharding"]["mass_weights[2]"] = str(geo.mass_weights[2].sharding)
                shard_shapes = {k: [list(s.data.shape) for s in getattr(geo, k).addressable_shards][:1]
                                for k in ("metric_jkl", "jacobian_j")}
                print(f"[{mode}] {len(sharded_leaves)} quadrature-grid leaves sharded over 'r', e.g.", flush=True)
                for n, s in sharded_leaves[:6]:
                    print(f"    {n} {s}", flush=True)
                for k, v in rec["geometry_sharding"].items():
                    print(f"    geometry.{k}: {v}", flush=True)
                print(f"    per-device shard shapes: {shard_shapes}", flush=True)
            def ctx():
                return constrain_quadrature(mesh) if mode == "gspmd_r_constrained" else contextlib.nullcontext()

            # the sharding the compiler gives two intermediates on the quadrature grid
            with ctx():
                pr = jax.jit(fresh(probe)).lower(ts_m, B_m, load_m).compile()     # load_m: any even 2-form vector
            rec["intermediate_sharding"] = dict(B_jk=str(pr.output_shardings[0]),
                                                cross_product_load=str(pr.output_shardings[1]))
            print(f"[{mode}] intermediate B_jk (n_q, 3): {pr.output_shardings[0]}\n"
                  f"[{mode}] its integral back to the DoFs: {pr.output_shardings[1]}", flush=True)

            programs = dict(matvec=(matvec, (ts_m, B_m, J_m, at_m)),
                            direction=(direction, (ts_m, B_m, J_m, load_m, a_m)), step=(step, (ts_m, state_m)))
            for name in args.programs:
                fn, fargs = programs[name]
                prec = rec[name] = {}
                before = metrics.snapshot()
                t = time.perf_counter()
                with ctx():
                    lowered = jax.jit(fresh(fn)).lower(*fargs)
                t_lower = time.perf_counter() - t
                t = time.perf_counter()
                compiled = lowered.compile()
                t_compile = time.perf_counter() - t
                t = time.perf_counter()
                out = jax.block_until_ready(compiled(*fargs))
                t_first = time.perf_counter() - t
                walls = []
                for _ in range(args.repeats):
                    t = time.perf_counter()
                    out = jax.block_until_ready(compiled(*fargs))
                    walls.append(time.perf_counter() - t)
                hlo = compiled.as_text()
                total, loops = hlo_collectives(hlo)
                if args.dump_hlo:
                    with open(os.path.join(args.out, f"{mode}_{name}.hlo.txt"), "w") as fh:
                        fh.write(hlo)
                prec.update(t_lower_s=t_lower, t_compile_s=t_compile, t_first_call_s=t_first,
                            t_steady_s=walls, t_median_s=statistics.median(walls),
                            compile=CompileMetrics.summary(CompileMetrics.delta(metrics.snapshot(), before)),
                            collectives_total=total, collectives_per_loop_iteration=loops[:8],
                            memory_analysis=_memory_analysis(compiled), memory_stats=_memory_stats(devices),
                            output_sharding=str(out[0].sharding))
                vec = np.asarray(out[0], dtype=np.float64)
                if name == "direction":
                    prec["minres_info"] = int(out[1])
                elif name == "step":
                    prec.update(dt=float(out[1]), newton_it=int(out[2]))
                if mode == "single":
                    reference[name] = vec
                    if name == "direction":
                        # The roundoff floor of the comparison: the same program on one chip with the right-hand
                        # side perturbed by a few ulps. A sharded run that differs by about this much is correct.
                        noise = jax.random.normal(jax.random.PRNGKey(1), load_m.shape, load_m.dtype)
                        load_p = load_m * (1 + 4 * mrx.EPS * noise)
                        vp = np.asarray(compiled(ts_m, B_m, J_m, load_p, a_m)[0], dtype=np.float64)
                        prec["roundoff_floor_rel_diff"] = float(np.linalg.norm(vp - vec) / np.linalg.norm(vec))
                        print(f"[{mode} {name}] roundoff floor (rhs perturbed by 4 ulp): "
                              f"rel diff {prec['roundoff_floor_rel_diff']:.2e}", flush=True)
                elif name in reference:
                    ref = reference[name]
                    rd = float(np.linalg.norm(vec - ref) / np.linalg.norm(ref))
                    floor = results["modes"].get("single", {}).get(name, {}).get("roundoff_floor_rel_diff", 0.0)
                    prec.update(rel_diff_vs_single=rd, matches_single=rd <= max(max_rel[name], 10 * floor))
                    single = results["modes"].get("single", {}).get(name, {})
                    if "t_median_s" in single:
                        prec["speedup_vs_single"] = single["t_median_s"] / prec["t_median_s"]
                loop_txt = "; ".join(f"{lp['body']}: {lp['per_iteration']}" for lp in loops[:3]) or "none"
                print(f"[{mode} {name}] lower {t_lower:.1f}s compile {t_compile:.1f}s first {t_first:.3f}s "
                      f"median {prec['t_median_s']:.4f}s"
                      + (f"  speedup {prec['speedup_vs_single']:.2f}x" if "speedup_vs_single" in prec else "")
                      + (f"  rel diff vs single {prec['rel_diff_vs_single']:.2e} "
                         f"({'OK' if prec['matches_single'] else 'MISMATCH'})" if "rel_diff_vs_single" in prec else "")
                      + f"\n    collectives total {total or 'none'}\n    per while-body iteration (top 3): {loop_txt}"
                      + f"\n    memory analysis {prec['memory_analysis']}", flush=True)
                write()
        except Exception as exc:  # noqa: BLE001  (record and go on with the other modes)
            traceback.print_exc()
            rec.update(error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc()[-TRACEBACK_CHARS:])
        write()

    print("\n[summary] median seconds (speedup vs single, rel diff vs single)", flush=True)
    for mode, rec in results["modes"].items():
        for name in args.programs:
            r = rec.get(name, {})
            if "t_median_s" in r:
                print(f"  {mode:22s} {name:9s} {r['t_median_s']:.4f}s  compile {r['t_compile_s']:.1f}s  "
                      f"speedup {r.get('speedup_vs_single', 1.0):.2f}x  rel diff {r.get('rel_diff_vs_single', 0.0):.1e}",
                      flush=True)
            elif "error" in rec:
                print(f"  {mode:22s} {name:9s} FAILED: {rec['error']}", flush=True)
    results.update(t_process_s=time.perf_counter() - t_process)
    write()
    print(f"\n[done] {time.perf_counter() - t_process:.1f}s, wrote {out_path}", flush=True)


def _memory_analysis(compiled):
    """The per-device sizes of a compiled program, from XLA's memory analysis, in bytes (None if unavailable)."""
    try:
        m = compiled.memory_analysis()
        keys = ("argument_size_in_bytes", "output_size_in_bytes", "temp_size_in_bytes", "alias_size_in_bytes",
                "generated_code_size_in_bytes")
        return {k: int(getattr(m, k)) for k in keys if hasattr(m, k)}
    except Exception:  # noqa: BLE001
        return None


def _memory_stats(devices):
    """``bytes_in_use`` and ``peak_bytes_in_use`` of every device, or None where the backend has no statistics."""
    out = []
    for d in devices:
        try:
            s = d.memory_stats() or {}
        except Exception:  # noqa: BLE001
            s = {}
        out.append({k: s.get(k) for k in ("bytes_in_use", "peak_bytes_in_use", "bytes_limit")} if s else None)
    return out


if __name__ == "__main__":
    main()
