#!/usr/bin/env python3
"""Run the MRX toroid-Poisson tutorial on a TPU and report what the hardware did.

This is a wrapper, not a fork: ``scripts/tutorials/toroid_poisson.py`` runs
unmodified under ``MRX_DTYPE=float32``, so the only things worth adding are the
checks that a TPU makes necessary.

The important one is matmul precision. MRX's ``precision.py`` sets
``jax_default_matmul_precision="highest"`` because float32 dot products in TF32
on Ampere GPUs made the W7-X map's ``dR/dtheta`` come out 19% wrong and drove
``det DF`` negative. A TPU's MXU is coarser still -- it decomposes float32 into
bfloat16 passes with an 8-bit mantissa -- so the same failure is more likely
here, and it would show up as quietly wrong physics rather than an exception.
We therefore verify the setting took effect and compare the resulting errors
against a CPU float32 reference measured on this same commit.

Usage:
    MRX_DTYPE=float32 python scripts/benchmark/poisson_regression.py
    MRX_DTYPE=float32 python scripts/benchmark/poisson_regression.py --n 6 8 --p 2
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Relative L2 errors from scripts/tutorials/toroid_poisson.py --n 6 8 --p 2 on
# static-dynamic-refactor (bdeae6e), CPU, float32 and float64 alike. Any TPU
# result that misses these indicates precision loss in the MXU, not a bug in the
# surrounding setup.
CPU_REFERENCE = {(2, 6): 1.0754e-02, (2, 8): 3.5617e-03}
REFERENCE_RTOL = 0.02

ROW = re.compile(r"^\s*(\d+)\s+(\d+)\s+([0-9.eE+-]+)\s+(\d+)\s*$")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, nargs="+", default=[6, 8])
    ap.add_argument("--p", type=int, nargs="+", default=[2])
    ap.add_argument("--repo", default=os.environ.get("MRX_REPO", "/mnt/data/mrx"))
    ap.add_argument("--outdir", default=os.environ.get(
        "MRX_OUTDIR", "/mnt/data/mrx_tpu_results"))
    ap.add_argument("--matmul-size", type=int, default=4096)
    ap.add_argument("--skip-benchmark", action="store_true")
    return ap.parse_args()


def tpu_generation(device_kind: str) -> str:
    """Map a JAX device_kind to a TPU generation label.

    Results from v5e, v5p and v6e all get compared against the same CPU
    reference, so the generation has to be recorded alongside them or the
    numbers are not interpretable after the fact.
    """
    kind = device_kind.lower()
    for needle, label in (("v6e", "v6e"), ("v5 lite", "v5e"), ("v5lite", "v5e"),
                          ("v5e", "v5e"), ("v5p", "v5p"), ("v4", "v4")):
        if needle in kind:
            return label
    return device_kind or "unknown"


# A TPU can be held by exactly one process at a time: a second one gets
# "ABORTED: The TPU is already in use by process with pid N". This driver runs
# the stock tutorial as a subprocess, so it must not hold the TPU itself.
# Every JAX-touching phase therefore runs in its own short-lived subprocess and
# communicates back over stdout as JSON.
_BACKEND_PROBE = r"""
import json, platform
import jax, mrx
devices = jax.devices()
print("@@JSON@@" + json.dumps({
    "jax_version": jax.__version__,
    "platform": devices[0].platform,
    "device_kind": devices[0].device_kind,
    "device_count": jax.device_count(),
    "local_device_count": jax.local_device_count(),
    "devices": [str(d) for d in devices],
    "mrx_dtype": str(mrx.DTYPE),
    "mrx_eps": float(mrx.EPS),
    "x64_enabled": bool(jax.config.jax_enable_x64),
    "matmul_precision": str(jax.config.jax_default_matmul_precision),
    "python": platform.python_version(),
}))
"""

_BENCH_PROBE = r"""
import json, time
import jax, jax.numpy as jnp
size = {size}
n = jax.device_count()
key = jax.random.PRNGKey(0)
a = jax.random.normal(key, (n, size, size), dtype=jnp.float32)
b = jax.random.normal(key, (n, size, size), dtype=jnp.float32)
matmul = jax.pmap(lambda x, y: x @ y)
jax.block_until_ready(matmul(a, b))
reps = 5
t0 = time.perf_counter()
for _ in range(reps):
    out = matmul(a, b)
jax.block_until_ready(out)
dt = time.perf_counter() - t0
print("@@JSON@@" + json.dumps({{
    "matmul_size": size, "device_count": n, "reps": reps,
    "seconds": dt, "tflops": 2.0 * size**3 * n * reps / dt / 1e12,
}}))
"""


def _run_json_probe(code: str, what: str) -> dict:
    """Execute a snippet in a fresh interpreter and parse its @@JSON@@ line."""
    proc = subprocess.run([sys.executable, "-u", "-c", code],
                          capture_output=True, text=True)
    for line in proc.stdout.splitlines():
        if line.startswith("@@JSON@@"):
            return json.loads(line[len("@@JSON@@"):])
    raise SystemExit(
        f"{what} probe failed (exit {proc.returncode}):\n{proc.stderr[-2000:]}")


def describe_backend() -> dict:
    info = _run_json_probe(_BACKEND_PROBE, "backend")
    info["tpu_generation"] = tpu_generation(info["device_kind"])
    return info


def check_matmul_precision(info: dict) -> list[str]:
    """Return a list of warnings; empty means the precision setup is sound."""
    warnings: list[str] = []
    if info["platform"] == "tpu" and info["matmul_precision"] != "highest":
        warnings.append(
            "jax_default_matmul_precision is "
            f"{info['matmul_precision']!r}, not 'highest'. On TPU the MXU will "
            "run float32 matmuls through bfloat16 passes (8-bit mantissa). MRX "
            "spline derivatives are cancelling sums and will lose accuracy "
            "silently -- the authors measured 19% error and a negative det DF "
            "from exactly this on GPU TF32."
        )
    if info["mrx_dtype"] == "float64" and info["platform"] == "tpu":
        warnings.append(
            "MRX_DTYPE=float64 on TPU: there is no native 64-bit path, so XLA "
            "emulates it with multiple 32-bit passes. Expect it to be very slow "
            "or to fail outright with UNIMPLEMENTED."
        )
    return warnings


def run_tutorial(repo: str, ns: list[int], ps: list[int]) -> tuple[list[dict], str, float]:
    """Run the stock tutorial and parse its table."""
    script = Path(repo) / "scripts" / "tutorials" / "toroid_poisson.py"
    if not script.is_file():
        raise SystemExit(f"tutorial not found: {script}")

    cmd = [sys.executable, "-u", str(script),
           "--n", *[str(n) for n in ns],
           "--p", *[str(p) for p in ps]]
    print(f"$ {' '.join(cmd)}", flush=True)

    start = time.time()
    proc = subprocess.run(cmd, cwd=repo, capture_output=True, text=True)
    elapsed = time.time() - start

    print(proc.stdout, end="")
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        raise SystemExit(f"tutorial failed with exit {proc.returncode}")

    rows = []
    for line in proc.stdout.splitlines():
        m = ROW.match(line)
        if m:
            rows.append({
                "p": int(m.group(1)),
                "n": int(m.group(2)),
                "error": float(m.group(3)),
                "iters": int(m.group(4)),
            })
    if not rows:
        raise SystemExit("could not parse any result rows from the tutorial")
    return rows, proc.stdout, elapsed


def compare_reference(rows: list[dict]) -> list[dict]:
    """Diff measured errors against the CPU float32 reference."""
    checks = []
    for row in rows:
        key = (row["p"], row["n"])
        expected = CPU_REFERENCE.get(key)
        if expected is None:
            continue
        rel = abs(row["error"] - expected) / expected
        checks.append({
            "p": row["p"],
            "n": row["n"],
            "measured": row["error"],
            "cpu_reference": expected,
            "relative_deviation": rel,
            "ok": rel <= REFERENCE_RTOL,
        })
    return checks


def benchmark_chips(size: int) -> dict:
    """Measure achieved float32 TFLOP/s across every chip via pmap.

    The MRX solve is a single-device matrix-free CG, so it occupies one chip no
    matter how many the VM has. Measuring the aggregate separately is what shows
    the headroom that is actually going unused.
    """
    return _run_json_probe(_BENCH_PROBE.format(size=size), "benchmark")


def write_outputs(outdir: Path, payload: dict, raw_stdout: str) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    (outdir / "toroid_poisson.stdout.txt").write_text(raw_stdout)

    info = payload["backend"]
    lines = [
        "# MRX on TPU",
        "",
        f"- Timestamp: {payload['timestamp']}",
        f"- Backend: {info['platform']} / {info['device_kind']} "
        f"({info['tpu_generation']})",
        f"- Devices: {info['device_count']} "
        f"(local {info['local_device_count']})",
        f"- JAX: {info['jax_version']}",
        f"- MRX dtype: {info['mrx_dtype']} (eps {info['mrx_eps']:.3e})",
        f"- x64 enabled: {info['x64_enabled']}",
        f"- Matmul precision: {info['matmul_precision']}",
        "",
        "## Toroid Poisson",
        "",
        "| p | n | error | CG iters |",
        "|---|---|---|---|",
    ]
    for row in payload["tutorial"]["rows"]:
        lines.append(
            f"| {row['p']} | {row['n']} | {row['error']:.4e} | {row['iters']} |")
    lines += ["", f"Wall clock: {payload['tutorial']['seconds']:.1f}s", ""]

    if payload["reference_checks"]:
        lines += ["## Agreement with the CPU float32 reference", "",
                  "| p | n | measured | reference | deviation | ok |",
                  "|---|---|---|---|---|---|"]
        for c in payload["reference_checks"]:
            lines.append(
                f"| {c['p']} | {c['n']} | {c['measured']:.4e} | "
                f"{c['cpu_reference']:.4e} | {c['relative_deviation']*100:.2f}% "
                f"| {'yes' if c['ok'] else 'NO'} |")
        lines.append("")

    if payload.get("benchmark"):
        b = payload["benchmark"]
        lines += [
            "## Aggregate chip throughput",
            "",
            f"{b['device_count']} chips, {b['matmul_size']}^2 float32 matmul: "
            f"**{b['tflops']:.1f} TFLOP/s**",
            "",
            "The MRX solve is single-device matrix-free CG and uses one chip; "
            "this figure is the headroom available to a sharded implementation.",
            "",
        ]

    if payload["warnings"]:
        lines += ["## Warnings", ""]
        lines += [f"- {w}" for w in payload["warnings"]]
        lines.append("")

    (outdir / "summary.md").write_text("\n".join(lines))
    return outdir


def main() -> int:
    args = parse_args()

    print("=" * 66)
    print("MRX on TPU")
    print("=" * 66)

    info = describe_backend()
    for key in ("jax_version", "platform", "device_kind", "tpu_generation",
                "device_count", "mrx_dtype", "mrx_eps", "x64_enabled",
                "matmul_precision"):
        print(f"  {key:<20s} {info[key]}")
    print(f"  {'devices':<20s} {info['devices']}")
    print()

    warnings = check_matmul_precision(info)
    for w in warnings:
        print(f"WARNING: {w}\n", file=sys.stderr)

    rows, raw_stdout, elapsed = run_tutorial(args.repo, args.n, args.p)
    print(f"\ntutorial wall clock: {elapsed:.1f}s\n")

    checks = compare_reference(rows)
    if checks:
        print("Agreement with the CPU float32 reference:")
        for c in checks:
            verdict = "ok" if c["ok"] else "DEVIATION"
            print(f"  p={c['p']} n={c['n']}  measured {c['measured']:.4e}  "
                  f"reference {c['cpu_reference']:.4e}  "
                  f"{c['relative_deviation']*100:6.2f}%  {verdict}")
        drifted = [c for c in checks if not c["ok"]]
        if drifted:
            warnings.append(
                f"{len(drifted)} result(s) deviate from the CPU float32 "
                "reference by more than "
                f"{REFERENCE_RTOL*100:.0f}%. On TPU the likely cause is "
                "reduced-precision matmul in the MXU, not a setup error."
            )
        print()

    benchmark = None
    if not args.skip_benchmark:
        print(f"Benchmarking {info['device_count']} chip(s), "
              f"{args.matmul_size}^2 float32 matmul...")
        try:
            benchmark = benchmark_chips(args.matmul_size)
            print(f"  {benchmark['tflops']:.1f} TFLOP/s aggregate\n")
        except Exception as exc:  # noqa: BLE001 - benchmark must never be fatal
            print(f"  benchmark failed: {exc}\n", file=sys.stderr)
            warnings.append(f"chip benchmark failed: {exc}")

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "backend": info,
        "tutorial": {
            "ns": args.n,
            "ps": args.p,
            "rows": rows,
            "seconds": elapsed,
        },
        "reference_checks": checks,
        "benchmark": benchmark,
        "warnings": warnings,
        "mrx_commit": git_commit(args.repo),
    }

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    outdir = write_outputs(Path(args.outdir) / stamp, payload, raw_stdout)
    print(f"Results written to {outdir}")

    return 1 if any(not c["ok"] for c in checks) else 0


def git_commit(repo: str) -> str | None:
    if not shutil.which("git"):
        return None
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             cwd=repo, capture_output=True, text=True)
        return out.stdout.strip() or None
    except OSError:
        return None


if __name__ == "__main__":
    sys.exit(main())
