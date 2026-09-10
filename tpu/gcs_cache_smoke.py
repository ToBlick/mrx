#!/usr/bin/env python3
"""Smoke-test a GCS-backed XLA compilation cache before trusting it.

A persistent compilation cache can live on a `gs://` path, which is the only
way to carry compiled kernels across the ephemeral local disk of a fresh TPU
node. But a cache path JAX cannot reach does not raise: it retries, and the
process appears to hang while compiling. So the path is proven on a program
that takes a fraction of a second before any real run depends on it.

Two things are checked, in this order:

1. the bucket is reachable and writable with plain object IO, which fails
   loudly and immediately if the scopes or the path are wrong;
2. JAX itself compiles a trivial program against the cache, and a second
   process sees the entry.

Usage:
    python gcs_cache_smoke.py --cache gs://bucket/prefix [--platform tpu]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True,
                    help="cache path, gs://bucket/prefix or a local directory")
    ap.add_argument("--platform", default=None,
                    help="JAX_PLATFORMS value for the compile step")
    ap.add_argument("--child", action="store_true",
                    help=argparse.SUPPRESS)
    return ap.parse_args()


def check_bucket_io(cache: str, timeout_s: int = 60) -> bool:
    """Write and read back one small object, so a bad path fails loudly.

    Returns True when the path is local (nothing to check) or the round trip
    succeeds. This runs before JAX is imported: an unreachable cache makes JAX
    retry rather than raise, and the failure then looks like a hang inside
    compilation instead of a permissions problem.
    """
    if not cache.startswith("gs://"):
        return True
    probe = f"{cache.rstrip('/')}/.smoke_probe"
    try:
        subprocess.run(["gcloud", "storage", "cp", "-", probe],
                       input=b"ok", check=True, timeout=timeout_s,
                       capture_output=True)
        out = subprocess.run(["gcloud", "storage", "cat", probe],
                             check=True, timeout=timeout_s, capture_output=True)
        subprocess.run(["gcloud", "storage", "rm", probe],
                       check=False, timeout=timeout_s, capture_output=True)
    except subprocess.TimeoutExpired:
        print(f"FAIL: {probe} timed out after {timeout_s}s -- "
              "this is what would look like a hang inside JAX.")
        return False
    except subprocess.CalledProcessError as exc:
        err = (exc.stderr or b"").decode().strip().splitlines()
        print(f"FAIL: cannot write {probe}")
        for line in err[-3:]:
            print(f"      {line}")
        return False
    if out.stdout.strip() != b"ok":
        print(f"FAIL: read back {out.stdout!r}, expected b'ok'")
        return False
    print(f"OK  : object round trip on {cache}")
    return True


def compile_trivial(cache: str, platform: str | None):
    """Compile one small program against the cache and report the time."""
    if platform:
        os.environ["JAX_PLATFORMS"] = platform
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_compilation_cache_dir", cache)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)

    dev = jax.devices()[0]
    # Not jnp.arange: the shape is part of the cache key, so a stamp keeps
    # repeated smoke tests from colliding with each other's entries only when
    # asked. Here we WANT the collision, since the point is a cache hit.
    x = jnp.ones((128, 128), dtype=jnp.float32)

    @jax.jit
    def f(a):
        return (a @ a.T).sum()

    t0 = time.perf_counter()
    f(x).block_until_ready()
    dt = time.perf_counter() - t0
    print(f"OK  : compiled on {dev.platform} ({dev.device_kind}) in {dt:.2f}s")
    return dt


def main():
    args = parse_args()

    if args.child:
        compile_trivial(args.cache, args.platform)
        return 0

    print(f"cache: {args.cache}")
    if not check_bucket_io(args.cache):
        return 1

    # Both compiles run in child processes, one at a time. The parent must not
    # touch JAX itself: a TPU admits one process, so a parent holding the chip
    # would make the second compile fail with "TPU is already in use" rather
    # than tell us anything about the cache. Separate processes are also the
    # only way to see a persistent-cache hit, since an in-process compilation
    # cache would answer the second call regardless.
    cmd = [sys.executable, os.path.abspath(__file__), "--child",
           "--cache", args.cache]
    if args.platform:
        cmd += ["--platform", args.platform]

    for label in ("first process (populates the cache):",
                  "second process (should hit the cache):"):
        print(label)
        proc = subprocess.run(cmd, capture_output=True, timeout=900)
        sys.stdout.write(proc.stdout.decode())
        if proc.returncode != 0:
            sys.stderr.write(proc.stderr.decode()[-2000:])
            print("FAIL: compile process errored")
            return 1

    listing = subprocess.run(["gcloud", "storage", "ls", args.cache],
                             capture_output=True, timeout=120) \
        if args.cache.startswith("gs://") else None
    if listing is not None and listing.returncode == 0:
        entries = [ln for ln in listing.stdout.decode().splitlines() if ln.strip()]
        print(f"OK  : {len(entries)} entr{'y' if len(entries) == 1 else 'ies'} "
              f"in the cache")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
