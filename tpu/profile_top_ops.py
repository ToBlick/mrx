#!/usr/bin/env python3
"""Reduce a jax.profiler trace to a top-N XLA op table by self time.

``jax.profiler.trace`` writes a protobuf (``*.xplane.pb``) that is meant to be
read by TensorBoard. This turns it into the one table worth reading: which XLA
operation actually consumed the device time. That names the bottleneck instead
of inferring it from wall-clock phase timings.

    python -u profile_top_ops.py <trace_dir> [--top 20]

The converter lives in a plugin that is not part of JAX itself, and its module
path has moved between releases, so several are tried. If none is importable
the script says so and exits 2 rather than failing obscurely; install it with

    pip install tensorboard-plugin-profile

The primitive timings in tpu_bench_mrx.py answer the same question less
directly, so a missing plugin degrades the evidence but does not block it.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import sys
from collections import defaultdict


def find_xplane(trace_dir):
    hits = sorted(glob.glob(os.path.join(trace_dir, "**", "*.xplane.pb"),
                            recursive=True))
    return hits[-1] if hits else None


def find_trace_json(trace_dir):
    hits = sorted(glob.glob(os.path.join(trace_dir, "**", "*trace.json*"),
                            recursive=True))
    return hits[-1] if hits else None


def via_plugin(xplane_path, top):
    """Use the profile plugin's own converter, if one of its paths imports."""
    convert = None
    for module_path, attr in (
        ("tensorboard_plugin_profile.convert.raw_to_tool_data", "xspace_to_tool_data"),
        ("xprof.convert.raw_to_tool_data", "xspace_to_tool_data"),
    ):
        try:
            mod = __import__(module_path, fromlist=[attr])
            convert = getattr(mod, attr)
            break
        except Exception:                                 # noqa: BLE001
            continue
    if convert is None:
        return None

    for tool in ("op_profile^", "tensorflow_stats^"):
        try:
            data, _ = convert([xplane_path], tool, {})
        except Exception:                                 # noqa: BLE001
            continue
        if not data:
            continue
        try:
            parsed = json.loads(data) if isinstance(data, (str, bytes)) else data
        except Exception:                                 # noqa: BLE001
            continue
        rows = _rows_from_tool(parsed)
        if rows:
            return rows[:top]
    return None


def _rows_from_tool(parsed):
    """Pull (name, self_time_s) pairs out of whichever schema came back."""
    rows = []
    if isinstance(parsed, dict) and "byCategory" in parsed:
        node = parsed.get("byProgram") or parsed.get("byCategory")
        stack = [node]
        while stack:
            cur = stack.pop()
            if not isinstance(cur, dict):
                continue
            metrics = cur.get("metrics") or {}
            name = cur.get("name")
            t = metrics.get("time") or metrics.get("selfTime")
            if name and t:
                rows.append((name, float(t)))
            for child in cur.get("children") or []:
                stack.append(child)
    elif isinstance(parsed, list):
        for entry in parsed:
            if not isinstance(entry, dict):
                continue
            name = entry.get("operation") or entry.get("name")
            t = entry.get("selfTimeUs") or entry.get("self_time_us")
            if name and t:
                rows.append((name, float(t) * 1e-6))
    rows.sort(key=lambda r: -r[1])
    return rows


def via_trace_json(path, top):
    """Fall back to the chrome-trace events, which JAX can also emit.

    Sums duration per event name on the device lanes. Coarser than true self
    time (a parent op's duration includes its children), but it still ranks
    the expensive kernels, which is what the decision needs.
    """
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as fh:
        try:
            doc = json.load(fh)
        except Exception:                                 # noqa: BLE001
            return None
    events = doc.get("traceEvents", doc) if isinstance(doc, dict) else doc

    device_pids = set()
    for ev in events:
        if ev.get("ph") == "M" and ev.get("name") == "process_name":
            label = str((ev.get("args") or {}).get("name", ""))
            if any(tag in label for tag in ("TPU", "GPU", "device", "Device")):
                device_pids.add(ev.get("pid"))

    totals = defaultdict(float)
    counts = defaultdict(int)
    for ev in events:
        if ev.get("ph") != "X" or "dur" not in ev:
            continue
        if device_pids and ev.get("pid") not in device_pids:
            continue
        name = ev.get("name")
        if not name:
            continue
        totals[name] += float(ev["dur"]) * 1e-6
        counts[name] += 1
    if not totals:
        return None
    rows = sorted(((n, t, counts[n]) for n, t in totals.items()),
                  key=lambda r: -r[1])
    return [(n, t) for n, t, _ in rows[:top]]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("trace_dir")
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    rows = None
    xplane = find_xplane(args.trace_dir)
    if xplane:
        print(f"xplane: {xplane}")
        rows = via_plugin(xplane, args.top)

    if rows is None:
        tj = find_trace_json(args.trace_dir)
        if tj:
            print(f"chrome trace: {tj}")
            rows = via_trace_json(tj, args.top)

    if not rows:
        print("Could not reduce the trace: no usable converter and no chrome "
              "trace found.\nInstall the plugin with "
              "'pip install tensorboard-plugin-profile' and re-run, or read "
              "the primitive timings from tpu_bench_mrx.py instead.",
              file=sys.stderr)
        return 2

    total = sum(t for _, t in rows)
    print()
    print(f"{'op':<52}{'time (s)':>12}{'share':>9}")
    print("-" * 73)
    for name, t in rows:
        share = 100.0 * t / total if total else 0.0
        print(f"{name[:52]:<52}{t:>12.4f}{share:>8.1f}%")
    print("-" * 73)
    print(f"{'sum of shown rows':<52}{total:>12.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
