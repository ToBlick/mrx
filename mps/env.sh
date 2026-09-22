# Environment for running MRX on an Apple GPU.  Source it, do not run it:
#
#     source mps/env.sh
#     python scripts/relax.py --ns 12,24,12 --p 3 --chunk 25 ...
#
# Two variables are needed and the rest of this file is the record of what
# was measured and found not to help, so that nobody sweeps these knobs a
# second time.  The numbers are li383 (12,24,12) p=3 float32 on an M3 Pro,
# jax-mps 0.10.11; docs/research/mps_benchmark_2026-09-21.md has the detail.

# Metal has no float64 and MLX refuses the buffer rather than downcasting it
# the way XLA:TPU does, so 64-bit mode goes off at the root.  See
# docs/source/mps.md; this is required, not a tuning choice.
export MRX_X64=0

# The plugin registers at priority 500 against the CPU backend's 0, so it is
# already the default in an environment that has it installed.  Set it
# explicitly anyway: it is what makes the CPU comparison a one-word edit.
export JAX_PLATFORMS=mps

# MRX_ASSEMBLY is deliberately unset.  On Metal the mass kernel defaults to
# the indexed form (one gather and one segment_sum), which measured 2.5x on
# the apply and took a 100-step L-BFGS li383 relaxation from 593.9 s to
# 283.8 s, ahead of this machine's CPU at 352.5 s.  Every other backend keeps
# the shifted form.  The Newton matvec pins itself back to shift: on the
# tutorial mesh the indexed form changed the direction.  Set
# MRX_ASSEMBLY=shift to compare the two on the GPU for everything else.

# --- Not set, and why -------------------------------------------------------
#
# MLX_ENABLE_TF32
#     Never set this.  mrx/precision.py asks for
#     jax_default_matmul_precision='highest' because TF32 put a 19% error in
#     the W7-X map's dR/dtheta.  Speed bought by undoing that is not speed.
#     mps/env_sweep.py asserts it is off before it times anything.
#
# MLX_METAL_FAST_SYNCH
#     Broken on this build: every configuration that set it died with
#     "[metal::Device] Unable to load kernel input_coherent".
#
# MLX_MAX_OPS_PER_BUFFER, MLX_MAX_MB_PER_BUFFER
#     The most promising knobs on paper, since the cost here is per-dispatch.
#     Swept over 50 / 200 / 1000 and with the size bound raised to 1024 MB:
#     every result landed inside the +-17% run-to-run spread of the baseline
#     itself.  The defaults are already right.
#
# JAX_MPS_ASYNC_DISPATCH
#     The plugin advertises it at startup and it does help eager, one-apply-
#     at-a-time use roughly 2x.  It does nothing for a jitted lax.scan, which
#     is what the relaxation is: 603 ms/step against a 585 ms/step baseline,
#     i.e. inside the noise.
#
# JAX_MPS_NO_OPTIMIZE=1, MLX_DISABLE_COMPILE=1
#     Controls, not candidates, and both confirm the stack is already doing
#     its job: turning the plugin's IR passes off costs 1.28x and turning
#     MLX's kernel fusion off costs 2.0x.
#
# JAX_COMPILATION_CACHE_DIR / jax_compilation_cache_dir
#     Has no effect on this plugin: it writes zero cache entries and a second
#     run recompiles in full.  The same script on the CPU backend writes 1118
#     entries and recompiles 2.7x faster.  Every MPS run pays full compile.
#
# --- The one thing that does matter: keep the chunk small -------------------
#
# Because compile is uncacheable (above) and its cost is linear in the chunk
# -- about 7.9 s per step of chunk at this mesh -- while the steady per-step
# cost is flat in the chunk (mps/scan_scaling.py measured 5506.8 vs 5504.8
# ms/step at chunk 5 and 10), a long chunk is pure loss on this backend.
# scripts/relax.py defaults to chunk 500 under L-BFGS, which is about 65
# minutes of compile before the first step.  For a 500-step run:
#
#     --chunk 500    ~65 min compile + ~46 min stepping   ~111 min
#     --chunk 25     ~3 min compile  + ~46 min stepping   ~49 min
#
# Pass --chunk 25 explicitly.  It is the only configuration change in this
# file that was worth making, and it is worth about 2.3x.
#
# JAX_MPS_DUMP_OPTIMIZED_IR=<dir> is the instrument behind most of the above:
# it writes <dir>/module_*.mlir and is how the op census was taken.
