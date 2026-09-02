# MRX on TPU

- Timestamp: 2026-09-01T22:28:43+00:00
- Backend: tpu / TPU v5 lite (v5e)
- Devices: 4 (local 4)
- JAX: 0.11.1
- MRX dtype: float32 (eps 1.192e-07)
- x64 enabled: False
- Matmul precision: highest

## Toroid Poisson

| p | n | error | CG iters |
|---|---|---|---|
| 2 | 6 | 1.0754e-02 | 6 |
| 2 | 8 | 3.5617e-03 | 7 |

Wall clock: 106.3s

## Agreement with the CPU float32 reference

| p | n | measured | reference | deviation | ok |
|---|---|---|---|---|---|
| 2 | 6 | 1.0754e-02 | 1.0754e-02 | 0.00% | yes |
| 2 | 8 | 3.5617e-03 | 3.5617e-03 | 0.00% | yes |

## Aggregate chip throughput

4 chips, 4096^2 float32 matmul: **71.7 TFLOP/s**

The MRX solve is single-device matrix-free CG and uses one chip; this figure is the headroom available to a sharded implementation.
