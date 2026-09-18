"""The Cary-Hanson diagnostic on the seeded li383 (6,1) island: O/X points,
residues, the pendulum width against the section's max(r) - min(r) and the
seed's linear estimate, with the shear from the UNSEEDED field's section."""
import numpy as np

from mrx.geometry import build_sequence
from mrx.gvec import load_clebsch
from mrx.initial_conditions import clebsch_potential_form, potential_two_form, resonant_rho
from mrx.nullspace import compute_nullspaces
from mrx.poincare import islands, poincare

seq, _ = build_sequence("data/wout_li383_1.4m.nc", (10, 16, 16), 2)
compute_nullspaces(seq)
cb = load_clebsch(seq.equilibrium, nfp=seq.nfp)
m, n = 6, 1
target, rho_file = seq.nfp * n / m, resonant_rho(cb, m, n)

# the unperturbed shear: a section of the equilibrium field, a linear fit around the chain
B0, _, _ = potential_two_form(seq, clebsch_potential_form(cb, None))
res0 = poincare(seq, B0, lines=48, periods=200)
shown = np.asarray(res0["shown"])
r0, i0 = np.asarray(res0["seed_r"])[shown], np.asarray(res0["iota"])[shown]
near = np.abs(r0 - rho_file) <= 0.1
iota_prime = float(np.polyfit(r0[near], i0[near], 1)[0])
print(f"unseeded: chain at rho {rho_file:.4f}, iota' = {iota_prime:+.4f} from {int(near.sum())} lines", flush=True)

for eps in (3e-3, 1e-2):
    seed = (m, n, 0.544, 0.1, eps)
    B, _, _ = potential_two_form(seq, clebsch_potential_form(cb, seed))
    res = poincare(seq, B, lines=48, periods=200)
    fit = islands(seq, B, m, n, res)
    out = islands(seq, B, m, n, res, iota_prime=iota_prime)
    print(f"eps {eps:.0e}: chain at r = {out['r_chain']:.4f}, fitted iota' = {fit['iota_prime']:+.4f}", flush=True)
    for k in range(2):
        print(f"   {out['kind'][k]}: r {out['r'][k]:.4f} theta {out['theta'][k]:.4f} "
              f"residue {out['residue'][k]:+.5f} det {out['det'][k]:.6f} defect {out['defect'][k]:.1e} "
              f"width {out['width'][k]:.4f} (fitted shear: {fit['width'][k]:.4f})", flush=True)
    sec = res["sections"][0.0]
    locked = np.asarray(res["shown"]) & (np.abs(np.asarray(res["iota"]) - target) < 2e-3)
    lr = sec["logr"][locked]
    print(f"   section: {int(locked.sum())} locked lines, max(r) - min(r) = {float(lr.max() - lr.min()):.4f}; "
          f"seed estimate 1.6 sqrt(eps nfp / (m |iota'|)) = {1.6 * np.sqrt(eps * seq.nfp / (m * abs(iota_prime))):.4f}",
          flush=True)
