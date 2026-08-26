#!/bin/bash
# RUN FROM THE REPO ROOT.
#
# w7x-ini-conv: the CONVERGED W7-X equilibrium, ~20 GPU-h of hyperparameters.
#
# WHY THIS CASE IS DIFFERENT FROM EVERYTHING BEFORE IT
# ----------------------------------------------------
# w7x-ini-conv and w7x-ini-clebsch come from the SAME GVEC run:
# State_0000_00020000.dat (20000 iterations, converged) against
# _00000000.dat (iteration 0, GVEC's initial GUESS).  Every w7x_ini result in
# the campaign so far was measured on the guess.
#
# That matters for one diagnostic in particular.  The scheme's fixed point is
# J x B = grad p, so "does the relaxed pressure profile approach the file's?"
# is only a meaningful question when the file HAS an equilibrium profile to
# approach.  On the initial guess it did not, which is why handoff s34.1 could
# not read that diagnostic.  Here it should be readable, and it is the reason
# every arm below carries --helicity-every 250: the pressure comparison is
# sampled on the same cadence.
#
# HOW THE ARMS ARE RANKED
# -----------------------
# On |dH| per unit ENERGY REMOVED, absolute rather than relative.  Two reasons,
# both learned the hard way:
#   * relative |dH|/H divides by a quantity that COLLAPSES and even changes
#     sign under refinement on a near-harmonic field (s35), and
#   * per-dE is budget-independent, so arms that remove energy at very
#     different rates -- which fixed-dt and linesearch arms do -- stay
#     comparable without matching step counts.
# w7x_ini's current-driven fraction was stable to three digits (s31.2 as
# corrected), so this case should be sound for the relative form too; both are
# recorded and the first job's IC output will say.
#
# NOT FLOOR RUNS.  Every arm stops on a step budget because relax_prelim.py
# has no convergence criterion (handoff P0).  These compare hyperparameters at
# a FIXED budget, which is a real question and not the same one as "where does
# it floor".  Nothing here supports an h- or p-refinement claim.
#
# BUDGET, re-costed against a 20-step smoke run on THIS geometry (16796455).
#
# The smoke reported 6.40 s/step, which is JIT and not cost: steps 10 -> 20
# took 10 s, so the steady state is ~1.0 s/step at 8^3 and the first ~90 s of
# any arm is compilation.  Budget on the steady rate plus a fixed ~90 s, not
# on the average of a short run -- averaging a 20-step job overstates a
# 3000-step one by 6x.  Add ~270 s for the two Poincare traces and ~100 s of
# setup per job; gamma=1 costs ~1.4x per step (measured, s33).
#
#   dt bracket        4 arms   3.8 h
#   mu at gamma=1     3 arms   3.9 h
#   levers composed   1 arm    1.3 h
#   resolution 12^3   2 arms   5.9 h
#   resistivity       1 arm    1.0 h
#   length            1 arm    2.9 h
#                             ------
#                             18.8 h   against the 20 h allocated
#
# TRIMMED to fit: a fourth mu arm at 1e-5, below M1's suspected optimum, which
# nothing in the campaign has sampled.  All 13 arms came to 20.1 h, and going
# over an allocation to buy the least-critical arm is the wrong trade -- the
# 1e-4/1e-3/1e-2 arms already bracket the optimum from above.  It stays on the
# shelf as A2.
set -u
G="--geometry w7x-ini-conv --ic clebsch --p 3"
PC="--poincare --pc-seeds 40 --pc-periods 150"
O=/scratch/tblickhan/mrx/out/relax_prelim
S=slurm/job_relax_prelim.sh

# sub <tag> <slurm-time> <seconds-per-arm> <args...>
#
# seconds-per-arm is deliberately set BELOW the slurm walltime with room for
# setup and Poincare.  S05/S06 were truncated mid-campaign by the opposite
# mistake -- a 9000 s arm budget inside a 4 h allocation -- and lost 1h20m of
# allocated GPU each.
sub () {
  tag=$1; wall=$2; budget=$3; shift 3
  mkdir -p "$O/$tag"
  # shellcheck disable=SC2086
  jid=$(sbatch --time="$wall" "$S" "$@" --helicity-every 250 \
        --seconds-per-arm "$budget" --arms cg $PC \
        --save-b "$O/$tag/B.h5" --out "$O/$tag/$tag.json" | awk '{print $4}')
  echo "$tag -> $jid"
  ln -sfn "/scratch/tblickhan/mrx/logs/relaxprelim_$jid.out" \
     "$O/logs/live_$tag.out"
}

# --- dt bracket -------------------------------------------------------------
# The step size is the largest lever in the study and the only free one (73x
# on w7x_ini, at no cost in force reduction).  C3 and C4 fill the decade
# between 3e-3 and the linesearch's ~3e-2 that no arm has ever sampled --
# handoff shelf item B3 -- so the cliff can be located rather than bracketed.
sub C1_ls        2:00:00  5400 $G --ns 8,16,8 --steps 3000
sub C2_dt3e3     2:00:00  5400 $G --ns 8,16,8 --steps 3000 --dt-mode fixed --dt0 3e-3
sub C3_dt1e2     2:00:00  5400 $G --ns 8,16,8 --steps 3000 --dt-mode fixed --dt0 1e-2
sub C4_dt3e2     2:00:00  5400 $G --ns 8,16,8 --steps 3000 --dt-mode fixed --dt0 3e-2

# --- hyperregularisation ----------------------------------------------------
# M1 (gamma=1, mu=1e-4) gave 41x less helicity loss per unit energy than
# gamma=0 on fmm002, for 1.4x the cost per step -- the largest non-free lever
# found.  It is ONE interior sample and the trend in mu is non-monotone, so
# these three re-test it on a different case and bracket it from above.  The
# below-side point (mu=1e-5) is the arm trimmed for budget; see the header.
sub C5_mu1e4     2:30:00  7200 $G --ns 8,16,8 --steps 3000 --gamma 1 --mu 1e-4
sub C6_mu1e3     2:30:00  7200 $G --ns 8,16,8 --steps 3000 --gamma 1 --mu 1e-3
sub C7_mu1e2     2:30:00  7200 $G --ns 8,16,8 --steps 3000 --gamma 1 --mu 1e-2

# --- do the two best levers COMPOSE? ---------------------------------------
# Capping dt and smoothing with mu both suppress reconnection, and nothing in
# the campaign has run them together.  They might compose, or the cap might
# already remove what mu was fixing.  Either answer is worth one arm.
sub C8_mu1e4_dt3e3 2:30:00 7200 $G --ns 8,16,8 --steps 3000 --gamma 1 --mu 1e-4 \
    --dt-mode fixed --dt0 3e-3

# --- resolution -------------------------------------------------------------
# NOT a refinement claim (P0/s32: these stop on a budget, not at a floor).
# They are here because 12^3 gave visibly better surfaces on w7x_ini and the
# Poincare pair is worth having on the converged case.
sub C9_r12_ls    5:00:00 14400 $G --ns 12,24,12 --steps 3000
sub C10_r12_dt3e3 5:00:00 14400 $G --ns 12,24,12 --steps 3000 --dt-mode fixed --dt0 3e-3

# --- resistivity ------------------------------------------------------------
# eta relaxes the frozen-flux constraint and should let the force fall further.
# S10 showed eta=1e-2 reaches a stationary energy in ~500 steps and then spends
# 87% of the job at round-off, so this uses the smaller eta and the tanh ramp.
sub C11_eta3     2:00:00  5400 $G --ns 8,16,8 --steps 3000 --eta-max 1e-3 \
    --eta-schedule tanh

# --- length -----------------------------------------------------------------
# Same config as C2 at 3.3x the steps.  On fmm002 the pressure-shape residual
# TURNED AROUND between 3000 and 13018 steps (s34.1) -- toward the reference,
# then away.  This is the paired arm that says whether that happens on a
# converged equilibrium too.
sub C12_dt3e3_long 5:00:00 14400 $G --ns 8,16,8 --steps 10000 \
    --dt-mode fixed --dt0 3e-3

echo "submitted 12 arms, ~18.8 GPU-h estimated"
