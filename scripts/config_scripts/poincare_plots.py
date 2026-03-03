"""
Poincaré plot generation script using Hydra for configuration management.

Loads intermediate states from a completed GVEC relaxation run and generates
Poincaré section plots for each saved iteration.

Usage:
    # Single run (run_dir is required)
    python scripts/config_scripts/poincare_plots.py run_dir=out/relax_from_nfs/20260206_072421

    # Override plotting parameters
    python scripts/config_scripts/poincare_plots.py run_dir=... plotting.dpi=300 fieldline.n_vmap=32

    # Override via SLURM
    sbatch slurm/job_poincare.sh "run_dir=out/relax_from_nfs/20260206_072421"
"""
import time
from pathlib import Path

import h5py
import hydra
import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import yaml
from omegaconf import DictConfig, OmegaConf

from mrx.derham_sequence import DeRhamSequence
from mrx.differential_forms import DiscreteFunction
from mrx.io import unique_id
from mrx.mappings import stellarator_map
from mrx.plotting import (get_iota_log, get_periodic_intersections,
                          integrate_fieldlines, poincare_plot)

jax.config.update("jax_enable_x64", True)
matplotlib.use("Agg")  # Non-interactive backend for batch jobs


def load_relaxation_config(run_dir: Path) -> DictConfig:
    """Load the Hydra config from a completed relaxation run."""
    config_path = run_dir / ".hydra" / "config.yaml"
    with open(config_path, "r") as f:
        return OmegaConf.create(yaml.safe_load(f))


def load_results(run_dir: Path, run_name: str) -> dict:
    """Load final results from the relaxation HDF5 file."""
    results = {}
    results_file = run_dir / f"{run_name}.h5"
    with h5py.File(results_file, "r") as f:
        for key in f.keys():
            results[key] = np.array(f[key])
    return results


def iter_traces(trace_file: Path):
    """Iterate over intermediate states from the relaxation run."""
    with h5py.File(trace_file, "r") as f:
        for group_name in sorted(f.keys()):
            group = f[group_name]
            yield {
                "B_dof": np.array(group["B_dof"]),
                "p_dof": np.array(group["p_dof"]),
                "iteration": group.attrs["iteration"],
                "force_norm": group.attrs["force_norm"],
            }


@hydra.main(version_base=None, config_path="../../conf", config_name="config_poincare")
def main(cfg: DictConfig) -> None:
    """
    Main entry point for Poincaré plot generation with Hydra configuration.
    """
    if cfg.run_dir is None:
        raise ValueError("run_dir must be specified. "
                         "Example: python poincare_plots.py run_dir=out/relax_from_nfs/20260206_072421")

    original_cwd = hydra.utils.get_original_cwd()
    run_dir = Path(original_cwd) / cfg.run_dir

    print("=" * 60)
    print("Poincaré Plot Generation")
    print("=" * 60)
    print(f"\nRun directory: {run_dir}")
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60)

    start_time = time.time()

    # ── Load relaxation config ───────────────────────────────────────────
    relax_cfg = load_relaxation_config(run_dir)
    run_name = relax_cfg.get("run_name")
    if run_name is None:
        # Find the h5 file in run_dir that's not intermediate_states.h5
        h5_files = [f for f in run_dir.glob(
            "*.h5") if f.name != "intermediate_states.h5"]
        if len(h5_files) == 0:
            raise FileNotFoundError(f"No results h5 file found in {run_dir}")
        elif len(h5_files) > 1:
            raise ValueError(
                f"Multiple h5 files found in {run_dir}: {[f.name for f in h5_files]}")
        run_name = h5_files[0].stem  # Get filename without extension

    nfp = relax_cfg.nfp

    # ── Load results ─────────────────────────────────────────────────────
    results = load_results(run_dir, run_name)

    # ── Output directory ─────────────────────────────────────────────────
    outdir = run_dir / cfg.output.subdir
    outdir.mkdir(exist_ok=True, parents=True)
    print(f"Output directory: {outdir}")

    # ── Build the map ────────────────────────────────────────────────────
    ns_map = (relax_cfg.map.ns_r, relax_cfg.map.ns_theta,
              relax_cfg.map.ns_zeta)
    ps_map = (relax_cfg.map.ps_r, relax_cfg.map.ps_theta,
              relax_cfg.map.ps_zeta)
    quad_map = relax_cfg.map.quad_order

    map_seq = DeRhamSequence(
        ns_map, ps_map, quad_map,
        ("clamped", "periodic", "periodic"),
        lambda x: x, polar=False, dirichlet=False,
    )

    R_dof = results["R_dof"]
    Z_dof = results["Z_dof"]
    X1_h = DiscreteFunction(R_dof, map_seq.basis_0, map_seq.e0)
    X2_h = DiscreteFunction(Z_dof, map_seq.basis_0, map_seq.e0)
    map_func = jax.jit(stellarator_map(X1_h, X2_h, nfp=nfp,
                       flip_zeta=relax_cfg.map.flip_zeta))

    print("Map reconstructed.")

    # ── Build the FEM sequence ───────────────────────────────────────────
    ns = (relax_cfg.fem.ns_r, relax_cfg.fem.ns_theta, relax_cfg.fem.ns_zeta)
    ps = (relax_cfg.fem.ps_r, relax_cfg.fem.ps_theta, relax_cfg.fem.ps_zeta)

    seq = DeRhamSequence(
        ns, ps, relax_cfg.fem.quad_order,
        ("clamped", "periodic", "periodic"),
        map_func, polar=True, dirichlet=True,
    )

    print("FEM sequence built.")

    # ── Fieldline integration setup ──────────────────────────────────────
    n_scan = cfg.fieldline.n_scan
    n_vmap = cfg.fieldline.n_vmap
    n_traj = n_scan * n_vmap
    T = cfg.fieldline.T_factor * jnp.pi * 2 * nfp
    N = int(T)
    axis_margin = cfg.fieldline.axis_margin

    integrate_fieldlines_jit = jax.jit(
        integrate_fieldlines, static_argnames=["T", "N", "seq"]
    )
    get_periodic_intersections_jit = jax.jit(
        get_periodic_intersections, static_argnames=["max_intersections"]
    )

    r_vals = jnp.linspace(axis_margin, 0.99, n_traj)
    key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key)
    x2_vals = jax.random.uniform(key1, shape=(n_traj,), minval=0.0, maxval=1.0)
    x3_vals = jax.random.uniform(key2, shape=(n_traj,), minval=0.0, maxval=1.0)
    x0s = jnp.stack([r_vals, x2_vals, x3_vals], axis=1)
    x0_grid = x0s.reshape((n_scan, n_vmap, 3))

    # ── Trace fieldlines for each intermediate state ─────────────────────
    trace_file = run_dir / "intermediate_states.h5"
    if not trace_file.exists():
        raise FileNotFoundError(
            f"No intermediate_states.h5 found in {run_dir}")

    ks_thresh = cfg.poincare.ks_thresh
    zeta_values = list(cfg.poincare.zeta_values)
    max_intersections = int(T // (2 * nfp))

    plot_data = []
    print(
        f"\nIntegrating fieldlines (n_traj={n_traj}, T={float(T):.0f}, N={N}) ...")

    for trace in tqdm.tqdm(list(iter_traces(trace_file))):
        B_dof = trace["B_dof"]
        p_dof = trace["p_dof"]
        iteration = trace["iteration"]
        force_norm = trace["force_norm"]

        if cfg.output.verbose:
            print(f"  Iteration {iteration}, force norm: {force_norm:.4e}")

        logical_trajectories, p_values = integrate_fieldlines_jit(
            x0_grid, B_dof, p_dof, seq, T, N
        )
        logical_trajectories = logical_trajectories.reshape((n_traj, N, 3))
        p_values = p_values.reshape((n_traj, N))

        iotas, flags, ks = jax.vmap(
            lambda c: get_iota_log(c, nfp, ks_thresh=ks_thresh)
        )(logical_trajectories)

        for zeta_val in zeta_values:
            logical_intersections, p_at_intersections, counts = jax.vmap(
                lambda t, p: get_periodic_intersections_jit(
                    t,
                    plane_normal=jnp.array([0.0, 0.0, 1.0]),
                    plane_point=jnp.array([0.0, 0.0, zeta_val]),
                    p_values=p,
                    max_intersections=max_intersections,
                )
            )(logical_trajectories, p_values)

            physical_intersections = jax.vmap(map_func)(
                logical_intersections.reshape(-1, 3)
            ).reshape(logical_intersections.shape)

            plot_data.append({
                "iteration": iteration,
                "force_norm": force_norm,
                "zeta": zeta_val,
                "physical_intersections": physical_intersections,
                "logical_intersections": logical_intersections,
                "p_at_intersections": p_at_intersections,
                "iotas": iotas,
                "ks": ks,
                "counts": counts,
            })

    # ── Compute global colour limits ─────────────────────────────────────
    p_min = 0.0
    p_max = float(jnp.nanmax(
        jnp.array([d["p_at_intersections"] for d in plot_data])))
    iota_min = float(jnp.nanmin(jnp.array([d["iotas"] for d in plot_data])))
    iota_max = float(jnp.nanmax(jnp.array([d["iotas"] for d in plot_data])))

    # ── Generate and save plots ──────────────────────────────────────────
    dpi = cfg.plotting.dpi
    markersize = cfg.plotting.markersize
    denom_max = cfg.plotting.denom_max
    fig_format = cfg.output.format

    Rlim = tuple(cfg.plotting.Rlim) if cfg.plotting.Rlim is not None else None
    zlim = tuple(cfg.plotting.zlim) if cfg.plotting.zlim is not None else None

    print(f"\nGenerating {len(plot_data)} Poincaré plots ...")

    for i, data in enumerate(tqdm.tqdm(plot_data)):
        phys_intersec = data["physical_intersections"]
        log_intersec = data["logical_intersections"]
        iotas = data["iotas"]
        p_at_intersections = data["p_at_intersections"]

        # Convert to cylindrical coordinates
        R_vals = (phys_intersec[:, :, 0] ** 2 +
                  phys_intersec[:, :, 1] ** 2) ** 0.5
        phi_vals = jnp.arctan2(phys_intersec[:, :, 1], phys_intersec[:, :, 0])
        z_vals = phys_intersec[:, :, 2]
        cyl_intersections = jnp.stack([R_vals, phi_vals, z_vals], axis=-1)
        iota_values = jnp.broadcast_to(
            iotas[:, None], phys_intersec[:, :, 0].shape)

        auto_Rlim = Rlim if Rlim is not None else (
            float(jnp.nanmin(R_vals)), float(jnp.nanmax(R_vals)))
        auto_zlim = zlim if zlim is not None else (
            float(jnp.nanmin(z_vals)), float(jnp.nanmax(z_vals)))

        fig, axes = poincare_plot(
            log_intersec,
            cyl_intersections,
            p_at_intersections if cfg.plotting.plot_pressure else None,
            iota_values,
            nfp,
            cmap_iota=cfg.plotting.cmap_iota,
            cmap_p=cfg.plotting.cmap_p,
            markersize=markersize,
            denom_max=denom_max,
            Rlim=auto_Rlim,
            zlim=auto_zlim,
            p_lim=(p_min, p_max),
            iota_lim=(iota_min, iota_max),
        )

        zeta_tag = f"_zeta{data['zeta']:.2f}" if len(zeta_values) > 1 else ""
        filename = f"poincare_plot_{data['iteration']:05d}{zeta_tag}.{fig_format}"
        fig.savefig(outdir / filename, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    elapsed = time.time() - start_time
    print(f"\nDone! {len(plot_data)} plots saved to {outdir} ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
