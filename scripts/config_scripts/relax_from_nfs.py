# %%
"""
Relaxation script using Hydra for configuration management.

Usage: 
    # Single run with defaults
    python scripts/config_scripts/relax_from_nfs.py
    
    # Override parameters
    python scripts/config_scripts/relax_from_nfs.py fem.ns_r=16 fem.ns_theta=32
    
    # Multirun sweep
    python scripts/config_scripts/relax_from_nfs.py -m fem.ns_r=8,12,16 eta.max=1e-6,1e-7

    # With custom run name
    python scripts/config_scripts/relax_from_nfs.py run_name=my_experiment
"""
import time
from pathlib import Path

import h5py
import hydra
import jax
import jax.numpy as jnp
from omegaconf import DictConfig, OmegaConf

from mrx.derham_sequence import DeRhamSequence
from mrx.differential_forms import DiscreteFunction, Pushforward
from mrx.io import interpolate_B, interpolate_map_from_points, unique_id
from mrx.relaxation import (DescentMethod, IntegrationScheme, MRXDiagnostics,
                            TimeStepChoice, TimeStepper, relaxation_loop)
from mrx.utils import default_trace_dict

jax.config.update("jax_enable_x64", True)


def create_eta_schedule(cfg: DictConfig):
    """Create resistivity schedule function based on config."""
    eta_max = cfg.eta.max
    num_iters_outer = cfg.relaxation.num_iters_outer
    schedule_type = cfg.eta.schedule_type

    if schedule_type == "tanh":
        # drops from eta_max to ~0 over the middle ~1/3rd of the iterations
        def eta_schedule(iter_outer):
            return eta_max * 0.5 * (1 - jnp.tanh(4 * jnp.pi * (iter_outer / num_iters_outer - 0.5)))
    elif schedule_type == "constant":
        def eta_schedule(iter_outer):
            return eta_max
    elif schedule_type == "linear":
        def eta_schedule(iter_outer):
            return eta_max * (1 - iter_outer / num_iters_outer)
    else:
        raise ValueError(f"Unknown eta.schedule_type: {schedule_type}")

    return eta_schedule


def create_noise_schedule(cfg: DictConfig):
    """Create resistivity schedule function based on config."""
    eta_max = cfg.noise.max
    num_iters_outer = cfg.relaxation.num_iters_outer
    schedule_type = cfg.noise.schedule_type

    if schedule_type == "tanh":
        # drops from eta_max to ~0 over the middle ~1/3rd of the iterations
        def eta_schedule(iter_outer):
            return eta_max * 0.5 * (1 - jnp.tanh(4 * jnp.pi * (iter_outer / num_iters_outer - 0.5)))
    elif schedule_type == "constant":
        def eta_schedule(iter_outer):
            return eta_max
    elif schedule_type == "linear":
        def eta_schedule(iter_outer):
            return eta_max * (1 - iter_outer / num_iters_outer)
    else:
        raise ValueError(f"Unknown noise.schedule_type: {schedule_type}")

    return eta_schedule


def create_hdf5_callback(seq, diagnostics, nfp, cfg: DictConfig, outdir: Path):
    """
    Create a callback function to save intermediate results to HDF5 after each outer iteration.

    Args:
        seq (DeRhamSequence):
        diagnostics (MRXDiagnostics): _description_
        nfp (int): _description_
        cfg (DictConfig): _description_
        outdir (Path): _description_
    """
    get_pressure = jax.jit(diagnostics.pressure)

    def hdf5_callback(state, iteration):
        if iteration % cfg.output.save_every != 0:
            return state

        print(
            f"  [Callback] Saving intermediate results to HDF5 at iteration {iteration}...")

        B_dof = state.B_n
        p_dof = get_pressure(B_dof)

        output_file = outdir / "intermediate_states.h5"
        # Use a group per iteration
        with h5py.File(output_file, "a") as f:
            group = f.create_group(f"iter_{iteration:05d}")
            group.create_dataset("B_dof", data=B_dof)
            group.create_dataset("p_dof", data=p_dof)
            group.attrs["iteration"] = iteration
            group.attrs["force_norm"] = float(state.F_norm)

        print(
            f"  [Callback] Saved intermediate results to {output_file} (group iter_{iteration:05d})")

        return state

    return hdf5_callback


@hydra.main(version_base=None, config_path="../../conf", config_name="config_relax_from_nfs")
def main(cfg: DictConfig) -> float:
    """
    Main entry point for relaxation with Hydra configuration.

    Returns the final force norm (useful for Hydra optimization).
    """
    # Hydra automatically changes to the output directory
    outdir = Path.cwd()

    # Generate run name if not provided
    run_name = cfg.run_name if cfg.run_name else unique_id(8)
    cfg.run_name = run_name  # Update config with the actual run name used

    print("=" * 60)
    print(f"Relaxation: {run_name}")
    print("=" * 60)
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60)

    start_time = time.time()

    # Initialize trace dictionary
    trace_dict = default_trace_dict.copy()
    trace_dict["start_time"] = start_time

    # Load NFS data
    # Use original working directory for data files
    original_cwd = hydra.utils.get_original_cwd()
    nfs_file = Path(original_cwd) / cfg.nfs_file

    print(f"Loading NFS data from {nfs_file}...")
    # read results file
    with h5py.File(nfs_file, "r") as f:
        pts = jnp.array(f["eval_points"])
        R = jnp.array(f["R"])
        Z = jnp.array(f["Z"])
        B_vals = jnp.array(f["B"])

    # Map interpolation
    ns_map = (cfg.map.ns_r, cfg.map.ns_theta, cfg.map.ns_zeta)
    ps_map = (cfg.map.ps_r, cfg.map.ps_theta, cfg.map.ps_zeta)

    # Validate that ps_map_x <= ns_map_x - 1
    ps_map = tuple(min(p, n - 1) for p, n in zip(ps_map, ns_map))
    nfp = cfg.nfp

    print("Interpolating map...")
    map_func, R_dof, Z_dof, map_resid = interpolate_map_from_points(
        pts, R, Z, nfp, ns=ns_map, ps=ps_map,
        quad_order=cfg.map.quad_order, flip_zeta=cfg.map.flip_zeta
    )
    map_func = jax.jit(map_func)
    print(
        f"Map interpolation residuals: {map_resid[0]:.2e}, {map_resid[1]:.2e}")

    # Setup FEM spaces
    ns = (cfg.fem.ns_r, cfg.fem.ns_theta, cfg.fem.ns_zeta)
    ps = (cfg.fem.ps_r, cfg.fem.ps_theta, cfg.fem.ps_zeta)

    # Validate that ps_x <= ns_x - 1
    ps = tuple(min(p, n - 1) for p, n in zip(ps, ns))

    print(f"Setting up FEM spaces with ns={ns}, ps={ps}...")
    seq = DeRhamSequence(
        ns, ps, cfg.fem.quad_order,
        ("clamped", "periodic", "periodic"),
        map_func, polar=True, dirichlet=True
    )

    assert jnp.min(seq.J_j) > 0, "Negative Jacobian!"

    seq.evaluate_1d()
    seq.assemble_all()
    seq.build_crossproduct_projections()
    seq.assemble_leray_projection()

    setup_time = time.time()
    trace_dict["setup_done_time"] = setup_time
    print(f"Setup completed in {setup_time - start_time:.2f}s")
    print(
        f"Minimum Jacobian: {jnp.min(seq.J_j):.2e}, Maximum Jacobian: {jnp.max(seq.J_j):.2e}")

    # B-field interpolation with train/validation split
    val_stride = cfg.interpolation.val_stride
    exclude_axis_tol = cfg.interpolation.exclude_axis_tol

    val_mask = (
        (jnp.arange(pts.shape[0]) % val_stride == 0) &
        (pts[:, 0] > exclude_axis_tol) &
        (pts[:, 0] < 1 - exclude_axis_tol)
    )
    train_mask = ~val_mask
    train_pts = pts[train_mask]
    train_B_vals = B_vals[train_mask]

    print("Interpolating B-field...")
    B_dof_0, resid_B = interpolate_B(
        train_pts, train_B_vals, seq, exclude_axis_tol=exclude_axis_tol
    )
    print(f"B-field interpolation residual (train): {resid_B[0]:.2e}")

    # Validate interpolation
    B_h = jax.jit(Pushforward(DiscreteFunction(
        B_dof_0, seq.Lambda_2, seq.E2), seq.F, 2))
    B_val_interp = jax.vmap(B_h)(pts[val_mask])
    val_error = jnp.linalg.norm(B_vals[val_mask] - B_val_interp, axis=1)
    val_rel_error = val_error / jnp.linalg.norm(B_vals[val_mask], axis=1)
    mean_val_error = float(jnp.mean(val_rel_error))
    max_val_error = float(jnp.max(val_rel_error))
    print(
        f"B-field validation error: mean={mean_val_error:.2e}, max={max_val_error:.2e}")

    div_B_initial = float(((seq.strong_div @ B_dof_0) @
                          seq.M3 @ (seq.strong_div @ B_dof_0))**0.5)
    print(f"div B after interpolation: {div_B_initial:.2e}")

    # Project to divergence-free and normalize
    B_dof_0 = seq.P_Leray @ B_dof_0
    B_dof_0 /= (B_dof_0 @ seq.M2 @ B_dof_0)**0.5
    B_dof = B_dof_0.copy()

    # Setup relaxation
    print("Setting up time stepper...")
    ts = TimeStepper(
        seq=seq,
        descent_method=DescentMethod[cfg.relaxation.descent_method.upper()],
        dt_mode=TimeStepChoice.ANALYTIC_LINESEARCH,
        timestep_mode=IntegrationScheme.EXPLICIT,
    )

    eta_schedule = create_eta_schedule(cfg)
    noise_schedule = create_noise_schedule(cfg)

    # Setup diagnostics (needed for callback)
    diagnostics = MRXDiagnostics(seq)

    # Setup fieldline callback if enabled
    callback = None

    if cfg.output.save_every > 0:
        print(
            f"HDF5 callback enabled: saving every {cfg.output.save_every} iterations")
        callback = create_hdf5_callback(seq, diagnostics, nfp, cfg, outdir)

    # Run relaxation loop
    print(
        f"Starting relaxation: {cfg.relaxation.num_iters_outer} outer x {cfg.relaxation.num_iters_inner} inner iterations...")

    final_state, traces = relaxation_loop(
        B_dof,
        ts,
        num_iters_inner=cfg.relaxation.num_iters_inner,
        num_iters_outer=cfg.relaxation.num_iters_outer,
        dt0=cfg.relaxation.dt0,
        force_tolerance=cfg.relaxation.force_tolerance,
        resistivity_schedule=eta_schedule,
        noise_schedule=noise_schedule,
        key=jax.random.PRNGKey(cfg.noise.key),
        callback=callback,
    )

    end_time = time.time()
    trace_dict["end_time"] = end_time

    # Final diagnostics
    B_dof_final = final_state.B_n

    final_force = float(traces["force_norm"][-1])
    final_energy = float(traces["energy"][-1])
    final_helicity = float(traces["helicity"][-1])
    helicity_change = abs(
        (final_helicity - traces["helicity"][0]) / traces["helicity"][0])

    print(f"\n{'=' * 60}")
    print("RESULTS")
    print("=" * 60)
    print(f"Final force norm: {final_force:.2e}")
    print(f"Final energy: {final_energy:.6f}")
    print(f"Relative helicity change: {helicity_change:.2e}")
    print(f"Total time: {end_time - start_time:.2f}s")
    print("=" * 60)

    # Save results
    if cfg.output.save_final:
        output_file = outdir / f"{run_name}.h5"
        print(f"Saving results to {output_file}...")
        with h5py.File(output_file, "w") as f:
            f.create_dataset("B_dof_initial", data=B_dof_0)
            f.create_dataset("B_dof_final", data=B_dof_final)
            f.create_dataset(
                "force_trace", data=jnp.array(traces["force_norm"]))
            f.create_dataset("energy_trace", data=jnp.array(traces["energy"]))
            f.create_dataset("helicity_trace",
                             data=jnp.array(traces["helicity"]))
            f.create_dataset("timestep_trace",
                             data=jnp.array(traces["timestep"]))
            f.create_dataset("eta_trace", data=jnp.array(traces["eta"]))
            f.create_dataset("iteration_trace",
                             data=jnp.array(traces["iteration"]))
            f.create_dataset("R_dof", data=R_dof)
            f.create_dataset("Z_dof", data=Z_dof)
            # Save flattened config
            f.attrs["config"] = OmegaConf.to_yaml(cfg)

    print("\nDone!")

    # Return final force norm for Hydra optimization
    return final_force


if __name__ == "__main__":
    main()
