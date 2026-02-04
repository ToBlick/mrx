# %%
"""
Interactive script to load and visualize results from relaxation simulations.

This script can handle both single runs and multirun (sweep) outputs from Hydra.
It loads the HDF5 results files and uses the trace_plot function from mrx.plotting.
"""
from pathlib import Path

import h5py
import numpy as np

from mrx.plotting import plot_twin_axis

# %%
# Configuration - set your output directory here
# For a single run:
# base_dir = Path("out/gvec_relaxation/20260203_120000")
# For a multirun:
# base_dir = Path("out/gvec_relaxation/multirun/20260203_120000")

base_dir = Path("out/gvec_relaxation/20260204_140136")

# %%
def find_result_files(base_dir: Path) -> list[Path]:
    """
    Find all HDF5 result files in a directory (handles both single and multirun).
    
    Looks for files matching *.h5 that contain trace data (not fieldlines_*.h5).
    """
    base_dir = Path(base_dir)
    
    # Check if this is a multirun directory (has numbered subdirectories)
    subdirs = [d for d in base_dir.iterdir() if d.is_dir() and d.name.isdigit()]
    
    if subdirs:
        # Multirun case - look in each subdirectory
        result_files = []
        for subdir in sorted(subdirs, key=lambda x: int(x.name)):
            h5_files = list(subdir.glob("*.h5"))
            # Filter out fieldline files
            for f in h5_files:
                if not f.name.startswith("fieldlines"):
                    result_files.append(f)
        return result_files
    else:
        # Single run case
        h5_files = list(base_dir.glob("*.h5"))
        return [f for f in h5_files if not f.name.startswith("fieldlines")]

# %%
def load_results(filepath: Path) -> dict:
    """Load results from an HDF5 file."""
    results = {}
    with h5py.File(filepath, "r") as f:
        # Load datasets
        for key in f.keys():
            results[key] = np.array(f[key])
        
        # Load attributes
        results["attrs"] = dict(f.attrs)
        
        # Try to parse config if present
        if "config" in f.attrs:
            results["config_str"] = f.attrs["config"]
    
    return results

# %%
def extract_param_from_config(config_str: str, param_path: str):
    """Extract a parameter value from a YAML config string."""
    import yaml
    try:
        config = yaml.safe_load(config_str)
        keys = param_path.split(".")
        value = config
        for key in keys:
            value = value[key]
        return value
    except:
        return None

# %%
def convert_to_trace_dict(results: dict) -> dict:
    """
    Convert loaded HDF5 results to the trace_dict format expected by trace_plot.
    
    The trace_plot function expects:
    - iterations: array of iteration numbers
    - force_trace: force norm at each iteration
    - helicity_trace: helicity at each iteration
    - divergence_trace: divergence at each iteration
    - velocity_trace: velocity norm at each iteration
    - wall_time_trace: wall time at each iteration
    - energy_trace (optional): energy at each iteration
    """
    n_iters = len(results.get("force_trace", []))
    
    trace_dict = {
        "iterations": np.arange(n_iters),
        "force_trace": results.get("force_trace", np.zeros(n_iters)),
        "helicity_trace": results.get("helicity_trace", np.zeros(n_iters)),
        # Use zeros for missing traces
        "divergence_trace": results.get("divergence_trace", np.zeros(n_iters)),
        "velocity_trace": results.get("velocity_trace", np.ones(n_iters)),
        "wall_time_trace": results.get("wall_time_trace", np.arange(n_iters, dtype=float)),
    }
    
    # Energy trace is optional
    if "energy_trace" in results:
        trace_dict["energy_trace"] = results["energy_trace"]
    
    return trace_dict

# %%
# Find and load all result files
result_files = find_result_files(base_dir)
print(f"Found {len(result_files)} result file(s):")
for f in result_files:
    print(f"  {f}")

# %%
# Load all results
all_results = {}
for filepath in result_files:
    run_name = filepath.stem
    # For multirun, include the job number in the name
    if filepath.parent.name.isdigit():
        run_name = f"job_{filepath.parent.name}_{run_name}"
    all_results[run_name] = load_results(filepath)
    all_results[run_name]["filepath"] = filepath
    print(f"Loaded: {run_name}")

# %%
# Display what's in the first result file
if all_results:
    first_key = list(all_results.keys())[0]
    first_result = all_results[first_key]
    print(f"\nContents of {first_key}:")
    print(f"  Datasets: {[k for k in first_result.keys() if k not in ['attrs', 'config_str', 'filepath']]}")
    print(f"  Attributes: {list(first_result.get('attrs', {}).keys())}")

# %%
# Generate plots using mrx.plotting.plot_twin_axis
for run_name, results in all_results.items():
    print(f"\nGenerating plots for {run_name}...")
    
    # Get output directory from filepath
    outdir = results["filepath"].parent
    
    # Extract num_iters_inner from config (default to 1 if not found)
    num_iters_inner = 1
    if "config_str" in results:
        val = extract_param_from_config(results["config_str"], "relaxation.num_iters_inner")
        if val is not None:
            num_iters_inner = int(val)
    print(f"  num_iters_inner: {num_iters_inner}")
    
    # Plot 1: Force vs time step
    force_trace = results.get("force_trace", np.array([]))
    timestep_trace = results.get("timestep_trace", np.array([]))
    if len(force_trace) > 0:
        iterations = np.arange(len(force_trace)) * num_iters_inner
        timesteps = timestep_trace
        fig, (ax1, ax2) = plot_twin_axis(
            left_y=force_trace,
            right_y=timestep_trace,
            x_left=iterations,
            x_right=iterations,
            left_label="Force",
            right_label="Time step",
            left_log=True,
            right_log=False,
            x_label="Iterations",
            return_axes=True,
            left_marker="",
            right_marker="",
        )
        fig.savefig(outdir / "force_vs_timestep.pdf", dpi=150, bbox_inches="tight")
        print(f"  Saved force_vs_timestep.pdf")
    
    # Plot 2: Force vs helicity
    helicity_trace = results.get("helicity_trace", np.array([]))
    if len(force_trace) > 0 and len(helicity_trace) > 0:
        iterations = np.arange(len(helicity_trace)) * num_iters_inner
        fig, (ax1, ax2) = plot_twin_axis(
            left_y=force_trace,
            right_y=(helicity_trace - helicity_trace[0]) / np.abs(helicity_trace[0]),
            x_left=iterations,
            x_right=iterations,
            left_label="Force",
            right_label="rel. Helicity change",
            left_log=True,
            right_log=False,
            x_label="Iterations",
            return_axes=True,
            left_marker="",
            right_marker="",
        )
        fig.savefig(outdir / "force_vs_helicity.pdf", dpi=150, bbox_inches="tight")
        print(f"  Saved force_vs_helicity.pdf")
# %%

# %%
def print_summary(results_dict: dict):
    """Print a summary table of all runs."""
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Run':<40} {'Final Force':<15} {'Final Energy':<15} {'ΔH/H₀':<15}")
    print("-" * 80)
    
    for run_name, results in results_dict.items():
        force_final = results.get("force_trace", [np.nan])[-1]
        energy_final = results.get("energy_trace", [np.nan])[-1]
        
        helicity = results.get("helicity_trace", None)
        if helicity is not None and len(helicity) > 0:
            dH_rel = abs(helicity[-1] - helicity[0]) / abs(helicity[0])
        else:
            dH_rel = np.nan
        
        print(f"{run_name:<40} {force_final:<15.2e} {energy_final:<15.6f} {dH_rel:<15.2e}")
    
    print("=" * 80)

# %%
if all_results:
    print_summary(all_results)

# %%
# Example: Compare runs by a specific parameter
# Uncomment and modify as needed:

# param_to_compare = "fem.ns_r"  # Change this to the parameter you want to compare
# 
# for run_name, results in all_results.items():
#     if "config_str" in results:
#         param_value = extract_param_from_config(results["config_str"], param_to_compare)
#         final_force = results.get("force_trace", [np.nan])[-1]
#         print(f"{run_name}: {param_to_compare}={param_value}, final_force={final_force:.2e}")
