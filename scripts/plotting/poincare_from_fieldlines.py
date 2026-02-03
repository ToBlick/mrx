"""
Create Poincare plots from fieldline HDF5 files.

This script processes fieldline data saved by relax_gvec.py and creates
Poincare plots showing the magnetic field structure.

Usage:
    # Single output directory
    python scripts/plotting/poincare_from_fieldlines.py /path/to/output_dir
    
    # Hydra multirun directory (processes all subdirectories)
    python scripts/plotting/poincare_from_fieldlines.py /path/to/multirun/20240101_120000
    
    # With options
    python scripts/plotting/poincare_from_fieldlines.py /path/to/output --zeta 0.0,0.25,0.5 --format pdf
    
    # Process only final fieldlines
    python scripts/plotting/poincare_from_fieldlines.py /path/to/output --final-only
"""
import argparse
from pathlib import Path
from typing import Optional, List
import sys

import h5py
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

# Add mrx to path if not installed
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mrx.plotting import intersect_with_plane_logical_periodic, get_iota_log

jax.config.update("jax_enable_x64", True)


def load_fieldlines(h5_path: Path) -> dict:
    """Load fieldline data from HDF5 file."""
    with h5py.File(h5_path, "r") as f:
        data = {
            "logical_trajectories": jnp.array(f["logical_trajectories"][:]),
            "physical_trajectories": jnp.array(f["physical_trajectories"][:]),
            "B_norm": jnp.array(f["B_norm"][:]),
            "pressure": jnp.array(f["pressure"][:]),
            "nfp": int(f.attrs["nfp"]),
            "T": float(f.attrs["T"]),
            "n_traj": int(f.attrs["n_traj"]),
        }
        # Optional attributes
        if "iteration" in f.attrs:
            data["iteration"] = int(f.attrs["iteration"])
        if "force_norm" in f.attrs:
            data["force_norm"] = float(f.attrs["force_norm"])
        if "final_force_norm" in f.attrs:
            data["final_force_norm"] = float(f.attrs["final_force_norm"])
        if "final_energy" in f.attrs:
            data["final_energy"] = float(f.attrs["final_energy"])
    return data


def create_poincare_plot(
    logical_trajectories: jnp.ndarray,
    physical_trajectories: jnp.ndarray,
    B_norm: jnp.ndarray,
    pressure: jnp.ndarray,
    nfp: int,
    zeta_value: float = 0.5,
    title: str = "",
    interpolation_degree: int = 3,
    cmap_iota: str = "berlin",
    cmap_p: str = "plasma",
    cmap_B: str = "viridis",
    markersize: float = 0.5,
    ks_thresh: int = 10,
    denom_max: int = 20,
    color_by: str = "iota",  # "iota", "pressure", "B_norm"
) -> plt.Figure:
    """
    Create a Poincare plot from pre-computed fieldline trajectories.
    
    Parameters
    ----------
    logical_trajectories : array (n_traj, n_points, 3)
        Trajectories in logical coordinates (r, theta, zeta)
    physical_trajectories : array (n_traj, n_points, 3)
        Trajectories in physical coordinates (x, y, z)
    B_norm : array (n_traj, n_points)
        |B| values along trajectories
    pressure : array (n_traj, n_points)
        Pressure values along trajectories
    nfp : int
        Number of field periods
    zeta_value : float
        Toroidal angle for the Poincare section (in [0, 1])
    title : str
        Plot title
    color_by : str
        What to color points by: "iota", "pressure", or "B_norm"
    
    Returns
    -------
    fig : matplotlib Figure
    """
    # Find intersections with the Poincare plane
    res = [
        intersect_with_plane_logical_periodic(
            traj, zeta_value=zeta_value, deg=interpolation_degree
        )
        for traj in logical_trajectories
    ]
    logical_intersections = jnp.array([r[0] for r in res])
    intersection_indices = jnp.array([r[1] for r in res])
    
    # Compute rotational transform (iota) for each trajectory
    iotas, flags, ks = jax.vmap(
        lambda c: get_iota_log(c, nfp, ks_thresh=ks_thresh)
    )(logical_trajectories)
    
    # Create mask for valid intersections
    mask = (~jnp.isnan(logical_intersections[..., 0])) & (
        logical_intersections[..., 2] < 0.5
    )
    
    pts_log = logical_intersections[mask] % 1.0
    
    # Get physical coordinates at intersections
    # We need to interpolate from the trajectories
    n_traj, n_pts, _ = logical_trajectories.shape
    traj_idx = jnp.arange(n_traj)[:, None]
    traj_idx_expanded = jnp.broadcast_to(traj_idx, logical_intersections.shape[:2])
    
    # For physical coordinates, use the intersection indices to interpolate
    # For simplicity, we'll compute R, Z from logical coordinates
    # This requires the mapping, but we don't have it here
    # Instead, we'll use a simpler approach: find nearest points in physical trajectory
    
    # Get the intersection indices (which points in trajectory are near the intersection)
    flat_traj_idx = traj_idx_expanded[mask]
    flat_intersection_idx = intersection_indices[mask].astype(int)
    
    # Clamp indices to valid range
    flat_intersection_idx = jnp.clip(flat_intersection_idx, 0, n_pts - 1)
    
    # Get physical coordinates at these indices
    pts_phys = physical_trajectories[flat_traj_idx, flat_intersection_idx]
    
    # Get iota values for each point
    iota_vals = iotas[flat_traj_idx]
    
    # Get B_norm and pressure values at intersections
    B_vals = B_norm[flat_traj_idx, flat_intersection_idx]
    p_vals = pressure[flat_traj_idx, flat_intersection_idx]
    
    # Separate valid and NaN iota points
    valid_mask = ~jnp.isnan(iota_vals)
    nan_mask = jnp.isnan(iota_vals)
    
    # Create figure
    fig = plt.figure(figsize=(14, 5))
    ax1 = fig.add_axes([0.05, 0.12, 0.25, 0.78])   # physical (R, Z)
    ax2 = fig.add_axes([0.38, 0.12, 0.25, 0.78])   # logical (r, theta)
    ax3 = fig.add_axes([0.71, 0.12, 0.25, 0.78])   # color by selection
    cax1 = fig.add_axes([0.31, 0.12, 0.015, 0.78])
    cax2 = fig.add_axes([0.64, 0.12, 0.015, 0.78])
    cax3 = fig.add_axes([0.97, 0.12, 0.015, 0.78])
    
    # Compute R from physical coordinates
    R = jnp.sqrt(pts_phys[:, 0]**2 + pts_phys[:, 1]**2)
    Z = pts_phys[:, 2]
    
    if jnp.any(valid_mask):
        # Left plot: physical coordinates colored by iota
        s1 = ax1.scatter(
            R[valid_mask], Z[valid_mask],
            c=iota_vals[valid_mask],
            cmap=cmap_iota,
            s=markersize,
        )
        fig.colorbar(s1, cax=cax1, label="ι")
        
        # Middle plot: logical coordinates colored by iota
        s2 = ax2.scatter(
            pts_log[valid_mask, 0], pts_log[valid_mask, 1],
            c=iota_vals[valid_mask],
            cmap=cmap_iota,
            s=markersize,
        )
        fig.colorbar(s2, cax=cax2, label="ι")
        
        # Right plot: colored by selection
        if color_by == "pressure":
            s3 = ax3.scatter(
                R[valid_mask], Z[valid_mask],
                c=p_vals[valid_mask],
                cmap=cmap_p,
                s=markersize,
            )
            fig.colorbar(s3, cax=cax3, label="p")
        elif color_by == "B_norm":
            s3 = ax3.scatter(
                R[valid_mask], Z[valid_mask],
                c=B_vals[valid_mask],
                cmap=cmap_B,
                s=markersize,
            )
            fig.colorbar(s3, cax=cax3, label="|B|")
        else:  # iota
            s3 = ax3.scatter(
                pts_log[valid_mask, 0], pts_log[valid_mask, 1],
                c=p_vals[valid_mask],
                cmap=cmap_p,
                s=markersize,
            )
            fig.colorbar(s3, cax=cax3, label="p")
    
    # Plot NaN points in grey
    if jnp.any(nan_mask):
        ax1.scatter(R[nan_mask], Z[nan_mask], c="grey", s=markersize, alpha=0.5)
        ax2.scatter(pts_log[nan_mask, 0], pts_log[nan_mask, 1], c="grey", s=markersize, alpha=0.5)
        ax3.scatter(R[nan_mask], Z[nan_mask], c="grey", s=markersize, alpha=0.5)
    
    # Set labels and titles
    ax1.set(xlabel="R", ylabel="Z", title=f"Physical (ζ={zeta_value:.2f})")
    ax2.set(xlabel="r", ylabel="θ", title="Logical", aspect="equal")
    ax3.set(xlabel="R", ylabel="Z", title=f"Colored by {color_by}")
    
    # Make physical plots square
    for ax in [ax1, ax3]:
        ax.set_aspect("equal", adjustable="box")
    
    if title:
        fig.suptitle(title, fontsize=12)
    
    return fig


def process_directory(
    dir_path: Path,
    zeta_values: List[float],
    output_format: str = "png",
    final_only: bool = False,
    color_by: str = "pressure",
    dpi: int = 150,
) -> List[Path]:
    """
    Process all fieldline HDF5 files in a directory.
    
    Parameters
    ----------
    dir_path : Path
        Directory containing fieldline HDF5 files
    zeta_values : list of float
        Toroidal angles for Poincare sections
    output_format : str
        Output format (png, pdf, svg)
    final_only : bool
        If True, only process fieldlines_final.h5
    color_by : str
        What to color points by
    dpi : int
        DPI for raster formats
    
    Returns
    -------
    output_files : list of Path
        Paths to created plot files
    """
    output_files = []
    
    # Find fieldline files
    if final_only:
        h5_files = list(dir_path.glob("fieldlines_final.h5"))
    else:
        h5_files = sorted(dir_path.glob("fieldlines_*.h5"))
    
    if not h5_files:
        print(f"  No fieldline files found in {dir_path}")
        return output_files
    
    # Create plots subdirectory
    plots_dir = dir_path / "poincare_plots"
    plots_dir.mkdir(exist_ok=True)
    
    for h5_file in h5_files:
        print(f"  Processing {h5_file.name}...")
        
        try:
            data = load_fieldlines(h5_file)
        except Exception as e:
            print(f"    Error loading {h5_file}: {e}")
            continue
        
        # Create title
        if "iteration" in data:
            title = f"Iteration {data['iteration']}"
            if "force_norm" in data:
                title += f" (|F| = {data['force_norm']:.2e})"
        elif "final_force_norm" in data:
            title = f"Final (|F| = {data['final_force_norm']:.2e})"
        else:
            title = h5_file.stem
        
        # Create plots for each zeta value
        for zeta in zeta_values:
            fig = create_poincare_plot(
                data["logical_trajectories"],
                data["physical_trajectories"],
                data["B_norm"],
                data["pressure"],
                data["nfp"],
                zeta_value=zeta,
                title=f"{title}, ζ = {zeta:.2f}",
                color_by=color_by,
            )
            
            # Save plot
            zeta_str = f"{zeta:.2f}".replace(".", "p")
            output_file = plots_dir / f"{h5_file.stem}_zeta{zeta_str}.{output_format}"
            fig.savefig(output_file, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            output_files.append(output_file)
            print(f"    Saved {output_file.name}")
    
    return output_files


def find_output_directories(base_path: Path) -> List[Path]:
    """
    Find all output directories that contain fieldline data.
    Handles both single runs and Hydra multirun directories.
    """
    dirs = []
    
    # Check if this is a multirun directory (contains numbered subdirs)
    subdirs = [d for d in base_path.iterdir() if d.is_dir()]
    numbered_subdirs = [d for d in subdirs if d.name.isdigit()]
    
    if numbered_subdirs:
        # This is a multirun directory
        dirs.extend(sorted(numbered_subdirs, key=lambda x: int(x.name)))
    else:
        # Check if this directory itself has fieldline files
        if list(base_path.glob("fieldlines_*.h5")):
            dirs.append(base_path)
        else:
            # Check one level down
            for subdir in subdirs:
                if list(subdir.glob("fieldlines_*.h5")):
                    dirs.append(subdir)
    
    return dirs


def main():
    parser = argparse.ArgumentParser(
        description="Create Poincare plots from fieldline HDF5 files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Path to output directory or Hydra multirun directory",
    )
    parser.add_argument(
        "--zeta", "-z",
        type=str,
        default="0.0,0.25,0.5",
        help="Comma-separated list of zeta values for Poincare sections (default: 0.0,0.25,0.5)",
    )
    parser.add_argument(
        "--format", "-f",
        type=str,
        choices=["png", "pdf", "svg"],
        default="png",
        help="Output format (default: png)",
    )
    parser.add_argument(
        "--final-only",
        action="store_true",
        help="Only process final fieldline files",
    )
    parser.add_argument(
        "--color-by", "-c",
        type=str,
        choices=["iota", "pressure", "B_norm"],
        default="pressure",
        help="What to color the third panel by (default: pressure)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for raster output (default: 150)",
    )
    
    args = parser.parse_args()
    
    # Parse zeta values
    zeta_values = [float(z.strip()) for z in args.zeta.split(",")]
    
    # Find output directories
    if not args.path.exists():
        print(f"Error: Path does not exist: {args.path}")
        sys.exit(1)
    
    output_dirs = find_output_directories(args.path)
    
    if not output_dirs:
        print(f"Error: No output directories with fieldline data found in {args.path}")
        sys.exit(1)
    
    print(f"Found {len(output_dirs)} output director{'ies' if len(output_dirs) > 1 else 'y'}")
    
    # Process each directory
    all_output_files = []
    for dir_path in output_dirs:
        print(f"\nProcessing {dir_path}...")
        output_files = process_directory(
            dir_path,
            zeta_values,
            output_format=args.format,
            final_only=args.final_only,
            color_by=args.color_by,
            dpi=args.dpi,
        )
        all_output_files.extend(output_files)
    
    print(f"\nCreated {len(all_output_files)} plot{'s' if len(all_output_files) != 1 else ''}")


if __name__ == "__main__":
    main()
