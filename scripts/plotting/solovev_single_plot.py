# %%
import h5py
import numpy as np

name = "solovev_shaped_8x8"
with h5py.File("../../script_outputs/solovev/" + name + ".h5", "r") as f:
        B_hat = f["B_hat"][:]
        p_hat = f["p_hat"][:]
        helicity_trace = f["helicity_trace"][:]
        energy_trace = f["energy_trace"][:]
        force_trace = f["force_trace"][:]

        cfg = {k: v for k, v in f["config"].attrs.items()}
        # decode strings back if needed
        cfg = {k: v.decode() if isinstance(v, bytes)
               else v for k, v in cfg.items()}
# %%
cfg
# %%
force_trace[-1]
# %%
