"""Where MRX writes its runs and finds input files that are not in the repository.

The environment variable ``MRX_DATA`` names one directory outside the repository, shared by the MRX
projects:

- ``$MRX_DATA/data``: input files that are not tracked in git (large wout files, mgrids, GVEC states),
- ``$MRX_DATA/runs/mrx/<experiment>``: the run directories and job logs of this repository,
- ``$MRX_DATA/cache``: compilation caches.

Without ``MRX_DATA`` runs go to ``outputs`` in the working directory and input files are read from the
path as given.
"""
import os


def runs_root():
    """The directory that run directories are created in: ``$MRX_DATA/runs/mrx`` if ``MRX_DATA`` is set,
    ``outputs`` in the working directory otherwise."""
    root = os.environ.get("MRX_DATA")
    return os.path.join(root, "runs", "mrx") if root else "outputs"


def data_file(path):
    """The input file ``path``. If it does not exist and ``MRX_DATA`` is set, the file of the same name
    in ``$MRX_DATA/data`` is returned when that one exists."""
    root = os.environ.get("MRX_DATA")
    if root and not os.path.exists(path):
        shared = os.path.join(root, "data", os.path.basename(path))
        if os.path.exists(shared):
            return shared
    return path
