"""``mrx.cli`` on ``mrx.relax_config``: the command line of the dataclasses round-trips.

Milliseconds, no sequence: the generated parser accepts relax.py's flags with their spellings, the
enums by their values, the defaults resolve by method, the validation is the parser's error, and
``params`` / ``from_params`` are inverse (a record rebuilds its configuration).
"""
import argparse

import pytest

from mrx.cli import add_arguments, flatten, from_namespace, unflatten
from mrx.relax_config import Method, Precision, RelaxConfig, Symmetry

GEOMETRY = "data/wout_li383_low_res_reference.nc"


def parse(*argv):
    ap = argparse.ArgumentParser()
    add_arguments(ap, RelaxConfig)
    return from_namespace(RelaxConfig, ap.parse_args(["--geometry", GEOMETRY, *argv]))


def test_defaults_resolve_by_method():
    cfg = parse()
    assert (cfg.budget.steps, cfg.budget.chunk, cfg.budget.floor_tol) == (100, 10, 1e-10)
    assert cfg.geometry.resolution == (32, 64, 64) and cfg.newton.penalty == 3.0
    assert cfg.geometry.symmetry is Symmetry.STELLARATOR and cfg.geometry.precision is Precision.MIXED
    cfg = parse("--method", "gradient")
    assert cfg.descent.method is Method.GRADIENT and (cfg.budget.steps, cfg.budget.chunk) == (2000, 200)


def test_flags_keep_their_spellings():
    cfg = parse("--resolution", "8,12,12", "--spline-degree", "3", "--newton-tol", "0.2", "--seed",
                "--seed-iotas", "0.5,0.6", "--seed-amplitudes", "1e-2,-1e-2", "--scheme", "midpoint",
                "--knots-r", "0,0.5,1", "--steps", "50", "--chunk", "25", "--symmetry", "field-period")
    assert cfg.geometry.resolution == (8, 12, 12) and cfg.geometry.spline_degree == 3 and cfg.newton.tol == 0.2
    assert cfg.seed and cfg.seed.iotas == (0.5, 0.6) and cfg.seed.amplitudes == (1e-2, -1e-2)
    assert cfg.descent.scheme == "midpoint" and cfg.geometry.symmetry is Symmetry.FIELD_PERIOD
    assert cfg.geometry.knots_r is not None


def test_validation_is_a_value_error():
    with pytest.raises(ValueError, match="multiple of --chunk"):
        parse("--steps", "45")
    with pytest.raises(ValueError, match="--drive-resistivity needs --drive-reference"):
        parse("--drive-resistivity", "0.2")
    with pytest.raises(ValueError, match="one value per"):
        parse("--seed", "--seed-iotas", "0.5", "--seed-amplitudes", "1,2")
    with pytest.warns(UserWarning, match="ignored"):
        cfg = parse("--seed", "--seed-amplitudes", "1e-2")
    assert cfg.seed.amplitudes is None


def test_params_round_trip():
    cfg = parse("--method", "gradient", "--resolution", "8,12,12", "--drive-resistivity", "0.2",
                "--drive-reference", "x.h5", "--precision", "float64")
    params = flatten(cfg)
    assert params["resolution"] == [8, 12, 12] and params["newton_penalty"] == 3.0 and params["steps"] == 2000
    assert params["precision"] == "float64" and params["method"] == "gradient"      # enums as their values
    params.update(geometry_path="/abs/path", h_r_sq=0.01, start_step=0)             # the driver's facts
    assert unflatten(RelaxConfig, params) == cfg
    assert RelaxConfig.from_params(cfg.params) == cfg
