"""``mrx.cli`` on ``mrx.relax_config``: the command line of the dataclasses round-trips.

Milliseconds, no sequence: the generated parser accepts relax.py's flags with their spellings, the
defaults resolve by method, the validation is the parser's error, and ``params`` / ``from_params``
are inverse (a record rebuilds its configuration).
"""
import argparse

import pytest

from mrx.cli import add_arguments, flatten, from_namespace, unflatten
from mrx.relax_config import RelaxConfig

GEOMETRY = "data/wout_li383_low_res_reference.nc"


def parse(*argv):
    ap = argparse.ArgumentParser()
    add_arguments(ap, RelaxConfig)
    return from_namespace(RelaxConfig, ap.parse_args(["--geometry", GEOMETRY, *argv]))


def test_defaults_resolve_by_method():
    cfg = parse()
    assert (cfg.budget.steps, cfg.budget.chunk, cfg.descent.potential_velocity) == (150, 25, False)
    assert cfg.geometry.ns == (32, 64, 64) and cfg.newton.penalty == 3.0
    cfg = parse("--method", "gradient")
    assert (cfg.budget.steps, cfg.budget.chunk, cfg.descent.potential_velocity) == (3000, 500, True)


def test_flags_keep_their_spellings():
    cfg = parse("--ns", "8,12,12", "--newton-tol", "0.2", "--seed", "6,1,0.5,0.1", "--seed-eps", "1e-2",
                "--auxiliary-B-field", "--no-helicity-correction", "--reconnect-window", "10:20",
                "--knots-r", "0,0.5,1", "--steps", "50", "--chunk", "25")
    assert cfg.geometry.ns == (8, 12, 12) and cfg.newton.tol == 0.2
    assert cfg.seed.spec == "6,1,0.5,0.1" and cfg.seed.eps == 1e-2 and cfg.seed
    assert cfg.descent.auxiliary_B_field and not cfg.descent.helicity_correction
    assert cfg.reconnect.window == (10, 20) and cfg.geometry.knots_r is not None


def test_validation_is_a_value_error():
    with pytest.raises(ValueError, match="multiple of --chunk"):
        parse("--steps", "40")
    with pytest.raises(ValueError, match="--drive needs"):
        parse("--drive", "6,1,0.5,0.1")


def test_params_round_trip():
    cfg = parse("--method", "gradient", "--ns", "8,12,12", "--reconnect-every", "100", "--resistivity", "0.2")
    params = flatten(cfg)
    assert params["ns"] == [8, 12, 12] and params["newton_penalty"] == 3.0 and params["steps"] == 3000
    params.update(geometry_path="/abs/path", h_r_sq=0.01, start_step=0)     # the driver's facts
    assert unflatten(RelaxConfig, params) == cfg
    assert RelaxConfig.from_params(cfg.params) == cfg
