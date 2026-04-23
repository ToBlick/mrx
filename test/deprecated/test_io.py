# %%
# test_io.py
import os
import sys
import tempfile
import shutil
import h5py
import numpy as np
import numpy.testing as npt
import pytest
from unittest.mock import patch
from mrx.io import parse_args, unique_id, epoch_time, load_sweep

import time


# Tests for parse_args
def test_parse_args_simple():
    """Test parse_args with simple key=value arguments."""
    test_args = ['script.py', 'key1=value1', 'key2=value2']
    with patch.object(sys, 'argv', test_args):
        kwargs = parse_args()
        assert kwargs['key1'] == 'value1'
        assert kwargs['key2'] == 'value2'


def test_parse_args_integer():
    """Test parse_args with integer values."""
    test_args = ['script.py', 'n=10', 'p=3']
    with patch.object(sys, 'argv', test_args):
        kwargs = parse_args()
        assert kwargs['n'] == 10
        assert isinstance(kwargs['n'], int)
        assert kwargs['p'] == 3
        assert isinstance(kwargs['p'], int)


def test_parse_args_float():
    """Test parse_args with float values."""
    test_args = ['script.py', 'eps=0.5', 'tol=1e-12']
    with patch.object(sys, 'argv', test_args):
        kwargs = parse_args()
        assert kwargs['eps'] == 0.5
        assert isinstance(kwargs['eps'], float)
        assert kwargs['tol'] == 1e-12
        assert isinstance(kwargs['tol'], float)


def test_parse_args_boolean():
    """Test parse_args with boolean values."""
    test_args = ['script.py', 'flag1=true', 'flag2=false']
    with patch.object(sys, 'argv', test_args):
        kwargs = parse_args()
        assert kwargs['flag1'] is True
        assert kwargs['flag2'] is False
        assert isinstance(kwargs['flag1'], bool)
        assert isinstance(kwargs['flag2'], bool)


def test_parse_args_mixed_types():
    """Test parse_args with mixed types."""
    test_args = ['script.py', 'n=10', 'eps=0.5', 'name=test', 'flag=true']
    with patch.object(sys, 'argv', test_args):
        kwargs = parse_args()
        assert kwargs['n'] == 10
        assert kwargs['eps'] == 0.5
        assert kwargs['name'] == 'test'
        assert kwargs['flag'] is True


def test_parse_args_no_equals():
    """Test parse_args ignores arguments without equals sign."""
    test_args = ['script.py', 'invalid', 'key=value']
    with patch.object(sys, 'argv', test_args):
        kwargs = parse_args()
        # Should ignore 'invalid' and only parse 'key=value'
        assert 'key' in kwargs
        assert kwargs['key'] == 'value'
        assert 'invalid' not in kwargs


def test_parse_args_ci_fallback():
    """Test parse_args handles CI fallback scenario (8 3 arguments)."""
    test_args = ['script.py', '8', '3']
    with patch.object(sys, 'argv', test_args):
        kwargs = parse_args()
        # Should return empty dict, allowing script to use defaults
        assert kwargs == {}


def test_parse_args_empty():
    """Test parse_args with no arguments."""
    test_args = ['script.py']
    with patch.object(sys, 'argv', test_args):
        kwargs = parse_args()
        assert kwargs == {}


def test_parse_args_value_with_equals():
    """Test parse_args with value containing equals sign."""
    test_args = ['script.py', 'path=/some/path=with=equals']
    with patch.object(sys, 'argv', test_args):
        kwargs = parse_args()
        assert kwargs['path'] == '/some/path=with=equals'


# Tests for unique_id
def test_unique_id_length():
    """Test that unique_id returns correct length."""
    for n in [5, 10, 20, 32]:
        uid = unique_id(n)
        assert len(uid) == n, f"unique_id({n}) should return string of length {n}"


def test_unique_id_alphanumeric():
    """Test that unique_id returns alphanumeric characters."""
    uid = unique_id(100)
    assert uid.isalnum(), "unique_id should return alphanumeric string"
    # Check that it contains both letters and digits (likely)
    has_letters = any(c.isalpha() for c in uid)
    has_digits = any(c.isdigit() for c in uid)
    # At least one should be true for a long enough string
    assert has_letters or has_digits, "unique_id should contain letters or digits"


def test_unique_id_uniqueness():
    """Test that unique_id generates different IDs."""
    n = 20
    ids = [unique_id(n) for _ in range(100)]
    # Check that all IDs are unique (very low probability of collision)
    assert len(set(ids)) == len(ids), "unique_id should generate unique IDs"


def test_unique_id_zero_length():
    """Test unique_id with zero length."""
    uid = unique_id(0)
    assert uid == "", "unique_id(0) should return empty string"


# Tests for epoch_time
def test_epoch_time_basic():
    """Test epoch_time returns integer."""
    t = epoch_time()
    assert isinstance(t, int), "epoch_time() should return integer"
    assert t > 0, "epoch_time() should return positive value"


def test_epoch_time_decimals():
    """Test epoch_time with different decimal places."""
    t0 = epoch_time(decimals=0)
    t1 = epoch_time(decimals=1)
    t2 = epoch_time(decimals=2)
    
    assert isinstance(t0, int)
    assert isinstance(t1, int)
    assert isinstance(t2, int)
    # Higher decimals should give larger values (or equal if rounding)
    assert t2 >= t1 >= t0, "Higher decimals should give larger or equal values"


def test_epoch_time_approximate():
    """Test that epoch_time is approximately current time."""
    t = epoch_time()
    current_time = int(time.time())
    # Should be within 1 second
    assert abs(t - current_time) <= 1, "epoch_time() should be approximately current time"


# Tests for load_sweep
@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def reference_h5_file(temp_dir):
    """Create a reference HDF5 file for testing."""
    # Put reference file in a subdirectory to exclude it from load_sweep
    ref_dir = os.path.join(temp_dir, "ref")
    os.makedirs(ref_dir, exist_ok=True)
    ref_file = os.path.join(ref_dir, "reference.h5")
    with h5py.File(ref_file, "w") as f:
        # Create config group with attributes
        config_group = f.create_group("config")
        config_group.attrs["n"] = 10
        config_group.attrs["p"] = 3
        config_group.attrs["eps"] = 0.5
        config_group.attrs["run_name"] = "ref_run"
        config_group.attrs["maxit"] = 1000
        config_group.attrs["save_every"] = 10
        
        # Create QOI dataset
        f.create_dataset("force", data=np.random.rand(100))
    return ref_file


def test_load_sweep_basic(temp_dir, reference_h5_file):
    """Test load_sweep with matching files."""
    # Create a matching file
    match_file = os.path.join(temp_dir, "match.h5")
    with h5py.File(match_file, "w") as f:
        config_group = f.create_group("config")
        config_group.attrs["n"] = 10
        config_group.attrs["p"] = 3
        config_group.attrs["eps"] = 0.5
        config_group.attrs["run_name"] = "match_run"
        config_group.attrs["maxit"] = 1000
        config_group.attrs["save_every"] = 10
        f.create_dataset("force", data=np.random.rand(100))
    
    cfgs, forces, iter_counts = load_sweep(
        temp_dir, reference_h5_file, "force", ["run_name"]
    )
    
    assert len(cfgs) == 1, "Should load one matching file"
    assert len(forces) == 1, "Should load one force trace"
    assert len(iter_counts) == 1, "Should load one iteration count"
    assert cfgs[0]["run_name"] == "match_run", "Config should match"


def test_load_sweep_multiple_files(temp_dir, reference_h5_file):
    """Test load_sweep with multiple matching files."""
    # Create multiple matching files
    for i in range(3):
        match_file = os.path.join(temp_dir, f"match_{i}.h5")
        with h5py.File(match_file, "w") as f:
            config_group = f.create_group("config")
            config_group.attrs["n"] = 10
            config_group.attrs["p"] = 3
            config_group.attrs["eps"] = 0.5
            config_group.attrs["run_name"] = f"run_{i}"
            config_group.attrs["maxit"] = 1000
            config_group.attrs["save_every"] = 10
            f.create_dataset("force", data=np.random.rand(100))
    
    cfgs, forces, iter_counts = load_sweep(
        temp_dir, reference_h5_file, "force", ["run_name"]
    )
    
    assert len(cfgs) == 3, "Should load three matching files"
    assert len(forces) == 3, "Should load three force traces"
    assert len(iter_counts) == 3, "Should load three iteration counts"


def test_load_sweep_skips_non_matching(temp_dir, reference_h5_file):
    """Test load_sweep skips files with unexpected differences."""
    # Create a non-matching file (different n)
    non_match_file = os.path.join(temp_dir, "non_match.h5")
    with h5py.File(non_match_file, "w") as f:
        config_group = f.create_group("config")
        config_group.attrs["n"] = 20  # Different from reference
        config_group.attrs["p"] = 3
        config_group.attrs["eps"] = 0.5
        config_group.attrs["run_name"] = "non_match_run"
        config_group.attrs["maxit"] = 1000
        config_group.attrs["save_every"] = 10
        f.create_dataset("force", data=np.random.rand(100))
    
    # Create a matching file
    match_file = os.path.join(temp_dir, "match.h5")
    with h5py.File(match_file, "w") as f:
        config_group = f.create_group("config")
        config_group.attrs["n"] = 10
        config_group.attrs["p"] = 3
        config_group.attrs["eps"] = 0.5
        config_group.attrs["run_name"] = "match_run"
        config_group.attrs["maxit"] = 1000
        config_group.attrs["save_every"] = 10
        f.create_dataset("force", data=np.random.rand(100))
    
    cfgs, forces, iter_counts = load_sweep(
        temp_dir, reference_h5_file, "force", ["run_name"]
    )
    
    assert len(cfgs) == 1, "Should load only matching file"
    assert cfgs[0]["run_name"] == "match_run", "Should load correct file"


def test_load_sweep_allows_sweep_params(temp_dir, reference_h5_file):
    """Test load_sweep allows differences in sweep parameters."""
    # Create file with different sweep parameter
    sweep_file = os.path.join(temp_dir, "sweep.h5")
    with h5py.File(sweep_file, "w") as f:
        config_group = f.create_group("config")
        config_group.attrs["n"] = 10
        config_group.attrs["p"] = 3
        config_group.attrs["eps"] = 0.6  # Different, but in sweep_params
        config_group.attrs["run_name"] = "sweep_run"
        config_group.attrs["maxit"] = 1000
        config_group.attrs["save_every"] = 10
        f.create_dataset("force", data=np.random.rand(100))
    
    cfgs, forces, iter_counts = load_sweep(
        temp_dir, reference_h5_file, "force", ["eps", "run_name"]
    )
    
    assert len(cfgs) == 1, "Should load file with allowed sweep parameter difference"
    assert cfgs[0]["eps"] == 0.6, "Should preserve sweep parameter value"


def test_load_sweep_iter_counts(temp_dir, reference_h5_file):
    """Test load_sweep generates correct iteration counts."""
    match_file = os.path.join(temp_dir, "match.h5")
    with h5py.File(match_file, "w") as f:
        config_group = f.create_group("config")
        config_group.attrs["n"] = 10
        config_group.attrs["p"] = 3
        config_group.attrs["eps"] = 0.5
        config_group.attrs["run_name"] = "match_run"
        config_group.attrs["maxit"] = 100  # Different from reference, so include in sweep_params
        config_group.attrs["save_every"] = 10
        f.create_dataset("force", data=np.random.rand(10))
    
    cfgs, forces, iter_counts = load_sweep(
        temp_dir, reference_h5_file, "force", ["run_name", "maxit"]
    )
    
    assert len(iter_counts) == 1, "Should generate iteration count"
    expected_iter = np.arange(0, 100, 10)
    npt.assert_array_equal(iter_counts[0], expected_iter, "Iteration count should match expected")


def test_load_sweep_skips_non_h5_files(temp_dir, reference_h5_file):
    """Test load_sweep skips non-HDF5 files."""
    # Create a text file
    text_file = os.path.join(temp_dir, "not_h5.txt")
    with open(text_file, "w") as f:
        f.write("not an h5 file")
    
    # Create a matching h5 file
    match_file = os.path.join(temp_dir, "match.h5")
    with h5py.File(match_file, "w") as f:
        config_group = f.create_group("config")
        config_group.attrs["n"] = 10
        config_group.attrs["p"] = 3
        config_group.attrs["eps"] = 0.5
        config_group.attrs["run_name"] = "match_run"
        config_group.attrs["maxit"] = 1000
        config_group.attrs["save_every"] = 10
        f.create_dataset("force", data=np.random.rand(100))
    
    cfgs, forces, iter_counts = load_sweep(
        temp_dir, reference_h5_file, "force", ["run_name"]
    )
    
    assert len(cfgs) == 1, "Should load only HDF5 files"


def test_load_sweep_handles_corrupted_file(temp_dir, reference_h5_file, capsys):
    """Test load_sweep handles corrupted HDF5 files gracefully."""
    # Create a corrupted file (empty file)
    corrupted_file = os.path.join(temp_dir, "corrupted.h5")
    with open(corrupted_file, "w") as f:
        f.write("not valid h5")
    
    # Create a matching file
    match_file = os.path.join(temp_dir, "match.h5")
    with h5py.File(match_file, "w") as f:
        config_group = f.create_group("config")
        config_group.attrs["n"] = 10
        config_group.attrs["p"] = 3
        config_group.attrs["eps"] = 0.5
        config_group.attrs["run_name"] = "match_run"
        config_group.attrs["maxit"] = 1000
        config_group.attrs["save_every"] = 10
        f.create_dataset("force", data=np.random.rand(100))
    
    cfgs, forces, iter_counts = load_sweep(
        temp_dir, reference_h5_file, "force", ["run_name"]
    )
    
    assert len(cfgs) == 1, "Should load valid file despite corrupted file"
    captured = capsys.readouterr()
    assert "Could not open" in captured.out or "corrupted" in captured.out.lower(), "Should print error for corrupted file"


def test_load_sweep_missing_qoi(temp_dir, reference_h5_file, capsys):
    """Test load_sweep handles missing QOI dataset."""
    # Create file without the QOI dataset
    no_qoi_file = os.path.join(temp_dir, "no_qoi.h5")
    with h5py.File(no_qoi_file, "w") as f:
        config_group = f.create_group("config")
        config_group.attrs["n"] = 10
        config_group.attrs["p"] = 3
        config_group.attrs["eps"] = 0.5
        config_group.attrs["run_name"] = "no_qoi_run"
        config_group.attrs["maxit"] = 1000
        config_group.attrs["save_every"] = 10
        # No "force" dataset
    
    # Create a matching file
    match_file = os.path.join(temp_dir, "match.h5")
    with h5py.File(match_file, "w") as f:
        config_group = f.create_group("config")
        config_group.attrs["n"] = 10
        config_group.attrs["p"] = 3
        config_group.attrs["eps"] = 0.5
        config_group.attrs["run_name"] = "match_run"
        config_group.attrs["maxit"] = 1000
        config_group.attrs["save_every"] = 10
        f.create_dataset("force", data=np.random.rand(100))
    
    cfgs, forces, iter_counts = load_sweep(
        temp_dir, reference_h5_file, "force", ["run_name"]
    )
    
    assert len(cfgs) == 1, "Should load valid file despite missing QOI in other file"
    captured = capsys.readouterr()
    assert "Could not open" in captured.out, "Should print error for file with missing QOI"


def test_load_sweep_bytes_attributes(temp_dir, reference_h5_file):
    """Test load_sweep handles byte string attributes."""
    # Create file with byte string attribute
    match_file = os.path.join(temp_dir, "match.h5")
    with h5py.File(match_file, "w") as f:
        config_group = f.create_group("config")
        config_group.attrs["n"] = 10
        config_group.attrs["p"] = 3
        config_group.attrs["eps"] = 0.5
        config_group.attrs["run_name"] = b"match_run"  # Byte string
        config_group.attrs["maxit"] = 1000
        config_group.attrs["save_every"] = 10
        f.create_dataset("force", data=np.random.rand(100))
    
    cfgs, forces, iter_counts = load_sweep(
        temp_dir, reference_h5_file, "force", ["run_name"]
    )
    
    assert len(cfgs) == 1, "Should load file with byte string attributes"
    assert isinstance(cfgs[0]["run_name"], str), "Byte strings should be decoded to strings"
    assert cfgs[0]["run_name"] == "match_run", "Decoded string should match"


def test_load_sweep_empty_directory(temp_dir, reference_h5_file):
    """Test load_sweep with empty directory."""
    cfgs, forces, iter_counts = load_sweep(
        temp_dir, reference_h5_file, "force", ["run_name"]
    )
    
    assert len(cfgs) == 0, "Should return empty lists for empty directory"
    assert len(forces) == 0, "Should return empty lists for empty directory"
    assert len(iter_counts) == 0, "Should return empty lists for empty directory"


def test_load_sweep_multiple_sweep_params(temp_dir, reference_h5_file):
    """Test load_sweep with multiple sweep parameters."""
    # Create file with multiple different sweep parameters
    sweep_file = os.path.join(temp_dir, "sweep.h5")
    with h5py.File(sweep_file, "w") as f:
        config_group = f.create_group("config")
        config_group.attrs["n"] = 10
        config_group.attrs["p"] = 4  # Different, in sweep_params
        config_group.attrs["eps"] = 0.6  # Different, in sweep_params
        config_group.attrs["run_name"] = "sweep_run"
        config_group.attrs["maxit"] = 1000
        config_group.attrs["save_every"] = 10
        f.create_dataset("force", data=np.random.rand(100))
    
    cfgs, forces, iter_counts = load_sweep(
        temp_dir, reference_h5_file, "force", ["p", "eps", "run_name"]
    )
    
    assert len(cfgs) == 1, "Should load file with multiple sweep parameter differences"
    assert cfgs[0]["p"] == 4, "Should preserve p value"
    assert cfgs[0]["eps"] == 0.6, "Should preserve eps value"

