"""
Tests for data_utils.py (read_dataset, load_training_data, set_seed).

Run with:  pytest test_data_utils.py -v

The tests build their own small synthetic CSV in a temporary directory,
so they don't depend on any pre-existing dataset file.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from utils import (
    INPUT_COLUMNS,
    O2_N2_COLUMNS,
    TARGET_COLUMNS,
    load_training_data,
    read_dataset,
    set_seed,
)

N_ROWS = 50

BURNT_COLUMNS = [
    "Mass Fraction (H2) Burnt",
    "Mass Fraction (CO) Burnt",
    "Mass Fraction (CO2) Burnt",
    "Mass Fraction (H2O) Burnt",
    "Mass Fraction (O2) Burnt",
    "Mass Fraction (N2) Burnt",
]
ALL_COLUMNS = (
    INPUT_COLUMNS + O2_N2_COLUMNS + BURNT_COLUMNS + TARGET_COLUMNS["both"]
)


# --------------------------------- fixtures ---------------------------------
@pytest.fixture
def csv_path(tmp_path):
    """Write a small synthetic dataset CSV and return its path."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.random((N_ROWS, len(ALL_COLUMNS))), columns=ALL_COLUMNS)
    # make rows uniquely identifiable through the Pressure column
    df["Pressure [Pa]"] = np.arange(N_ROWS, dtype=float)
    path = tmp_path / "dataset.csv"
    df.to_csv(path, index=False)
    return str(path)


# -------------------------------- read_dataset ------------------------------
def test_read_dataset_returns_full_dataframe(csv_path):
    df = read_dataset(csv_path)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (N_ROWS, len(ALL_COLUMNS))
    assert list(df.columns) == ALL_COLUMNS


def test_read_dataset_missing_column_raises(tmp_path):
    df = pd.DataFrame({"Pressure [Pa]": [1.0], "Temperature [K]": [300.0]})
    path = tmp_path / "bad.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing expected columns"):
        read_dataset(str(path))


# ----------------------------- load_training_data ---------------------------
def test_returned_dictionary_structure(csv_path):
    data = load_training_data(csv_path)
    assert set(data.keys()) == {"x_train", "x_val", "y_train", "y_val"}
    for value in data.values():
        assert isinstance(value, torch.Tensor)
        assert value.dtype == torch.float32


def test_default_split_is_80_20(csv_path):
    data = load_training_data(csv_path)
    assert data["x_train"].shape[0] == int(0.8 * N_ROWS)
    assert data["x_val"].shape[0] == int(0.2 * N_ROWS)
    assert data["x_train"].shape[0] + data["x_val"].shape[0] == N_ROWS
    assert data["y_train"].shape[0] == data["x_train"].shape[0]
    assert data["y_val"].shape[0] == data["x_val"].shape[0]


def test_custom_val_split(csv_path):
    data = load_training_data(csv_path, val_split=0.5)
    assert data["x_train"].shape[0] == N_ROWS // 2
    assert data["x_val"].shape[0] == N_ROWS // 2


@pytest.mark.parametrize(
    "target, n_outputs",
    [("flame_speed", 1), ("density_ratio", 1), ("both", 2)],
)
def test_target_options_give_expected_y_width(csv_path, target, n_outputs):
    data = load_training_data(csv_path, target=target)
    assert data["y_train"].shape[1] == n_outputs
    assert data["y_val"].shape[1] == n_outputs


def test_x_has_six_inputs_by_default(csv_path):
    data = load_training_data(csv_path)
    assert data["x_train"].shape[1] == 6
    assert data["x_val"].shape[1] == 6


def test_include_o2_n2_gives_eight_inputs(csv_path):
    data = load_training_data(csv_path, include_o2_n2=True)
    assert data["x_train"].shape[1] == 8
    assert data["x_val"].shape[1] == 8


def test_y_values_match_csv_columns(csv_path):
    """y must contain exactly the target columns of the CSV, row-aligned to x."""
    df = read_dataset(csv_path)
    data = load_training_data(csv_path, target="both")

    # the Pressure column (x[:, 0]) uniquely identifies each row
    lookup = df.set_index("Pressure [Pa]")[TARGET_COLUMNS["both"]]
    for split in ("train", "val"):
        x = data["x_" + split].numpy()
        y = data["y_" + split].numpy()
        expected = lookup.loc[x[:, 0]].to_numpy()
        assert np.allclose(y, expected, atol=1e-6)


def test_train_and_val_are_disjoint_and_cover_dataset(csv_path):
    data = load_training_data(csv_path)
    train_ids = set(data["x_train"][:, 0].tolist())  # unique Pressure values
    val_ids = set(data["x_val"][:, 0].tolist())
    assert train_ids.isdisjoint(val_ids)
    assert len(train_ids | val_ids) == N_ROWS


def test_split_is_shuffled(csv_path):
    """The validation set should not simply be the first 20% of the file."""
    data = load_training_data(csv_path)
    first_rows = set(float(i) for i in range(int(0.2 * N_ROWS)))
    val_ids = set(data["x_val"][:, 0].tolist())
    assert val_ids != first_rows


def test_same_seed_gives_identical_split(csv_path):
    a = load_training_data(csv_path, seed=7)
    b = load_training_data(csv_path, seed=7)
    for key in a:
        assert torch.equal(a[key], b[key])


def test_different_seed_gives_different_split(csv_path):
    a = load_training_data(csv_path, seed=7)
    b = load_training_data(csv_path, seed=8)
    assert not torch.equal(a["x_train"], b["x_train"])


def test_custom_dtype(csv_path):
    data = load_training_data(csv_path, dtype=torch.float64)
    for value in data.values():
        assert value.dtype == torch.float64


def test_invalid_target_raises(csv_path):
    with pytest.raises(ValueError, match="target must be one of"):
        load_training_data(csv_path, target="enthalpy")


@pytest.mark.parametrize("bad_split", [0.0, 1.0, -0.1, 1.5])
def test_invalid_val_split_raises(csv_path, bad_split):
    with pytest.raises(ValueError, match="val_split"):
        load_training_data(csv_path, val_split=bad_split)


# ---------------------------------- set_seed --------------------------------
def test_set_seed_makes_numpy_and_torch_reproducible():
    set_seed(123)
    np_a, torch_a = np.random.random(5), torch.rand(5)
    set_seed(123)
    np_b, torch_b = np.random.random(5), torch.rand(5)
    assert np.array_equal(np_a, np_b)
    assert torch.equal(torch_a, torch_b)