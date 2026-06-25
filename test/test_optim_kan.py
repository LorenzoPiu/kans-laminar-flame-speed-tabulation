"""Tests for the KAN HPO module (optimization_KAN.py).

Run with:  pytest test_optimization_kan.py -v

Strategy
--------
The cheap, pure helpers (compute_tail, normalize_split_keys, to_device,
width_from_params, resolve_device, build_parser, suggest_hidden_widths,
save_results) are tested directly. The expensive runtime dependencies - real
KAN training and dataset loading - are replaced with lightweight fakes via
monkeypatch for the objective/main tests, except for ONE small real-KAN
integration test that exercises the actual pykan wiring.

Importing this test (and the module under test) requires `kan`, `utils` and
`global_vars` to be importable, exactly as optimization_ANN.py's tests assume.
"""

import matplotlib

matplotlib.use("Agg")  # kan imports matplotlib transitively; force headless

import csv
import json
import math
import types

import numpy as np
import optuna
import pytest
import torch
from optuna.trial import FixedTrial

import optim_kan as okan


# --------------------------------------------------------------------- fakes
class FakeKAN:
    """Stand-in for kan.KAN that records the kwargs it was built/fit with and
    returns a controllable RMSE history (KAN records RMSE)."""

    test_loss = [0.3, 0.2, 0.1]   # RMSE per step; override per test
    last_init = None
    last_fit = None

    def __init__(self, **kwargs):
        FakeKAN.last_init = kwargs

    def fit(self, data, **kwargs):
        FakeKAN.last_fit = kwargs
        n = len(FakeKAN.test_loss)
        return {
            "train_loss": list(FakeKAN.test_loss),
            "test_loss":  list(FakeKAN.test_loss),
            "reg":        [0.0] * n,
        }


@pytest.fixture(autouse=True)
def _reset_fake():
    FakeKAN.test_loss = [0.3, 0.2, 0.1]
    FakeKAN.last_init = None
    FakeKAN.last_fit = None
    yield


@pytest.fixture
def kan_data():
    """Synthetic data already in KAN key scheme (float32, cpu)."""
    torch.manual_seed(0)
    x = torch.randn(100, okan.INPUT_DIM)
    y = x @ torch.randn(okan.INPUT_DIM, okan.OUTPUT_DIM)
    return {
        "train_input": x[:70], "train_label": y[:70],
        "test_input":  x[70:], "test_label":  y[70:],
    }


def _sample_widths(n=200, seed=0):
    """Draw n funnel proposals from suggest_hidden_widths via a random study."""
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=seed))
    out = []
    for _ in range(n):
        t = study.ask()
        out.append(okan.suggest_hidden_widths(t))
        study.tell(t, 0.0)
    return out


def _fixed_trial(**overrides):
    params = {"n_layers": 2, "n_neurons_l0": 8, "n_neurons_l1": 4,
              "grid": 7, "k": 3, "lr": 0.5, "lamb": 1e-4}
    params.update(overrides)
    return FixedTrial(params)


# ----------------------------------------------------------------- helpers
class TestComputeTail:
    def test_one_twentieth(self):
        assert okan.compute_tail(10000) == 500
        assert okan.compute_tail(100) == 5

    def test_floor_at_one(self):
        assert okan.compute_tail(5) == 1
        assert okan.compute_tail(1) == 1
        assert okan.compute_tail(0) == 1


class TestNormalizeSplitKeys:
    def test_maps_val_keys(self):
        data = {"x_train": "a", "y_train": "b", "x_val": "c", "y_val": "d"}
        out = okan.normalize_split_keys(data)
        assert out == {"train_input": "a", "train_label": "b",
                       "test_input": "c", "test_label": "d"}

    def test_maps_test_keys(self):
        data = {"x_train": "a", "y_train": "b", "x_test": "c", "y_test": "d"}
        out = okan.normalize_split_keys(data)
        assert out["test_input"] == "c" and out["test_label"] == "d"

    def test_missing_validation_split_raises(self):
        with pytest.raises(KeyError):
            okan.normalize_split_keys({"x_train": "a", "y_train": "b"})


class TestToDevice:
    def test_casts_to_float32_on_cpu(self):
        data = {"train_input": torch.randn(4, 3, dtype=torch.float64)}
        out = okan.to_device(data, "cpu")
        assert out["train_input"].dtype == torch.float32
        assert out["train_input"].device.type == "cpu"

    def test_passes_non_tensors_through(self):
        out = okan.to_device({"meta": "flame_speed"}, "cpu")
        assert out["meta"] == "flame_speed"


class TestWidthFromParams:
    def test_reconstructs_full_width(self):
        params = {"n_layers": 3, "n_neurons_l0": 8, "n_neurons_l1": 4,
                  "n_neurons_l2": 2}
        assert okan.width_from_params(params) == [okan.INPUT_DIM, 8, 4, 2,
                                                  okan.OUTPUT_DIM]

    def test_single_hidden_layer(self):
        params = {"n_layers": 1, "n_neurons_l0": 5}
        assert okan.width_from_params(params) == [okan.INPUT_DIM, 5,
                                                  okan.OUTPUT_DIM]


class TestResolveDevice:
    def test_cpu_when_not_requested(self):
        assert okan.resolve_device(False) == "cpu"

    def test_gpu_request_matches_availability(self):
        expected = "cuda" if torch.cuda.is_available() else "cpu"
        assert okan.resolve_device(True) == expected


class TestBuildParser:
    def test_defaults(self):
        args = okan.build_parser().parse_args(["--data_path", "d.csv"])
        assert args.target == "flame_speed"
        assert args.steps == 100
        assert args.trials == 200
        assert args.output_dir == okan.HPO_OUTPUT_DIR
        assert args.use_gpu is False

    def test_data_path_required(self):
        with pytest.raises(SystemExit):
            okan.build_parser().parse_args([])

    def test_invalid_target_rejected(self):
        with pytest.raises(SystemExit):
            okan.build_parser().parse_args(["--data_path", "d.csv",
                                            "--target", "nonsense"])

    def test_use_gpu_flag(self):
        args = okan.build_parser().parse_args(["--data_path", "d.csv", "--use_gpu"])
        assert args.use_gpu is True


# ------------------------------------------------------- funnel architecture
class TestSuggestHiddenWidths:
    def test_strictly_decreasing_default(self):
        assert okan.STRICTLY_DECREASING is True
        for h in _sample_widths(200, seed=0):
            assert 1 <= len(h) <= okan.MAX_HIDDEN_LAYERS
            assert all(okan.MIN_NEURONS_PER_LAYER <= v <= okan.MAX_NEURONS_PER_LAYER
                       for v in h)
            for a, b in zip(h, h[1:]):
                assert b < a  # strictly decreasing

    def test_non_increasing_when_flag_off(self, monkeypatch):
        monkeypatch.setattr(okan, "STRICTLY_DECREASING", False)
        for h in _sample_widths(200, seed=0):
            assert 1 <= len(h) <= okan.MAX_HIDDEN_LAYERS
            for a, b in zip(h, h[1:]):
                assert b <= a  # non-increasing (equal allowed)

    def test_strict_feasibility_cap(self, monkeypatch):
        # span = MAX - MIN + 1 = 3, so depth must be capped at 3 even though
        # MAX_HIDDEN_LAYERS is larger; no infeasible config may be proposed.
        monkeypatch.setattr(okan, "STRICTLY_DECREASING", True)
        monkeypatch.setattr(okan, "MIN_NEURONS_PER_LAYER", 1)
        monkeypatch.setattr(okan, "MAX_NEURONS_PER_LAYER", 3)
        monkeypatch.setattr(okan, "MAX_HIDDEN_LAYERS", 10)
        for h in _sample_widths(200, seed=1):
            assert len(h) <= 3
            assert all(1 <= v <= 3 for v in h)
            for a, b in zip(h, h[1:]):
                assert b < a


# -------------------------------------------------------------- objective
class TestMakeObjective:
    def test_returns_mean_squared_rmse_tail(self, monkeypatch, kan_data, tmp_path):
        monkeypatch.setattr(okan, "KAN", FakeKAN)
        FakeKAN.test_loss = [3.0, 2.0, 1.0]  # RMSE
        records = []
        objective = okan.make_objective(kan_data, steps=10, n_tail=2,
                                        trial_records=records,
                                        ckpt_path=str(tmp_path))
        val = objective(_fixed_trial())
        # tail RMSE = [2, 1] -> MSE = [4, 1] -> mean = 2.5
        assert val == pytest.approx(2.5)

    def test_records_expected_fields(self, monkeypatch, kan_data, tmp_path):
        monkeypatch.setattr(okan, "KAN", FakeKAN)
        records = []
        objective = okan.make_objective(kan_data, steps=10, n_tail=1,
                                        trial_records=records,
                                        ckpt_path=str(tmp_path))
        objective(_fixed_trial())
        assert len(records) == 1
        rec = records[0]
        assert set(rec) == set(okan.CSV_FIELDNAMES)
        assert rec["n_layers"] == 2
        assert rec["width"] == "6-8-4-1"   # [INPUT, 8, 4, OUTPUT]
        assert rec["grid"] == 7 and rec["k"] == 3
        assert rec["batch"] == -1          # TUNE_BATCH is False by default
        assert math.isfinite(rec["val_loss"])

    def test_full_batch_passed_to_fit(self, monkeypatch, kan_data, tmp_path):
        monkeypatch.setattr(okan, "KAN", FakeKAN)
        objective = okan.make_objective(kan_data, steps=10, n_tail=1,
                                        trial_records=[], ckpt_path=str(tmp_path))
        objective(_fixed_trial())
        assert FakeKAN.last_fit["batch"] == -1

    def test_kan_built_with_grid_k_and_safe_flags(self, monkeypatch, kan_data, tmp_path):
        monkeypatch.setattr(okan, "KAN", FakeKAN)
        objective = okan.make_objective(kan_data, steps=10, n_tail=1,
                                        trial_records=[], ckpt_path=str(tmp_path))
        objective(_fixed_trial())
        init = FakeKAN.last_init
        assert init["grid"] == 7 and init["k"] == 3
        assert init["width"] == [okan.INPUT_DIM, 8, 4, okan.OUTPUT_DIM]
        assert init["auto_save"] is False
        assert init["save_act"] is True

    def test_nan_loss_becomes_inf(self, monkeypatch, kan_data, tmp_path):
        monkeypatch.setattr(okan, "KAN", FakeKAN)
        FakeKAN.test_loss = [1.0, float("nan")]
        objective = okan.make_objective(kan_data, steps=10, n_tail=1,
                                        trial_records=[], ckpt_path=str(tmp_path))
        assert objective(_fixed_trial()) == float("inf")

    def test_empty_history_raises(self, monkeypatch, kan_data, tmp_path):
        monkeypatch.setattr(okan, "KAN", FakeKAN)
        FakeKAN.test_loss = []  # fit recorded nothing
        objective = okan.make_objective(kan_data, steps=10, n_tail=1,
                                        trial_records=[], ckpt_path=str(tmp_path))
        with pytest.raises(RuntimeError):
            objective(_fixed_trial())


class TestMakeObjectiveRealKAN:
    """One genuine pykan run to exercise the real wiring (small + fast)."""

    def test_real_kan_objective_runs(self, kan_data, tmp_path):
        records = []
        objective = okan.make_objective(kan_data, steps=3, n_tail=1,
                                        trial_records=records, device="cpu",
                                        ckpt_path=str(tmp_path))
        trial = FixedTrial({"n_layers": 1, "n_neurons_l0": 2,
                            "grid": 3, "k": 2, "lr": 1.0, "lamb": 1e-3})
        val = objective(trial)
        assert isinstance(val, float)
        assert val >= 0.0          # finite MSE or +inf, never negative
        assert len(records) == 1
        assert records[0]["width"] == "6-2-1"


# -------------------------------------------------------------- persistence
class TestSaveResults:
    def _fake_study(self):
        params = {"n_layers": 2, "n_neurons_l0": 8, "n_neurons_l1": 4,
                  "grid": 7, "k": 3, "lr": 0.5, "lamb": 1e-4}
        return types.SimpleNamespace(
            best_trial=types.SimpleNamespace(params=params),
            best_value=0.123,
        )

    def test_writes_best_params_json(self, tmp_path):
        bp = tmp_path / "best.json"
        cp = tmp_path / "hist.csv"
        okan.save_results(self._fake_study(), [], str(bp), str(cp))
        saved = json.loads(bp.read_text())
        assert saved["val_loss"] == 0.123
        assert saved["grid"] == 7 and saved["k"] == 3
        assert saved["width"] == "6-8-4-1"

    def test_writes_csv_history(self, tmp_path):
        bp = tmp_path / "best.json"
        cp = tmp_path / "hist.csv"
        records = [
            {"trial": 0, "n_layers": 1, "width": "6-2-1", "grid": 5, "k": 3,
             "batch": -1, "lr": 0.1, "lamb": 1e-5, "val_loss": 0.5},
            {"trial": 1, "n_layers": 2, "width": "6-4-2-1", "grid": 8, "k": 2,
             "batch": -1, "lr": 0.2, "lamb": 1e-6, "val_loss": 0.4},
        ]
        okan.save_results(self._fake_study(), records, str(bp), str(cp))
        rows = list(csv.DictReader(cp.open()))
        assert [r["trial"] for r in rows] == ["0", "1"]
        assert rows[1]["width"] == "6-4-2-1"
        assert list(rows[0].keys()) == okan.CSV_FIELDNAMES


# ------------------------------------------------------------------- main
class TestMain:
    def _fake_loader(self, *a, **k):
        torch.manual_seed(0)
        x = torch.randn(80, okan.INPUT_DIM)
        y = x @ torch.randn(okan.INPUT_DIM, okan.OUTPUT_DIM)
        return {"x_train": x[:60], "y_train": y[:60],
                "x_val": x[60:], "y_val": y[60:]}

    def test_main_runs_and_writes_artifacts(self, monkeypatch, tmp_path):
        monkeypatch.setattr(okan, "KAN", FakeKAN)
        monkeypatch.setattr(okan, "load_training_data", self._fake_loader)

        study = okan.main([
            "--data_path", "x.csv",
            "--target", "flame_speed",
            "--steps", "20",
            "--trials", "3",
            "--output_dir", str(tmp_path),
        ])
        assert len(study.trials) == 3

        tag, target = okan.MODEL_TAG, "flame_speed"
        csv_path = tmp_path / okan.HPO_CSV_FILENAME.format(kan_or_ann=tag, target=target)
        best_path = tmp_path / okan.HPO_BEST_PARAMS_FILENAME.format(kan_or_ann=tag, target=target)
        log_path = tmp_path / okan.HPO_LOG_FILENAME.format(kan_or_ann=tag, target=target)
        assert csv_path.exists() and best_path.exists() and log_path.exists()

        rows = list(csv.DictReader(csv_path.open()))
        assert len(rows) == 3
        saved = json.loads(best_path.read_text())
        assert "width" in saved and "val_loss" in saved