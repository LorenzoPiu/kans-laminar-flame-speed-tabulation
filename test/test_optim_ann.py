"""Tests for optimization_ANN.py.

Run with:  pytest test_optimization_ann.py -v

These tests do NOT need a real dataset CSV: load_training_data is
monkeypatched with small synthetic tensors. The end-to-end test runs the
whole pipeline (real FCNN training + real Optuna study) in-process with a
narrowed search space, 3 trials and 20 epochs, so it finishes in seconds.
"""

import csv
import json

import numpy as np
import optuna
import pytest
import torch

import optim_ann as hpo


# --------------------------------------------------------------- fixtures
@pytest.fixture
def fake_split():
    """Tiny synthetic dataset using the loader's 'x_val'/'y_val' convention."""
    torch.manual_seed(0)
    x = torch.randn(64, hpo.INPUT_DIM)
    y = 0.5 * x[:, :1] + 0.1
    return {"x_train": x[:48], "y_train": y[:48],
            "x_val": x[48:], "y_val": y[48:]}


class DummyFCNN:
    """Stand-in for the real network: records its arguments and fakes a
    decreasing validation-loss history, so objective bookkeeping can be
    checked deterministically and instantly."""

    last_instance = None

    def __init__(self, input_dim, n_layers, n_neurons, output_dim):
        self.init_args = dict(input_dim=input_dim, n_layers=n_layers,
                              n_neurons=n_neurons, output_dim=output_dim)
        self.fit_kwargs = None
        self.test_mse_loss = []
        DummyFCNN.last_instance = self

    def fit(self, **kwargs):
        self.fit_kwargs = kwargs
        epochs = kwargs["epochs"]
        self.test_mse_loss = [float(v) for v in np.linspace(1.0, 0.1, epochs)]


def fixed_trial():
    return optuna.trial.FixedTrial({
        "n_layers": 3,
        "n_neurons": 32,
        "batch_size": 512,
        "lr": 1e-3,
        "weight_decay": 1e-5,
    })


# ------------------------------------------------------------ pure helpers
class TestHelpers:
    def test_compute_tail_is_one_twentieth(self):
        assert hpo.compute_tail(10000) == 500
        assert hpo.compute_tail(100) == 5

    def test_compute_tail_never_zero(self):
        assert hpo.compute_tail(19) == 1
        assert hpo.compute_tail(1) == 1

    def test_normalize_maps_val_to_test_keys(self, fake_split):
        out = hpo.normalize_split_keys(fake_split)
        assert torch.equal(out["x_test"], fake_split["x_val"])
        assert torch.equal(out["y_test"], fake_split["y_val"])

    def test_normalize_passes_test_keys_through(self, fake_split):
        d = {"x_train": fake_split["x_train"], "y_train": fake_split["y_train"],
             "x_test": fake_split["x_val"], "y_test": fake_split["y_val"]}
        out = hpo.normalize_split_keys(d)
        assert torch.equal(out["x_test"], d["x_test"])

    def test_normalize_missing_split_raises(self, fake_split):
        with pytest.raises(KeyError):
            hpo.normalize_split_keys({"x_train": fake_split["x_train"],
                                      "y_train": fake_split["y_train"]})


# ----------------------------------------------------------------- parser
class TestParser:
    def test_defaults(self):
        args = hpo.build_parser().parse_args(["-d", "data.csv"])
        assert args.target == "flame_speed"
        assert args.epochs == 10000
        assert args.trials == 300
        assert args.output_dir == "hpo_output"

    def test_data_path_is_required(self):
        with pytest.raises(SystemExit):
            hpo.build_parser().parse_args([])

    def test_invalid_target_rejected(self):
        with pytest.raises(SystemExit):
            hpo.build_parser().parse_args(["-d", "x.csv", "-y", "pressure"])


# -------------------------------------------------------------- objective
class TestObjective:
    def test_returns_tail_mean_and_records_trial(self, monkeypatch, fake_split):
        monkeypatch.setattr(hpo, "FCNN", DummyFCNN)
        records = []
        data = hpo.normalize_split_keys(fake_split)
        objective = hpo.make_objective(
            data, epochs=100, n_tail=10, trial_records=records, use_gpu=False
        )

        val = objective(fixed_trial())

        model = DummyFCNN.last_instance
        expected = float(np.mean(model.test_mse_loss[-10:]))
        assert val == pytest.approx(expected)

        # the model was built with the fixed dims and suggested hparams
        assert model.init_args == {"input_dim": hpo.INPUT_DIM, "n_layers": 3,
                                   "n_neurons": 32, "output_dim": hpo.OUTPUT_DIM}
        assert model.fit_kwargs["learning_rate"] == pytest.approx(1e-3)
        assert model.fit_kwargs["weight_decay"] == pytest.approx(1e-5)
        assert model.fit_kwargs["batch_size"] == 512
        assert model.fit_kwargs["verbose"] is False

        # bookkeeping for the CSV export
        assert len(records) == 1
        assert records[0]["val_loss"] == pytest.approx(expected)
        assert records[0]["n_layers"] == 3

    def test_empty_val_history_raises(self, monkeypatch, fake_split):
        """Regression test for the x_val/x_test key mismatch: if fit()
        records no validation losses, fail loudly instead of returning NaN."""

        class BrokenFCNN(DummyFCNN):
            def fit(self, **kwargs):
                self.test_mse_loss = []

        monkeypatch.setattr(hpo, "FCNN", BrokenFCNN)
        objective = hpo.make_objective(
            hpo.normalize_split_keys(fake_split), epochs=10, n_tail=1,
            trial_records=[], use_gpu=False,
        )
        with pytest.raises(RuntimeError, match="no validation losses"):
            objective(fixed_trial())

    def test_diverged_trial_returns_inf_not_nan(self, monkeypatch, fake_split):
        class DivergedFCNN(DummyFCNN):
            def fit(self, **kwargs):
                self.fit_kwargs = kwargs
                self.test_mse_loss = [float("nan")] * kwargs["epochs"]

        monkeypatch.setattr(hpo, "FCNN", DivergedFCNN)
        records = []
        objective = hpo.make_objective(
            hpo.normalize_split_keys(fake_split), epochs=10, n_tail=5,
            trial_records=records, use_gpu=False,
        )
        val = objective(fixed_trial())
        assert val == float("inf")
        assert records[0]["val_loss"] == float("inf")


# ------------------------------------------------------------- end-to-end
class TestEndToEnd:
    def test_main_produces_consistent_artifacts(self, tmp_path, monkeypatch,
                                                 fake_split):
        """Full pipeline in-process: real FCNN, real Optuna study, fake data.

        The search space is narrowed via monkeypatch so 3 trials x 20 epochs
        stay fast and numerically stable on CPU.
        """
        monkeypatch.setattr(hpo, "load_training_data",
                            lambda *a, **k: fake_split)
        monkeypatch.setattr(hpo, "set_seed",
                            lambda seed, **k: torch.manual_seed(seed))
        monkeypatch.setattr(hpo, "N_LAYERS_HIGH", 2)
        monkeypatch.setattr(hpo, "N_NEURONS_HIGH", 16)
        monkeypatch.setattr(hpo, "BATCH_SIZE_LOW", 16)
        monkeypatch.setattr(hpo, "BATCH_SIZE_HIGH", 64)
        monkeypatch.setattr(hpo, "LR_LOW", 1e-4)
        monkeypatch.setattr(hpo, "LR_HIGH", 1e-2)
        monkeypatch.setattr(hpo, "WD_HIGH", 1e-4)

        out_dir = tmp_path / "hpo"
        study = hpo.main(["-d", "fake.csv", "-e", "20", "-t", "3",
                          "-o", str(out_dir)])

        # --- artifacts exist ---------------------------------------------
        best_path = out_dir / "best_params.json"
        csv_path = out_dir / "hpo_results.csv"
        log_path = out_dir / "hpo_log.txt"
        assert best_path.exists() and csv_path.exists() and log_path.exists()

        # --- CSV: one row per trial, all losses finite ---------------------
        with open(csv_path) as fp:
            rows = list(csv.DictReader(fp))
        assert len(rows) == 3
        csv_losses = [float(r["val_loss"]) for r in rows]
        assert all(np.isfinite(v) for v in csv_losses)

        # --- best_params.json consistent with study and CSV ----------------
        best = json.loads(best_path.read_text())
        for key in ("n_layers", "n_neurons", "batch_size", "lr",
                    "weight_decay", "val_loss"):
            assert key in best
        assert best["val_loss"] == pytest.approx(min(csv_losses))
        assert best["val_loss"] == pytest.approx(study.best_value)

        # --- suggested values respect the (patched) bounds -----------------
        for r in rows:
            assert int(r["n_layers"]) == 2  # LOW == patched HIGH == 2
            assert 8 <= int(r["n_neurons"]) <= 16
            assert 16 <= int(r["batch_size"]) <= 64

        # --- the log captured the run, stdout was restored -----------------
        log = log_path.read_text()
        assert "HPO complete." in log
        assert "Best validation MSE" in log

    def test_stdout_restored_even_if_loader_crashes(self, tmp_path,
                                                    monkeypatch):
        import sys

        def boom(*a, **k):
            raise FileNotFoundError("no such csv")

        monkeypatch.setattr(hpo, "load_training_data", boom)
        monkeypatch.setattr(hpo, "set_seed", lambda *a, **k: None)

        original_stdout = sys.stdout
        with pytest.raises(FileNotFoundError):
            hpo.main(["-d", "missing.csv", "-o", str(tmp_path / "out")])
        assert sys.stdout is original_stdout