"""Tests for train_ANN.py.

Run with:  pytest test_train_ann.py -v

No real dataset CSV is needed: ``load_training_data`` is monkeypatched with
small synthetic tensors, and ``best_params.json`` is fabricated in a tmp dir.
The save/load round-trip uses a real (tiny) FCNN trained for a few epochs, so
the checkpoint genuinely reconstructs into a working model.
"""

import json
import os

import numpy as np
import pytest
import torch

import train_ann as tann


# --------------------------------------------------------------- fixtures
@pytest.fixture
def fake_split():
    """Tiny synthetic dataset using the loader's 'x_val'/'y_val' convention.
    Has tann.INPUT_DIM feature columns so a real FCNN can train on it."""
    torch.manual_seed(0)
    x = torch.randn(120, tann.INPUT_DIM)
    y = (0.5 * x[:, :1] + 0.1).contiguous()
    return {"x_train": x[:90], "y_train": y[:90],
            "x_val": x[90:], "y_val": y[90:]}


@pytest.fixture
def best_params_file(tmp_path):
    """Write a best_params.json exactly as optimization_ANN.save_results would,
    including the integer hparams stored as JSON numbers and a val_loss field."""
    payload = {"n_layers": 2, "n_neurons": 8, "batch_size": 16,
               "lr": 5e-3, "weight_decay": 1e-5, "val_loss": 0.123}
    path = tmp_path / "best_params.json"
    path.write_text(json.dumps(payload, indent=4))
    return path


class DummyFCNN:
    """Fast stand-in for the real network. Records constructor/fit arguments
    and fakes loss histories. state_dict() returns a real (tiny) tensor dict
    so save_model() can serialise it; we never reload a DummyFCNN checkpoint."""

    last_instance = None

    def __init__(self, input_dim, n_layers, n_neurons, output_dim):
        self.init_args = dict(input_dim=input_dim, n_layers=n_layers,
                              n_neurons=n_neurons, output_dim=output_dim)
        self.init_params = [input_dim, n_layers, n_neurons, output_dim]
        self.fit_kwargs = None
        self.train_mse_loss = []
        self.test_mse_loss = []
        DummyFCNN.last_instance = self

    def fit(self, **kwargs):
        self.fit_kwargs = kwargs
        epochs = kwargs["epochs"]
        self.train_mse_loss = [1.0] * epochs
        self.test_mse_loss = [float(v) for v in np.linspace(1.0, 0.1, epochs)]

    def state_dict(self):
        return {"w": torch.zeros(2, 2)}


# ------------------------------------------------------------ pure helpers
class TestHelpers:
    def test_normalize_maps_val_to_test_keys(self, fake_split):
        out = tann.normalize_split_keys(fake_split)
        assert torch.equal(out["x_test"], fake_split["x_val"])
        assert torch.equal(out["y_test"], fake_split["y_val"])

    def test_normalize_passes_test_keys_through(self, fake_split):
        d = {"x_train": fake_split["x_train"], "y_train": fake_split["y_train"],
             "x_test": fake_split["x_val"], "y_test": fake_split["y_val"]}
        out = tann.normalize_split_keys(d)
        assert torch.equal(out["x_test"], d["x_test"])

    def test_normalize_missing_split_raises(self, fake_split):
        with pytest.raises(KeyError):
            tann.normalize_split_keys({"x_train": fake_split["x_train"],
                                       "y_train": fake_split["y_train"]})

    def test_load_best_params_drops_val_loss(self, best_params_file):
        hp = tann.load_best_params(str(best_params_file))
        assert "val_loss" not in hp
        assert set(hp) == set(tann.REQUIRED_HPARAMS)

    def test_load_best_params_coerces_int_hparams(self, tmp_path):
        # JSON may store integer hparams as floats (e.g. 2.0); they must come
        # back as int, otherwise nn.Linear(2.0, ...) blows up.
        path = tmp_path / "bp.json"
        path.write_text(json.dumps({"n_layers": 2.0, "n_neurons": 8.0,
                                    "batch_size": 16.0, "lr": 1e-3,
                                    "weight_decay": 1e-5, "val_loss": 0.1}))
        hp = tann.load_best_params(str(path))
        for k in ("n_layers", "n_neurons", "batch_size"):
            assert isinstance(hp[k], int)
        assert isinstance(hp["lr"], float)
        assert isinstance(hp["weight_decay"], float)

    def test_load_best_params_missing_key_raises(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"n_layers": 2, "n_neurons": 8}))
        with pytest.raises(KeyError, match="missing hyperparameter"):
            tann.load_best_params(str(path))

    def test_default_best_params_path_uses_hpo_convention(self):
        p = tann.default_best_params_path("flame_speed")
        expected = os.path.join(
            tann.HPO_OUTPUT_DIR,
            tann.HPO_BEST_PARAMS_FILENAME.format(kan_or_ann="ann",
                                                 target="flame_speed"),
        )
        assert p == expected


# ----------------------------------------------------------------- parser
class TestParser:
    def test_defaults(self):
        args = tann.build_parser().parse_args(["-d", "data.csv"])
        assert args.target == "flame_speed"
        assert args.epochs == 1000
        assert args.best_params is None
        assert args.output_dir == tann.MODELS_DIR
        assert args.no_gpu is False

    def test_data_path_is_required(self):
        with pytest.raises(SystemExit):
            tann.build_parser().parse_args([])

    def test_invalid_target_rejected(self):
        with pytest.raises(SystemExit):
            tann.build_parser().parse_args(["-d", "x.csv", "-y", "pressure"])

    def test_no_gpu_flag(self):
        args = tann.build_parser().parse_args(["-d", "x.csv", "--no_gpu"])
        assert args.no_gpu is True


# ------------------------------------------------------ train_final_model
class TestTrainFinalModel:
    def test_builds_with_fixed_dims_and_passes_hparams(self, monkeypatch,
                                                       fake_split):
        monkeypatch.setattr(tann, "FCNN", DummyFCNN)
        hp = {"n_layers": 3, "n_neurons": 32, "batch_size": 16,
              "lr": 1e-3, "weight_decay": 1e-5}
        data = tann.normalize_split_keys(fake_split)

        model = tann.train_final_model(data, hp, epochs=5, use_gpu=False)

        # input/output dims are fixed by the module, hidden dims come from hp
        assert model.init_args == {"input_dim": tann.INPUT_DIM, "n_layers": 3,
                                   "n_neurons": 32, "output_dim": tann.OUTPUT_DIM}
        assert model.fit_kwargs["learning_rate"] == pytest.approx(1e-3)
        assert model.fit_kwargs["weight_decay"] == pytest.approx(1e-5)
        assert model.fit_kwargs["batch_size"] == 16
        assert model.fit_kwargs["epochs"] == 5
        assert model.fit_kwargs["use_gpu"] is False
        assert len(model.test_mse_loss) == 5


# ----------------------------------------------------------- save / load
class TestSaveLoad:
    def _train_tiny(self, fake_split):
        hp = {"n_layers": 1, "n_neurons": 8, "batch_size": 32,
              "lr": 1e-2, "weight_decay": 0.0}
        data = tann.normalize_split_keys(fake_split)
        model = tann.train_final_model(data, hp, epochs=3, use_gpu=False)
        return model, hp

    def test_round_trip_predicts(self, tmp_path, fake_split):
        model, hp = self._train_tiny(fake_split)
        path = str(tmp_path / "model.pt")
        tann.save_model(model, path, "flame_speed", hp, epochs=3)

        loaded, ckpt = tann.load_model(path)
        assert ckpt["init_params"] == [tann.INPUT_DIM, 1, 8, tann.OUTPUT_DIM]
        assert ckpt["hyperparameters"] == hp
        out = loaded.predict(torch.randn(5, tann.INPUT_DIM))
        assert tuple(out.shape) == (5, tann.OUTPUT_DIM)

    def test_loaded_model_matches_original_outputs(self, tmp_path, fake_split):
        model, hp = self._train_tiny(fake_split)
        path = str(tmp_path / "model.pt")
        tann.save_model(model, path, "flame_speed", hp, epochs=3)
        loaded, _ = tann.load_model(path)

        x = torch.randn(7, tann.INPUT_DIM)
        assert torch.allclose(model.predict(x), loaded.predict(x), atol=1e-6)

    def test_checkpoint_contains_metadata_and_histories(self, tmp_path,
                                                        fake_split):
        model, hp = self._train_tiny(fake_split)
        path = str(tmp_path / "model.pt")
        tann.save_model(model, path, "density_ratio", hp, epochs=3)

        ckpt = torch.load(path, weights_only=False)
        assert ckpt["target"] == "density_ratio"
        assert ckpt["epochs"] == 3
        assert ckpt["seed"] == tann.SEED
        assert len(ckpt["train_mse_loss"]) == 3
        assert len(ckpt["test_mse_loss"]) == 3
        assert ckpt["final_val_loss"] == pytest.approx(model.test_mse_loss[-1])

    def test_saved_state_dict_is_on_cpu(self, tmp_path, fake_split):
        model, hp = self._train_tiny(fake_split)
        path = str(tmp_path / "model.pt")
        tann.save_model(model, path, "flame_speed", hp, epochs=3)
        ckpt = torch.load(path, weights_only=False)
        assert all(v.device.type == "cpu" for v in ckpt["state_dict"].values())

    def test_load_model_returns_eval_mode(self, tmp_path, fake_split):
        model, hp = self._train_tiny(fake_split)
        path = str(tmp_path / "model.pt")
        tann.save_model(model, path, "flame_speed", hp, epochs=3)
        loaded, _ = tann.load_model(path)
        assert loaded.training is False


# --------------------------------------------------------------- main()
class TestMain:
    def test_main_joins_data_dir_with_data_path(self, tmp_path, monkeypatch,
                                                fake_split, best_params_file):
        """Regression test for the data-path handling: load_training_data must
        receive os.path.join(DATA_DIR, data_path), not the bare data_path."""
        captured = {}

        def recorder(path, target=None, seed=None):
            captured["path"] = path
            captured["target"] = target
            return fake_split

        monkeypatch.setattr(tann, "load_training_data", recorder)
        monkeypatch.setattr(tann, "set_seed", lambda *a, **k: None)
        monkeypatch.setattr(tann, "FCNN", DummyFCNN)  # fast, not reloaded

        tann.main(["-d", "fake.csv",
                   "-b", str(best_params_file),
                   "-o", str(tmp_path / "models"),
                   "-e", "4", "--no_gpu"])

        assert captured["path"] == os.path.join(tann.DATA_DIR, "fake.csv")
        assert captured["target"] == "flame_speed"

    def test_main_produces_loadable_model_file(self, tmp_path, monkeypatch,
                                               fake_split, best_params_file):
        """Full pipeline with a real (tiny) FCNN: the file main() writes must
        reload through load_model and predict."""
        monkeypatch.setattr(tann, "load_training_data",
                            lambda *a, **k: fake_split)
        monkeypatch.setattr(tann, "set_seed",
                            lambda seed, **k: torch.manual_seed(seed))

        out_dir = tmp_path / "models"
        path = tann.main(["-d", "fake.csv",
                          "-b", str(best_params_file),
                          "-o", str(out_dir),
                          "-e", "3", "--no_gpu"])

        expected = out_dir / tann.MODEL_FILENAME.format(
            kan_or_ann="ann", target="flame_speed")
        assert path == str(expected)
        assert expected.exists()

        loaded, ckpt = tann.load_model(path)
        assert ckpt["target"] == "flame_speed"
        out = loaded.predict(torch.randn(5, tann.INPUT_DIM))
        assert tuple(out.shape) == (5, tann.OUTPUT_DIM)

    def test_main_falls_back_to_default_best_params_path(self, tmp_path,
                                                        monkeypatch, fake_split):
        """When --best_params is omitted, main() must read from
        default_best_params_path()."""
        captured = {}

        def fake_load_best(path):
            captured["path"] = path
            return {"n_layers": 1, "n_neurons": 8, "batch_size": 16,
                    "lr": 1e-2, "weight_decay": 0.0}

        monkeypatch.setattr(tann, "load_best_params", fake_load_best)
        monkeypatch.setattr(tann, "load_training_data",
                            lambda *a, **k: fake_split)
        monkeypatch.setattr(tann, "set_seed", lambda *a, **k: None)
        monkeypatch.setattr(tann, "FCNN", DummyFCNN)

        tann.main(["-d", "fake.csv", "-o", str(tmp_path / "models"),
                   "-e", "3", "--no_gpu"])

        assert captured["path"] == tann.default_best_params_path("flame_speed")