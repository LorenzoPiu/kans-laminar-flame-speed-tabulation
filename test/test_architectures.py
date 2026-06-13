"""Tests for the FCNN module (fcnn.py).

Run with:  pytest test_fcnn.py -v
"""

import matplotlib

matplotlib.use("Agg")  # headless backend, must be set before pyplot is used

import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn

from architectures import FCNN


# ---------------------------------------------------------------- fixtures
@pytest.fixture
def model():
    return FCNN(input_dim=3, n_layers=2, n_neurons=16, output_dim=2)


@pytest.fixture
def data():
    """Small synthetic linear regression problem (learnable, noisy)."""
    torch.manual_seed(0)
    x = torch.randn(200, 3)
    w = torch.randn(3, 2)
    y = x @ w + 0.05 * torch.randn(200, 2)
    return {
        "x_train": x[:150],
        "y_train": y[:150],
        "x_test": x[150:],
        "y_test": y[150:],
    }


def quick_fit(model, data, **kwargs):
    """fit() with fast, quiet defaults for testing."""
    defaults = dict(epochs=5, use_gpu=False, verbose=True, plot_loss=False)
    defaults.update(kwargs)
    model.fit(data, **defaults)


# ----------------------------------------------------------- architecture
class TestArchitecture:
    def test_number_of_layers(self, model):
        # n_layers hidden layers -> n_layers + 1 Linear modules
        assert len(model.layers) == model.n_layers + 1

    def test_layer_dimensions(self):
        m = FCNN(input_dim=4, n_layers=3, n_neurons=10, output_dim=2)
        dims = [(l.in_features, l.out_features) for l in m.layers]
        assert dims == [(4, 10), (10, 10), (10, 10), (10, 2)]

    def test_single_hidden_layer(self):
        m = FCNN(input_dim=5, n_layers=1, n_neurons=8, output_dim=1)
        dims = [(l.in_features, l.out_features) for l in m.layers]
        assert dims == [(5, 8), (8, 1)]

    def test_all_layers_are_linear(self, model):
        assert all(isinstance(l, nn.Linear) for l in model.layers)

    def test_init_params_stored(self, model):
        assert model.init_params == [3, 2, 16, 2]
        assert (model.input_dim, model.n_layers, model.n_neurons,
                model.output_dim) == (3, 2, 16, 2)

    def test_histories_start_empty(self, model):
        for attr in ("train_mse_loss", "train_reg_loss", "train_total_loss",
                     "test_mse_loss", "test_reg_loss", "test_total_loss"):
            assert getattr(model, attr) == []
        assert model.weight_decay == 0.0


# ----------------------------------------------------------- forward pass
class TestForward:
    def test_output_shape(self, model):
        x = torch.randn(7, 3)
        assert model(x).shape == (7, 2)

    def test_single_sample(self, model):
        x = torch.randn(1, 3)
        assert model(x).shape == (1, 2)

    def test_last_layer_has_no_activation(self):
        """Outputs must be able to go negative (no final ReLU)."""
        torch.manual_seed(0)
        m = FCNN(input_dim=2, n_layers=1, n_neurons=8, output_dim=1)
        out = m(torch.randn(1000, 2))
        assert (out < 0).any()

    def test_predict_shape_and_device(self, model):
        x = torch.randn(5, 3)
        out = model.predict(x)
        assert out.shape == (5, 2)
        assert out.device.type == "cpu"

    def test_predict_no_grad(self, model):
        out = model.predict(torch.randn(5, 3))
        assert not out.requires_grad


# ------------------------------------------------------- weight init
class TestInitialization:
    def test_seed_reproducibility(self):
        m1 = FCNN(3, 2, 16, 2)
        m2 = FCNN(3, 2, 16, 2)
        m1._initialize_weights(seed=123)
        m2._initialize_weights(seed=123)
        for p1, p2 in zip(m1.parameters(), m2.parameters()):
            assert torch.equal(p1, p2)

    def test_different_seeds_differ(self):
        m1 = FCNN(3, 2, 16, 2)
        m2 = FCNN(3, 2, 16, 2)
        m1._initialize_weights(seed=1)
        m2._initialize_weights(seed=2)
        assert any(not torch.equal(p1, p2)
                   for p1, p2 in zip(m1.parameters(), m2.parameters()))


# --------------------------------------------------------------- training
class TestFit:
    def test_history_lengths(self, model, data):
        quick_fit(model, data, epochs=7)
        for attr in ("train_mse_loss", "train_reg_loss", "train_total_loss",
                     "test_mse_loss", "test_reg_loss", "test_total_loss"):
            assert len(getattr(model, attr)) == 7

    def test_weight_decay_stored(self, model, data):
        quick_fit(model, data, weight_decay=1e-3)
        assert model.weight_decay == 1e-3

    def test_total_is_mse_plus_weighted_reg(self, model, data):
        wd = 1e-4
        quick_fit(model, data, weight_decay=wd, epochs=4)
        for mse, reg, total in zip(model.train_mse_loss,
                                   model.train_reg_loss,
                                   model.train_total_loss):
            assert total == pytest.approx(mse + wd * reg)
        for mse, reg, total in zip(model.test_mse_loss,
                                   model.test_reg_loss,
                                   model.test_total_loss):
            assert total == pytest.approx(mse + wd * reg)

    def test_reg_loss_is_raw_l1_norm(self, model, data):
        """test_reg_loss must equal the post-epoch L1 norm of the parameters,
        with no weight_decay factor applied."""
        quick_fit(model, data, weight_decay=1e-3, epochs=3)
        expected = sum(p.abs().sum() for p in model.parameters()).item()
        assert model.test_reg_loss[-1] == pytest.approx(expected, rel=1e-6)

    def test_zero_weight_decay_total_equals_mse(self, model, data):
        quick_fit(model, data, weight_decay=0.0, epochs=3)
        assert model.train_total_loss == pytest.approx(model.train_mse_loss)

    def test_loss_decreases_on_learnable_problem(self, model, data):
        model._initialize_weights(seed=0)
        quick_fit(model, data, epochs=100, batch_size=32, learning_rate=0.01)
        assert model.train_mse_loss[-1] < 0.1 * model.train_mse_loss[0]
        assert model.test_mse_loss[-1] < model.test_mse_loss[0]

    def test_histories_append_across_fit_calls(self, model, data):
        quick_fit(model, data, epochs=3)
        quick_fit(model, data, epochs=4)
        assert len(model.train_mse_loss) == 7
        assert len(model.test_mse_loss) == 7

    def test_no_test_set_leaves_test_histories_empty(self, model, data):
        train_only = {"x_train": data["x_train"], "y_train": data["y_train"]}
        quick_fit(model, train_only, epochs=3)
        assert len(model.train_mse_loss) == 3
        assert model.test_mse_loss == []
        assert model.test_reg_loss == []
        assert model.test_total_loss == []

    def test_full_batch_default(self, model, data):
        """batch_size=None must not crash (single full batch per epoch)."""
        quick_fit(model, data, epochs=2, batch_size=None)
        assert len(model.train_mse_loss) == 2

    def test_batch_size_larger_than_test_set(self, model, data):
        """Test loader must cap batch size at the test set size."""
        quick_fit(model, data, epochs=2, batch_size=10_000)
        assert len(model.test_mse_loss) == 2

    def test_few_epochs_many_prints_does_not_crash(self, model, data):
        """Regression test: epochs < n_prints used to raise ZeroDivisionError."""
        quick_fit(model, data, epochs=2, n_prints=20, verbose=True)

    def test_verbose_false_prints_nothing(self, model, data, capsys):
        quick_fit(model, data, epochs=5, verbose=False)
        assert capsys.readouterr().out == ""

    def test_verbose_true_prints(self, model, data, capsys):
        quick_fit(model, data, epochs=10, n_prints=2, verbose=True)
        out = capsys.readouterr().out
        assert "train MSE" in out and "test MSE" in out

    def test_custom_criterion(self, model, data):
        quick_fit(model, data, epochs=3, criterion=nn.L1Loss())
        assert len(model.train_mse_loss) == 3

    def test_custom_optimizer(self, model, data):
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        quick_fit(model, data, epochs=3, optimizer=opt)
        assert len(model.train_mse_loss) == 3

    def test_training_updates_parameters(self, model, data):
        before = [p.clone() for p in model.parameters()]
        quick_fit(model, data, epochs=2)
        assert any(not torch.equal(b, a)
                   for b, a in zip(before, model.parameters()))

    def test_loss_values_are_finite_floats(self, model, data):
        quick_fit(model, data, epochs=3, weight_decay=1e-4)
        for attr in ("train_mse_loss", "train_reg_loss", "train_total_loss",
                     "test_mse_loss", "test_reg_loss", "test_total_loss"):
            values = getattr(model, attr)
            assert all(isinstance(v, float) for v in values)
            assert all(torch.isfinite(torch.tensor(v)) for v in values)


# ---------------------------------------------------------------- plotting
class TestPlotting:
    def test_plot_before_fit_raises(self, model):
        with pytest.raises(RuntimeError):
            model.plot_losses()

    def test_plot_after_fit(self, model, data):
        quick_fit(model, data, epochs=3)
        fig, ax = model.plot_losses()
        # train total/MSE + test total/MSE = 4 curves
        assert len(ax.lines) == 4
        plt.close(fig)

    def test_plot_without_test_set(self, model, data):
        train_only = {"x_train": data["x_train"], "y_train": data["y_train"]}
        quick_fit(model, train_only, epochs=3)
        fig, ax = model.plot_losses()
        assert len(ax.lines) == 2
        plt.close(fig)