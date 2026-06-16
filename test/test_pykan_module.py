"""Tests for the (locally modified) pykan MLP class.

The MLP lives in the editable `pykan` install and is imported as
`from kan.MLP import MLP`.

Note on the local modifications this suite assumes (vs. stock pykan):
  * fit() reads the dataset with the keys
    'x_train', 'y_train', 'x_val', 'test_label'
  * fit() records the *raw* loss (MSE), not its square root (RMSE)
  * fit() exposes the running histories as `train_mse_loss` / `test_mse_loss`

Run with:  pytest test_mlp.py -v
"""

import matplotlib

matplotlib.use("Agg")  # headless backend, must be set before pyplot is used

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
import torch.nn as nn

from pykan.kan.MLP import MLP


# ---------------------------------------------------------------- fixtures
@pytest.fixture
def model():
    return MLP(width=[3, 16, 2], seed=0)


@pytest.fixture
def data():
    """Small synthetic linear regression problem (learnable, noisy).

    Uses the modified key scheme: x_train / y_train / x_val / test_label.
    """
    torch.manual_seed(0)
    x = torch.randn(200, 3)
    w = torch.randn(3, 2)
    y = x @ w + 0.05 * torch.randn(200, 2)
    return {
        "x_train": x[:150],
        "y_train": y[:150],
        "x_val": x[150:],
        "test_label": y[150:],
    }


def quick_fit(model, data, **kwargs):
    """fit() with fast, quiet defaults for testing (Adam, few steps)."""
    defaults = dict(opt="Adam", steps=5, lr=0.01, log=1)
    defaults.update(kwargs)
    return model.fit(data, **defaults)


# ----------------------------------------------------------- architecture
class TestArchitecture:
    def test_number_of_layers(self, model):
        # depth == len(width) - 1 Linear modules
        assert model.depth == len(model.width) - 1
        assert len(model.linears) == model.depth

    def test_layer_dimensions(self):
        m = MLP(width=[4, 10, 10, 10, 2], seed=0)
        dims = [(l.in_features, l.out_features) for l in m.linears]
        assert dims == [(4, 10), (10, 10), (10, 10), (10, 2)]

    def test_single_hidden_layer(self):
        m = MLP(width=[5, 8, 1], seed=0)
        dims = [(l.in_features, l.out_features) for l in m.linears]
        assert dims == [(5, 8), (8, 1)]

    def test_all_layers_are_linear(self, model):
        assert all(isinstance(l, nn.Linear) for l in model.linears)

    def test_width_and_depth_stored(self, model):
        assert model.width == [3, 16, 2]
        assert model.depth == 2

    def test_w_property_returns_layer_weights(self, model):
        ws = model.w
        assert len(ws) == model.depth
        for got, layer in zip(ws, model.linears):
            assert got is layer.weight

    def test_initial_state(self, model):
        # nothing has been run through the net yet
        assert model.acts is None
        assert model.cache_data is None
        assert model.save_act is True


# ----------------------------------------------------------- forward pass
class TestForward:
    def test_output_shape(self, model):
        x = torch.randn(7, 3)
        assert model(x).shape == (7, 2)

    def test_single_sample(self, model):
        x = torch.randn(1, 3)
        assert model(x).shape == (1, 2)

    def test_last_layer_has_no_activation(self):
        """Final layer is linear, so outputs must be unbounded below.

        A final SiLU would floor every output near -0.278; raw outputs go
        well below that.
        """
        torch.manual_seed(0)
        m = MLP(width=[2, 8, 1], seed=0)
        out = m(torch.randn(2000, 2) * 3)
        assert (out < -0.5).any()

    def test_forward_caches_input(self, model):
        x = torch.randn(5, 3)
        model(x)
        assert model.cache_data is x


# ------------------------------------------------------------- weight init
class TestInitialization:
    def test_seed_reproducibility(self):
        m1 = MLP(width=[3, 16, 2], seed=123)
        m2 = MLP(width=[3, 16, 2], seed=123)
        for p1, p2 in zip(m1.parameters(), m2.parameters()):
            assert torch.equal(p1, p2)

    def test_different_seeds_differ(self):
        m1 = MLP(width=[3, 16, 2], seed=1)
        m2 = MLP(width=[3, 16, 2], seed=2)
        assert any(not torch.equal(p1, p2)
                   for p1, p2 in zip(m1.parameters(), m2.parameters()))


# ------------------------------------------------- activations & caching
class TestActivations:
    def test_save_act_true_populates_acts(self, model):
        model(torch.randn(10, 3))
        assert len(model.acts) == model.depth

    def test_save_act_false_leaves_acts_empty(self):
        m = MLP(width=[3, 8, 2], seed=0, save_act=False)
        m(torch.randn(10, 3))
        assert m.acts == []

    def test_get_act_from_dict(self, model, data):
        model.get_act(data)  # dict path reads data['x_train']
        assert len(model.acts) == model.depth

    def test_get_act_uses_cached_data(self, model):
        model(torch.randn(10, 3))
        model.acts = None
        model.get_act()  # None path falls back to cache_data
        assert len(model.acts) == model.depth

    def test_get_act_without_data_raises(self, model):
        with pytest.raises(Exception):
            model.get_act()  # no argument, no cached data


# --------------------------------------------------------------- training
class TestFit:
    def test_results_keys(self, model, data):
        res = quick_fit(model, data)
        for key in ("train_loss", "test_loss", "reg"):
            assert key in res

    def test_history_lengths(self, model, data):
        res = quick_fit(model, data, steps=7)
        for key in ("train_loss", "test_loss", "reg"):
            assert len(res[key]) == 7

    def test_sets_loss_attributes(self, model, data):
        quick_fit(model, data, steps=6)
        assert len(model.train_mse_loss) == 6
        assert len(model.test_mse_loss) == 6

    def test_reg_zero_while_lamb_zero(self, model, data):
        """With lamb=0, save_act is disabled during training, so the recorded
        reg is exactly 0 for every step except the final one (where save_act
        is restored)."""
        res = quick_fit(model, data, steps=5, lamb=0.0)
        assert all(float(r) == 0.0 for r in res["reg"][:-1])

    def test_reg_positive_when_lamb_positive(self, model, data):
        res = quick_fit(model, data, steps=5, lamb=0.01)
        assert all(float(r) > 0.0 for r in res["reg"])

    def test_loss_decreases_on_learnable_problem(self, model, data):
        res = quick_fit(model, data, steps=150, lr=0.02)
        assert float(res["train_loss"][-1]) < 0.3 * float(res["train_loss"][0])
        assert float(res["test_loss"][-1]) < float(res["test_loss"][0])

    def test_lbfgs_optimizer_runs(self, model, data):
        res = model.fit(data, opt="LBFGS", steps=3, lr=0.5)
        assert len(res["train_loss"]) == 3

    def test_custom_loss_fn(self, model, data):
        mae = lambda p, t: torch.mean(torch.abs(p - t))
        res = quick_fit(model, data, steps=3, loss_fn=mae)
        assert len(res["train_loss"]) == 3

    def test_metrics_are_recorded(self, model, data):
        def custom_metric():
            return torch.tensor(1.234)

        res = quick_fit(model, data, steps=4, metrics=[custom_metric])
        assert "custom_metric" in res
        assert len(res["custom_metric"]) == 4

    def test_batch_smaller_than_dataset(self, model, data):
        res = quick_fit(model, data, steps=3, batch=32)
        assert len(res["train_loss"]) == 3

    def test_training_updates_parameters(self, model, data):
        before = [p.clone() for p in model.parameters()]
        quick_fit(model, data, steps=3)
        assert any(not torch.equal(b, a)
                   for b, a in zip(before, model.parameters()))

    def test_loss_values_are_finite(self, model, data):
        res = quick_fit(model, data, steps=3, lamb=0.01)
        for key in ("train_loss", "test_loss", "reg"):
            assert all(np.isfinite(float(v)) for v in res[key])

    def test_save_act_restored_after_fit(self, model, data):
        """fit() temporarily disables save_act when lamb=0 but must restore
        the original value by the time it returns."""
        assert model.save_act is True
        quick_fit(model, data, steps=4, lamb=0.0)
        assert model.save_act is True


# ----------------------------------------------------------- attribution
class TestAttribution:
    def test_attribute_score_lengths(self, model):
        model(torch.randn(20, 3))
        model.attribute()
        # one node-score vector per layer boundary, one edge-score per layer
        assert len(model.node_scores) == model.depth + 1
        assert len(model.edge_scores) == model.depth

    def test_attribute_sets_backward_alias(self, model):
        model(torch.randn(20, 3))
        model.attribute()
        assert model.wa_backward is model.edge_scores

    def test_connection_cost_is_nonnegative_scalar(self, model):
        cc = model.connection_cost
        assert isinstance(cc, torch.Tensor)
        assert cc.ndim == 0
        assert cc.item() >= 0.0


# ----------------------------------------------------------------- swaps
class TestSwap:
    def test_swap_is_reversible(self, model):
        snapshot = ([l.weight.clone() for l in model.linears]
                    + [l.bias.clone() for l in model.linears])
        model.swap(1, 0, 3)
        model.swap(1, 0, 3)  # swapping the same pair again undoes it
        after = ([l.weight.clone() for l in model.linears]
                 + [l.bias.clone() for l in model.linears])
        assert all(torch.equal(b, a) for b, a in zip(snapshot, after))

    def test_swap_changes_parameters(self, model):
        before = [l.weight.clone() for l in model.linears]
        model.swap(1, 0, 3)
        after = [l.weight for l in model.linears]
        assert any(not torch.equal(b, a) for b, a in zip(before, after))

    def test_auto_swap_runs(self):
        m = MLP(width=[3, 6, 2], seed=0)
        m(torch.randn(20, 3))
        m.auto_swap()  # should iterate hidden layers without error


# ---------------------------------------------------------------- plotting
class TestPlotting:
    def test_plot_metric_w(self, model):
        model(torch.randn(20, 3))
        model.plot(metric="w")
        assert plt.get_fignums()  # a figure was created
        plt.close("all")

    def test_plot_metric_act(self, model):
        model(torch.randn(20, 3))  # populates wa_forward
        model.plot(metric="act")
        assert plt.get_fignums()
        plt.close("all")

    def test_plot_metric_fa(self, model):
        model(torch.randn(20, 3))  # attribute() is called inside plot()
        model.plot(metric="fa")
        assert plt.get_fignums()
        plt.close("all")

    def test_plot_invalid_metric_raises(self, model):
        model(torch.randn(20, 3))
        with pytest.raises(Exception):
            model.plot(metric="not_a_metric")
        plt.close("all")