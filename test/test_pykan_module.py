"""Tests for the (locally modified) pykan MLP class and the stock MultKAN class.

Both live in the editable `pykan` install:
    from pykan.kan.MLP import MLP
    from pykan.kan.MultKAN import MultKAN

Note on the local MLP modifications this suite assumes (vs. stock pykan):
  * fit() reads the dataset with the keys
    'x_train', 'y_train', 'x_val', 'test_label'
  * fit() records the *raw* loss (MSE), not its square root (RMSE)
  * fit() exposes the running histories as `train_mse_loss` / `test_mse_loss`

The MultKAN tests target stock pykan behaviour:
  * fit() reads 'train_input'/'train_label'/'test_input'/'test_label'
  * fit() records RMSE in results['train_loss']/['test_loss']
  * capacity lives in learnable splines -> grid (nodes) and k (order) matter

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
from pykan.kan.MultKAN import MultKAN
from pykan.kan.KANLayer import KANLayer


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


# =====================================================================
#                       MultKAN (KAN) tests
# =====================================================================
# Stock pykan KAN (KAN is MultKAN). A few behaviours differ from the MLP and
# drive the choices below:
#   * dataset keys are 'train_input'/'train_label'/'test_input'/'test_label'
#   * recorded losses are RMSE (sqrt of MSE), not MSE
#   * forward stores acts of length depth + 1 (the MLP stores depth)
#   * capacity is in splines -> grid (nodes) and k (order) change param count
#   * seed init is NOT fully reproducible (spline coefficients vary), so we
#     only assert that *different* seeds differ
#   * refine() needs a prior forward pass (it reads cache_data)
#   * prune() writes history.txt into ckpt_path, so the dir must exist
# KAN models are therefore built with auto_save=False and a writable, existing
# ckpt_path (pytest's tmp_path) so disk-logging methods don't fail.


@pytest.fixture
def kan(tmp_path):
    return MultKAN(width=[3, 5, 2], grid=5, k=3, seed=0,
                   auto_save=False, ckpt_path=str(tmp_path))


@pytest.fixture
def kan_data():
    """Small synthetic problem in the stock KAN key scheme."""
    torch.manual_seed(0)
    x = torch.randn(200, 3)
    w = torch.randn(3, 2)
    y = x @ w + 0.05 * torch.randn(200, 2)
    return {
        "train_input": x[:150],
        "train_label": y[:150],
        "test_input": x[150:],
        "test_label": y[150:],
    }


def quick_kan_fit(model, data, **kwargs):
    """fit() with fast defaults for testing (LBFGS, few steps, no reg)."""
    defaults = dict(opt="LBFGS", steps=5, lamb=0.0)
    defaults.update(kwargs)
    return model.fit(data, **defaults)


def _nparams(m):
    return sum(p.numel() for p in m.parameters())


# ------------------------------------------------------- KAN architecture
class TestKANArchitecture:
    def test_number_of_layers(self, kan):
        # depth == len(width_in) - 1 KANLayers
        assert kan.depth == len(kan.width_in) - 1
        assert len(kan.act_fun) == kan.depth

    def test_width_in_stored(self, kan):
        assert kan.width_in == [3, 5, 2]
        assert kan.depth == 2

    def test_layers_are_kanlayers(self, kan):
        assert all(isinstance(l, KANLayer) for l in kan.act_fun)

    def test_grid_and_spline_order_stored(self, kan):
        assert kan.grid == 5
        assert kan.act_fun[0].num == 5   # grid intervals ("nodes")
        assert kan.act_fun[0].k == 3     # spline order

    def test_layer_count_for_deeper_net(self, tmp_path):
        m = MultKAN(width=[4, 10, 3, 2], grid=3, k=3, seed=0,
                    auto_save=False, ckpt_path=str(tmp_path))
        assert m.width_in == [4, 10, 3, 2]
        assert len(m.act_fun) == 3


# -------------------------------------------------------- KAN forward pass
class TestKANForward:
    def test_output_shape(self, kan):
        assert kan(torch.randn(8, 3)).shape == (8, 2)

    def test_single_sample(self, kan):
        assert kan(torch.randn(1, 3)).shape == (1, 2)

    def test_forward_populates_acts(self, kan):
        kan(torch.randn(20, 3))
        # KAN stores one activation per layer boundary (depth + 1)
        assert len(kan.acts) == kan.depth + 1


# ------------------------------------------------------ KAN initialization
class TestKANInitialization:
    def test_different_seeds_differ(self, tmp_path):
        m1 = MultKAN(width=[3, 5, 2], grid=5, k=3, seed=1,
                     auto_save=False, ckpt_path=str(tmp_path))
        m2 = MultKAN(width=[3, 5, 2], grid=5, k=3, seed=2,
                     auto_save=False, ckpt_path=str(tmp_path))
        assert any(not torch.equal(p1, p2)
                   for p1, p2 in zip(m1.parameters(), m2.parameters()))


# ----------------------------------------------------------- KAN splines
class TestKANSplines:
    def test_grid_increases_parameter_count(self, tmp_path):
        coarse = MultKAN(width=[3, 5, 2], grid=5, k=3, seed=0,
                         auto_save=False, ckpt_path=str(tmp_path))
        fine = MultKAN(width=[3, 5, 2], grid=20, k=3, seed=0,
                       auto_save=False, ckpt_path=str(tmp_path))
        # finer grid = more spline coefficients
        assert _nparams(fine) > _nparams(coarse)

    def test_refine_increases_grid(self, kan):
        kan(torch.randn(30, 3))          # refine() needs cache_data
        refined = kan.refine(12)
        assert refined.grid == 12
        assert _nparams(refined) > _nparams(kan)


# --------------------------------------------------------------- KAN fit
class TestKANFit:
    def test_results_keys(self, kan, kan_data):
        res = quick_kan_fit(kan, kan_data)
        for key in ("train_loss", "test_loss", "reg"):
            assert key in res

    def test_history_lengths(self, kan, kan_data):
        res = quick_kan_fit(kan, kan_data, steps=6)
        for key in ("train_loss", "test_loss", "reg"):
            assert len(res[key]) == 6

    def test_loss_decreases_on_learnable_problem(self, kan, kan_data):
        res = quick_kan_fit(kan, kan_data, steps=20)
        assert float(res["train_loss"][-1]) < float(res["train_loss"][0])

    def test_adam_optimizer_runs(self, kan, kan_data):
        res = quick_kan_fit(kan, kan_data, opt="Adam", steps=3, lr=0.01)
        assert len(res["train_loss"]) == 3

    def test_metrics_are_recorded(self, kan, kan_data):
        def custom_metric():
            return torch.tensor(1.234)

        res = quick_kan_fit(kan, kan_data, steps=4, metrics=[custom_metric])
        assert "custom_metric" in res
        assert len(res["custom_metric"]) == 4

    def test_loss_values_are_finite(self, kan, kan_data):
        res = quick_kan_fit(kan, kan_data, steps=3)
        for key in ("train_loss", "test_loss", "reg"):
            assert all(np.isfinite(float(v)) for v in res[key])

    def test_training_updates_parameters(self, kan, kan_data):
        before = [p.clone() for p in kan.parameters()]
        quick_kan_fit(kan, kan_data, steps=3)
        assert any(not torch.equal(b, a)
                   for b, a in zip(before, kan.parameters()))


# ----------------------------------------------------- KAN attribution
class TestKANAttribution:
    def test_attribute_score_lengths(self, kan):
        kan(torch.randn(20, 3))
        kan.attribute()
        assert len(kan.node_scores) == kan.depth + 1
        assert len(kan.edge_scores) == kan.depth

    def test_feature_score_shape(self, kan):
        kan(torch.randn(20, 3))
        kan.attribute()
        # one importance score per input feature
        assert tuple(kan.feature_score.shape) == (kan.width_in[0],)


# ------------------------------------------------- KAN prune & plotting
class TestKANPruneAndPlot:
    def test_prune_returns_kan(self, kan, kan_data):
        quick_kan_fit(kan, kan_data, steps=5)
        pruned = kan.prune()
        assert isinstance(pruned, MultKAN)
        assert pruned.depth == kan.depth

    def test_plot_runs(self, kan, tmp_path):
        kan(torch.randn(20, 3))  # plot needs activations
        kan.plot(folder=str(tmp_path / "figs"), scale=0.3)
        assert plt.get_fignums()  # a figure was created
        plt.close("all")