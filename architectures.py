import warnings

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# -----------------------------------------------------------------------------
#                               FCNN
# -----------------------------------------------------------------------------
class FCNN(nn.Module):
    """Fully connected feed-forward network with a built-in training loop.

    Architecture: input_dim -> [n_neurons] * n_layers -> output_dim,
    with ReLU activations on every layer except the last.

    After calling ``.fit()`` the following attributes hold the per-epoch
    loss history (plain Python lists, one entry per epoch):

    - ``train_mse_loss``   : mean data-fit loss on the training set
    - ``train_reg_loss``   : raw L1 norm of the parameters (NOT multiplied
                             by ``weight_decay``), averaged over the batches
                             of the epoch
    - ``train_total_loss`` : ``train_mse_loss + weight_decay * train_reg_loss``
    - ``test_mse_loss``    : mean data-fit loss on the test set
    - ``test_reg_loss``    : raw L1 norm of the parameters at the end of the
                             epoch (the model is the same for train and test,
                             so this is simply the post-update L1 norm)
    - ``test_total_loss``  : ``test_mse_loss + weight_decay * test_reg_loss``

    The ``weight_decay`` used during training is stored in ``self.weight_decay``.
    Successive calls to ``.fit()`` keep appending to the histories, so you can
    resume training without losing the previous curves.
    """

    def __init__(self, input_dim, n_layers, n_neurons, output_dim):
        super().__init__()

        self.init_params = [input_dim, n_layers, n_neurons, output_dim]
        self.input_dim = input_dim
        self.n_layers = n_layers
        self.n_neurons = n_neurons
        self.output_dim = output_dim

        dims = [input_dim] + [n_neurons] * n_layers + [output_dim]
        self.layers = nn.ModuleList(
            nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)
        )

        # Training configuration / history (filled in by .fit())
        self.weight_decay = 0.0
        self.train_mse_loss = []
        self.train_reg_loss = []
        self.train_total_loss = []
        self.test_mse_loss = []
        self.test_reg_loss = []
        self.test_total_loss = []

    # ------------------------------------------------------------------ utils
    def _initialize_weights(self, seed=None):
        """Re-initialize all linear layers.

        Note: the previous manual code (kaiming_uniform with a=sqrt(5) +
        uniform biases with bound 1/sqrt(fan_in)) is exactly what PyTorch's
        default ``reset_parameters()`` does, so we just call that.
        """
        if seed is not None:
            torch.manual_seed(seed)
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()

    def _l1_norm(self):
        """Raw L1 norm of all parameters (weights and biases)."""
        return sum(param.abs().sum() for param in self.parameters())

    @staticmethod
    def _resolve_device(use_gpu):
        if use_gpu:
            if torch.cuda.is_available():
                return torch.device("cuda")
            if torch.backends.mps.is_available():
                return torch.device("mps")
            warnings.warn("No GPU found on this system, training on CPU.")
        return torch.device("cpu")

    # ---------------------------------------------------------------- forward
    def forward(self, x):
        for layer in self.layers[:-1]:  # no activation on the last layer
            x = torch.relu(layer(x))
        return self.layers[-1](x)

    @torch.no_grad()
    def predict(self, x):
        """Forward pass in eval mode, without tracking gradients."""
        self.eval()
        device = next(self.parameters()).device
        return self(x.to(device)).cpu()

    # -------------------------------------------------------------------- fit
    def fit(
        self,
        data,
        epochs=1000,
        criterion=None,
        optimizer=None,
        weight_decay=0.0,
        batch_size=None,
        learning_rate=0.01,
        shuffle=True,
        verbose=True,
        n_prints=20,
        use_gpu=True,
        plot_loss=False,
    ):
        """Train the model.

        Parameters
        ----------
        data : dict with 'x_train', 'y_train' and optionally 'x_test', 'y_test'
               as 2-D tensors.
        weight_decay : coefficient of the L1 penalty added to the loss.
                       (Stored in ``self.weight_decay``. Note this is an L1
                       penalty, unlike the ``weight_decay`` argument of PyTorch
                       optimizers, which is L2.)
        criterion : loss with ``reduction='mean'`` (default: ``nn.MSELoss()``).
        batch_size : defaults to full-batch training.
        shuffle : whether to reshuffle the training set every epoch.
        """
        self.weight_decay = weight_decay

        if criterion is None:
            criterion = nn.MSELoss()
        if optimizer is None:
            optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        device = self._resolve_device(use_gpu)
        self.to(device)

        # ---- DataLoaders -----------------------------------------------------
        train_dataset = TensorDataset(data["x_train"], data["y_train"])
        if batch_size is None:
            batch_size = len(train_dataset)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=shuffle
        )

        has_test = "x_test" in data and "y_test" in data
        if has_test:
            test_dataset = TensorDataset(data["x_test"], data["y_test"])
            test_loader = DataLoader(
                test_dataset,
                batch_size=min(batch_size, len(test_dataset)),
                shuffle=False,
            )

        print_every = max(1, epochs // n_prints)

        # ---- Training loop ---------------------------------------------------
        for epoch in range(epochs):
            self.train()
            running_mse = 0.0
            running_reg = 0.0
            n_samples = 0
            n_batches = 0

            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                optimizer.zero_grad()
                outputs = self(x_batch)

                mse = criterion(outputs, y_batch)
                reg = self._l1_norm()
                loss = mse + weight_decay * reg

                loss.backward()
                optimizer.step()

                running_mse += mse.item() * x_batch.size(0)
                running_reg += reg.item()
                n_samples += x_batch.size(0)
                n_batches += 1

            epoch_mse = running_mse / n_samples
            epoch_reg = running_reg / n_batches  # params change each step -> average
            self.train_mse_loss.append(epoch_mse)
            self.train_reg_loss.append(epoch_reg)
            self.train_total_loss.append(epoch_mse + weight_decay * epoch_reg)

            # ---- Evaluation --------------------------------------------------
            if has_test:
                self.eval()
                running_mse = 0.0
                n_samples = 0
                with torch.no_grad():
                    for x_batch, y_batch in test_loader:
                        x_batch = x_batch.to(device)
                        y_batch = y_batch.to(device)
                        mse = criterion(self(x_batch), y_batch)
                        running_mse += mse.item() * x_batch.size(0)
                        n_samples += x_batch.size(0)
                    test_reg = self._l1_norm().item()

                test_mse = running_mse / n_samples
                self.test_mse_loss.append(test_mse)
                self.test_reg_loss.append(test_reg)
                self.test_total_loss.append(test_mse + weight_decay * test_reg)

            # ---- Logging -----------------------------------------------------
            if verbose and (epoch + 1) % print_every == 0:
                msg = (
                    f"Epoch [{epoch + 1}/{epochs}] "
                    f"train MSE: {self.train_mse_loss[-1]:.6f}"
                )
                if has_test:
                    msg += f" | test MSE: {self.test_mse_loss[-1]:.6f}"
                if weight_decay > 0:
                    msg += f" | L1 norm: {self.train_reg_loss[-1]:.4f}"
                print(msg)

        if plot_loss:
            self.plot_losses()

    # ------------------------------------------------------------------- plot
    def plot_losses(self, log_scale=True):
        """Plot the per-epoch loss histories recorded during fit()."""
        if not self.train_total_loss:
            raise RuntimeError("No training history found. Call .fit() first.")

        epochs = range(1, len(self.train_total_loss) + 1)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(epochs, self.train_total_loss, label="train total", color="C0")
        ax.plot(epochs, self.train_mse_loss, "--", label="train MSE", color="C0")
        if self.test_total_loss:
            ax.plot(epochs, self.test_total_loss, label="test total", color="C1")
            ax.plot(epochs, self.test_mse_loss, "--", label="test MSE", color="C1")
        if log_scale:
            ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        plt.show()
        return fig, ax