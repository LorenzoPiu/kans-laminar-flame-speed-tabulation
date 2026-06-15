#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_ANN.py
============
Train a final FCNN with the hyperparameters selected by the Optuna HPO
(``optimization_ANN.py``) and save the trained model to ``MODELS_DIR``.

What it does
------------
- Reads the best hyperparameters from the HPO artefact
  ``best_params.json`` (n_layers, n_neurons, batch_size, lr, weight_decay).
- Loads the laminar-flame dataset via ``load_training_data`` from utils.py.
- Retrains a fresh FCNN with those hyperparameters for ``--epochs`` epochs.
- Saves a checkpoint (state_dict + architecture + metadata + loss curves)
  to ``MODELS_DIR`` so the model can be reloaded with ``load_model``.

Usage
-----
    python train_ANN.py \\
        --data_path  /path/to/dataset.csv \\
        --target     flame_speed \\
        --epochs     10000

By default the best-params file is looked up in the HPO output directory
using the same naming convention as ``optimization_ANN.py``. Override it
with ``--best_params /path/to/best_params.json`` if needed.

Notes
-----
- Input  dimensionality : 6  (Pressure, Temperature, H2, CO, CO2, H2O mass fractions)
- Output dimensionality : 1  (flame speed *or* density ratio, selected via --target)
- Optimiser             : Adam (fixed)
- Seed                  : 42  (set in main() via set_seed)
- Import-safe: nothing runs until ``main()`` is called, so it can be unit-tested.
"""

import argparse
import json
import os

import torch

# ---------------------------------------------------------------------------
# Local imports  (adjust relative paths if needed)
# ---------------------------------------------------------------------------
from architectures import FCNN
from utils import load_training_data, set_seed
from global_vars import (
    MODELS_DIR,
    DATA_DIR,
    HPO_OUTPUT_DIR,
    HPO_BEST_PARAMS_FILENAME,
    MODEL_FILENAME
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED = 42

INPUT_DIM = 6   # Pressure, Temperature, H2, CO, CO2, H2O mass fractions
OUTPUT_DIM = 1  # flame_speed or density_ratio (single scalar target)

# Hyperparameters expected in best_params.json (everything except "val_loss").
REQUIRED_HPARAMS = ("n_layers", "n_neurons", "batch_size", "lr", "weight_decay")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def normalize_split_keys(data: dict) -> dict:
    """Map the data dict to the 'x_test'/'y_test' keys FCNN.fit() expects.

    ``load_training_data`` returns 'x_val'/'y_val'; accept either convention.
    """
    out = {"x_train": data["x_train"], "y_train": data["y_train"]}
    if "x_test" in data and "y_test" in data:
        out["x_test"], out["y_test"] = data["x_test"], data["y_test"]
    elif "x_val" in data and "y_val" in data:
        out["x_test"], out["y_test"] = data["x_val"], data["y_val"]
    else:
        raise KeyError(
            "data must contain a validation split: either "
            "('x_test', 'y_test') or ('x_val', 'y_val')"
        )
    return out


def load_best_params(path: str) -> dict:
    """Read best_params.json and return only the hyperparameters.

    Drops the bookkeeping 'val_loss' field and coerces the integer-valued
    hyperparameters back to ``int`` (JSON may store them as floats).
    """
    with open(path, "r") as fp:
        raw = json.load(fp)

    missing = [k for k in REQUIRED_HPARAMS if k not in raw]
    if missing:
        raise KeyError(
            f"{path} is missing hyperparameter(s): {missing}. "
            f"Expected keys: {list(REQUIRED_HPARAMS)}."
        )

    return {
        "n_layers":     int(raw["n_layers"]),
        "n_neurons":    int(raw["n_neurons"]),
        "batch_size":   int(raw["batch_size"]),
        "lr":           float(raw["lr"]),
        "weight_decay": float(raw["weight_decay"]),
    }


def default_best_params_path(target: str) -> str:
    """Where optimization_ANN.py writes best_params.json for an ANN run."""
    return os.path.join(
        HPO_OUTPUT_DIR,
        HPO_BEST_PARAMS_FILENAME.format(kan_or_ann="ann", target=target),
    )


def train_final_model(data, hparams, epochs, use_gpu=True):
    """Build a fresh FCNN with ``hparams`` and train it on ``data``."""
    model = FCNN(
        input_dim=INPUT_DIM,
        n_layers=hparams["n_layers"],
        n_neurons=hparams["n_neurons"],
        output_dim=OUTPUT_DIM,
    )
    model.fit(
        data=data,
        epochs=epochs,
        weight_decay=hparams["weight_decay"],
        learning_rate=hparams["lr"],
        batch_size=hparams["batch_size"],
        verbose=True,
        use_gpu=use_gpu,
    )
    return model


def save_model(model, path, target, hparams, epochs):
    """Save a self-contained checkpoint that can be reloaded with load_model."""
    final_val_loss = model.test_mse_loss[-1] if model.test_mse_loss else None
    checkpoint = {
        "state_dict":       model.state_dict(),
        "init_params":      model.init_params,  # [in, n_layers, n_neurons, out]
        "hyperparameters":  hparams,
        "target":           target,
        "epochs":           epochs,
        "seed":             SEED,
        "train_mse_loss":   model.train_mse_loss,
        "test_mse_loss":    model.test_mse_loss,
        "final_val_loss":   final_val_loss,
    }
    # CPU tensors so the checkpoint loads anywhere, regardless of training device.
    checkpoint["state_dict"] = {k: v.cpu() for k, v in checkpoint["state_dict"].items()}
    torch.save(checkpoint, path)
    return final_val_loss


def load_model(path, map_location="cpu"):
    """Reconstruct an FCNN from a checkpoint saved by save_model()."""
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    input_dim, n_layers, n_neurons, output_dim = checkpoint["init_params"]
    model = FCNN(input_dim, n_layers, n_neurons, output_dim)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a final FCNN with HPO-selected hyperparameters."
    )
    parser.add_argument(
        "--data_path", "-d",
        type=str,
        required=True,
        help="Path to the laminar-flame dataset CSV file.",
    )
    parser.add_argument(
        "--target", "-y",
        type=str,
        choices=["flame_speed", "density_ratio"],
        default="flame_speed",
        help="Output variable to predict. Default: 'flame_speed'.",
    )
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=10000,
        help="Number of training epochs for the final model. Default: 10000.",
    )
    parser.add_argument(
        "--best_params", "-b",
        type=str,
        default=None,
        help="Path to best_params.json. Default: looked up in HPO_OUTPUT_DIR "
             "using the standard ANN naming convention.",
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default=MODELS_DIR,
        help="Directory where the trained model is saved. Default: MODELS_DIR.",
    )
    parser.add_argument(
        "--no_gpu",
        action="store_true",
        help="Force CPU training even if a GPU is available.",
    )
    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv=None) -> str:
    """Train the final model and return the path it was saved to."""
    args = build_parser().parse_args(argv)

    os.makedirs(args.output_dir, exist_ok=True)
    best_params_path = args.best_params or default_best_params_path(args.target)
    model_path = os.path.join(
        args.output_dir,
        MODEL_FILENAME.format(kan_or_ann="ann", target=args.target),
    )

    set_seed(SEED, deterministic=True)

    print(f"Reading best params from: {best_params_path}")
    hparams = load_best_params(best_params_path)
    print(f"Hyperparameters         : {hparams}")
    print(f"Target variable         : {args.target}")
    print(f"Epochs                  : {args.epochs}\n")

    data = normalize_split_keys(
        load_training_data(
            os.path.join(DATA_DIR, args.data_path), 
            target=args.target, 
            seed=SEED
            )
    )
    print(
        f"Dataset sizes  ->  "
        f"train: {data['x_train'].shape[0]}  |  "
        f"val: {data['x_test'].shape[0]}\n"
    )

    model = train_final_model(
        data, hparams, epochs=args.epochs, use_gpu=not args.no_gpu
    )

    final_val_loss = save_model(model, model_path, args.target, hparams, args.epochs)

    print("\n" + "=" * 70)
    print("Training complete.")
    if final_val_loss is not None:
        print(f"  Final validation MSE : {final_val_loss:.6f}")
    print(f"  Model saved to       : {model_path}")
    print("=" * 70)

    return model_path


if __name__ == "__main__":
    main()