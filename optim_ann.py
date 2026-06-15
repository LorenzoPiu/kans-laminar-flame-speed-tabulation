#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
optimization_ANN.py
===================
Hyperparameter Optimisation (HPO) for the FCNN architecture using Optuna.

What it does
------------
- Loads a laminar-flame dataset (CSV) via ``load_training_data`` from utils.py.
- Runs a TPE-sampler Optuna study to minimise the mean validation MSE loss
  computed over the **last 1/20th of the training epochs** (e.g. the last 500
  epochs out of 10 000).
- Saves three artefacts in --output_dir:
    * ``best_params.json``  - best hyperparameters + corresponding val loss
    * ``hpo_results.csv``   - full trial history (one row per trial)
    * ``hpo_log.txt``       - everything printed during the run

Usage
-----
    python optimization_ANN.py \\
        --data_path  /path/to/dataset.csv \\
        --target     flame_speed \\
        --epochs     10000 \\
        --trials     300

Hyperparameters tuned
---------------------
    n_layers     : number of hidden layers
    n_neurons    : neurons per hidden layer (same for all layers)
    batch_size   : mini-batch size (log-uniform over integers)
    lr           : learning rate  (log-uniform)
    weight_decay : L1 regularisation coefficient (log-uniform)

Notes
-----
- Input  dimensionality : 6  (Pressure, Temperature, H2, CO, CO2, H2O mass fractions)
- Output dimensionality : 1  (flame speed *or* density ratio, selected via --target)
- Optimiser             : Adam (fixed)
- Seed                  : 42  (set in main() via set_seed)
- The module is import-safe: nothing runs until ``main()`` is called, so it
  can be unit-tested with pytest (see test_optimization_ann.py).
"""

import argparse
import contextlib
import csv
import json
import os

import numpy as np
import optuna
from optuna.samplers import TPESampler

# ---------------------------------------------------------------------------
# Local imports  (adjust relative paths if needed)
# ---------------------------------------------------------------------------
from architectures import FCNN
from utils import load_training_data, set_seed
from global_vars import (
    DATA_DIR, 
    HPO_OUTPUT_DIR, 
    HPO_CSV_FILENAME, 
    HPO_BEST_PARAMS_FILENAME, 
    HPO_LOG_FILENAME
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED = 42

INPUT_DIM = 6   # Pressure, Temperature, H2, CO, CO2, H2O mass fractions
OUTPUT_DIM = 1  # flame_speed or density_ratio (single scalar target)

# Search-space bounds (tweak here without touching the objective)
N_LAYERS_LOW,   N_LAYERS_HIGH   = 2,    16
N_NEURONS_LOW,  N_NEURONS_HIGH  = 8,    256
BATCH_SIZE_LOW, BATCH_SIZE_HIGH = 512,  4096   # sampled in log space
LR_LOW,         LR_HIGH         = 1e-5, 1e-1
WD_LOW,         WD_HIGH         = 1e-8, 1e-0

CSV_FIELDNAMES = ["trial", "n_layers", "n_neurons", "batch_size",
                  "lr", "weight_decay", "val_loss"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def compute_tail(epochs: int) -> int:
    """Length of the validation-loss window: last 1/20th of the epochs,
    but never less than one epoch."""
    return max(1, epochs // 20)


def normalize_split_keys(data: dict) -> dict:
    """Map the data dict to the keys FCNN.fit() expects.

    ``load_training_data`` returns 'x_val'/'y_val', but FCNN.fit() looks for
    'x_test'/'y_test' to populate its validation history. Without this
    mapping, model.test_mse_loss stays empty and the objective is NaN.
    Accepts either naming convention.
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Optuna HPO for the FCNN architecture."
    )
    parser.add_argument(
        "--data_path", "-d",
        type=str,
        required=True,
        help="Path to the laminar-flame dataset CSV file. Assumes the data directory is DATA_DIR",
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
        help="Number of training epochs per trial. Default: 10000.",
    )
    parser.add_argument(
        "--trials", "-t",
        type=int,
        default=300,
        help="Number of Optuna trials. Default: 300.",
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default=HPO_OUTPUT_DIR,
        help="Directory where best_params.json, hpo_results.csv and "
             "hpo_log.txt are saved. Default: 'hpo_output'.",
    )
    return parser


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------
def make_objective(data, epochs, n_tail, trial_records, use_gpu=True):
    """Build the Optuna objective.

    ``data`` must already use 'x_test'/'y_test' keys (see normalize_split_keys).
    Completed trials are appended to ``trial_records`` for the CSV export.
    """

    def objective(trial: optuna.Trial) -> float:
        """Return the mean validation MSE over the last n_tail epochs."""

        # --- suggest hyperparameters ------------------------------------
        n_layers = trial.suggest_int("n_layers", N_LAYERS_LOW, N_LAYERS_HIGH)
        n_neurons = trial.suggest_int("n_neurons", N_NEURONS_LOW, N_NEURONS_HIGH)
        batch_size = trial.suggest_int(
            "batch_size", BATCH_SIZE_LOW, BATCH_SIZE_HIGH, log=True
        )
        lr = trial.suggest_float("lr", LR_LOW, LR_HIGH, log=True)
        wd = trial.suggest_float("weight_decay", WD_LOW, WD_HIGH, log=True)

        # --- build & train model -----------------------------------------
        model = FCNN(
            input_dim=INPUT_DIM,
            n_layers=n_layers,
            n_neurons=n_neurons,
            output_dim=OUTPUT_DIM,
        )
        model.fit(
            data=data,
            epochs=epochs,
            weight_decay=wd,
            learning_rate=lr,
            batch_size=batch_size,
            verbose=True,
            use_gpu=use_gpu,
        )

        # --- validation metric --------------------------------------------
        if not model.test_mse_loss:
            raise RuntimeError(
                "FCNN.fit() recorded no validation losses. Check that the "
                "data dict contains 'x_test'/'y_test' keys "
                "(see normalize_split_keys)."
            )
        val_loss = float(np.mean(model.test_mse_loss[-n_tail:]))
        if not np.isfinite(val_loss):
            # Diverged trial (NaN/inf loss): give it the worst possible
            # score instead of poisoning the study with NaNs.
            val_loss = float("inf")

        # --- record for CSV ------------------------------------------------
        trial_records.append(
            {
                "trial":        trial.number,
                "n_layers":     n_layers,
                "n_neurons":    n_neurons,
                "batch_size":   batch_size,
                "lr":           lr,
                "weight_decay": wd,
                "val_loss":     val_loss,
            }
        )
        return val_loss

    return objective


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save_results(study, trial_records, best_params_path, csv_path):
    best_params_out = {**study.best_trial.params, "val_loss": study.best_value}
    with open(best_params_path, "w") as fp:
        json.dump(best_params_out, fp, indent=4)
    print(f"Best parameters saved  -> {best_params_path}")

    with open(csv_path, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(trial_records)
    print(f"Full HPO history saved -> {csv_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv=None) -> optuna.Study:
    args = build_parser().parse_args(argv)

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, HPO_CSV_FILENAME.format(kan_or_ann="ann", target=args.target))
    best_params_path = os.path.join(args.output_dir, HPO_BEST_PARAMS_FILENAME.format(kan_or_ann="ann", target=args.target))
    log_path = os.path.join(args.output_dir, HPO_LOG_FILENAME.format(kan_or_ann="ann", target=args.target))

    n_tail = compute_tail(args.epochs)

    set_seed(SEED, deterministic=True)

    # redirect_stdout restores sys.stdout even if the study crashes
    with open(log_path, "w") as log_file, contextlib.redirect_stdout(log_file):
        print(f"Loading dataset from: {os.path.join(DATA_DIR, args.data_path)}")
        print(f"Target variable     : {args.target}")
        print(f"Epochs per trial    : {args.epochs}")
        print(f"Optuna trials       : {args.trials}")
        print(f"Val-loss tail length: {n_tail} epochs\n")

        data = normalize_split_keys(
            load_training_data(
                os.path.join(DATA_DIR, args.data_path), 
                target=args.target, 
                seed=SEED)
        )
        print(
            f"Dataset sizes  ->  "
            f"train: {data['x_train'].shape[0]}  |  "
            f"val: {data['x_test'].shape[0]}"
        )

        trial_records = []
        sampler = TPESampler(seed=SEED)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(
            make_objective(data, args.epochs, n_tail, trial_records),
            n_trials=args.trials,
            show_progress_bar=True,  # the bar goes to stderr, not the log
        )

        print("\n" + "=" * 70)
        print("HPO complete.")
        print(f"  Best validation MSE  : {study.best_value:.6f}")
        print(f"  Best hyperparameters : {study.best_trial.params}")
        print(f"  Val-loss tail length : {n_tail} epochs")
        print("=" * 70 + "\n")

        save_results(study, trial_records, best_params_path, csv_path)

    return study


if __name__ == "__main__":
    main()