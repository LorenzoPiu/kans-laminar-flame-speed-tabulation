#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
optimization_KAN.py
===================
Hyperparameter Optimisation (HPO) for the **pykan KAN** (MultKAN) architecture
using Optuna.

This is the KAN counterpart of ``optimization_ANN.py`` / ``optimization_MLP.py``.
It targets the pykan ``KAN`` class (``KAN is MultKAN``), imported as
``from kan import KAN``. Unlike the plain MLP, the KAN's capacity lives in the
learnable activation splines, so the **spline grid and order are tuned**:

    * grid : number of spline grid intervals  ("spline nodes")  -> finer = more params
    * k    : spline (B-spline) order            (k=3 is the usual cubic default)

What it does
------------
- Loads a laminar-flame dataset (CSV) via ``load_training_data`` from utils.py.
- Runs a TPE-sampler Optuna study to minimise the mean validation MSE loss
  computed over the **last 1/20th of the training steps**.
- Saves three artefacts in --output_dir:
    * ``best_params.json``  - best hyperparameters (+ resolved width) + val loss
    * ``hpo_results.csv``   - full trial history (one row per trial)
    * ``hpo_log.txt``       - everything printed during the run

Architecture search space (funnel, same rule as the ANN/MLP studies)
--------------------------------------------------------------------
    * n_layers          in [1, MAX_HIDDEN_LAYERS]
    * first hidden layer in [MIN_NEURONS_PER_LAYER, MAX_NEURONS_PER_LAYER]
    * every later layer  is strictly smaller than the previous one
      (STRICTLY_DECREASING = False -> non-increasing, i.e. "<=").
The full width passed to KAN is ``[INPUT_DIM, h_1, ..., h_L, OUTPUT_DIM]``.
KANs are expressive per edge, so keep MAX_NEURONS_PER_LAYER small.

Spline + training search space
------------------------------
    * grid in [GRID_LOW, GRID_HIGH]   (integer)
    * k    in [K_LOW, K_HIGH]         (integer; set K_LOW==K_HIGH to fix it)
    * lr   in [LR_LOW, LR_HIGH]       (log-uniform; ranges tuned for LBFGS)
    * lamb in [LAMB_LOW, LAMB_HIGH]   (log-uniform; overall reg strength)

Key KAN-specific choices (see notes at the bottom; all flippable up top)
------------------------------------------------------------------------
- ``opt`` defaults to "LBFGS" (the pykan workhorse). ``steps`` are LBFGS steps,
  far fewer than the ANN's epochs (default 100).
- KAN's ``fit`` records **RMSE**; we square per step and average to report the
  mean **MSE** over the tail, matching the ANN/MLP objectives.
- Regularisation uses KAN's spline-edge metric ``reg_metric='edge_forward_spline_n'``
  with ``lamb_l1=1, lamb_entropy=2`` (pykan defaults); only ``lamb`` is tuned.
- A single fixed grid is used per trial (``grid`` is the searched value). This
  does NOT do progressive grid refinement (grid 3 -> 5 -> 10 ...); that would be
  a different training loop - see the notes.
- ``auto_save=False`` and a throwaway ``ckpt_path`` so trials don't litter disk.
- KAN's ``fit`` does NOT move data to the model device and does NOT cap the
  validation batch at the validation-set size, so the dataset is moved to the
  device once, and (if tuned) ``batch`` is clamped to ``min(n_train, n_val)``.
  By default ``batch=-1`` (full batch, standard for LBFGS).

Usage
-----
    python optimization_KAN.py \\
        --data_path  /path/to/dataset.csv \\
        --target     flame_speed \\
        --steps      100 \\
        --trials     200

Notes
-----
- Input  dimensionality : 6  (Pressure, Temperature, H2, CO, CO2, H2O mass fractions)
- Output dimensionality : 1  (flame speed *or* density ratio, selected via --target)
- Seed                  : 42
- Import-safe: nothing runs until ``main()`` is called.
"""

import argparse
import contextlib
import csv
import json
import os
import tempfile

import numpy as np
import optuna
import torch
from optuna.samplers import TPESampler

# ---------------------------------------------------------------------------
# Local imports  (adjust relative paths if needed)
# ---------------------------------------------------------------------------
from kan import KAN
from utils import load_training_data, set_seed
from global_vars import (
    DATA_DIR,
    HPO_OUTPUT_DIR,
    HPO_CSV_FILENAME,
    HPO_BEST_PARAMS_FILENAME,
    HPO_LOG_FILENAME,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED = 42

INPUT_DIM = 6    # Pressure, Temperature, H2, CO, CO2, H2O mass fractions
OUTPUT_DIM = 1   # flame_speed or density_ratio (single scalar target)

# Filename tag for the global_vars filename templates (placeholder "kan_or_ann").
MODEL_TAG = "kan"

# --- architecture (funnel) bounds -- set YOUR upper limits here ------------
MAX_HIDDEN_LAYERS     = 4     # upper limit on number of hidden layers
MAX_NEURONS_PER_LAYER = 10    # upper limit on neurons in any single layer (keep small for KANs)
MIN_NEURONS_PER_LAYER = 1     # lower limit on neurons in any single layer
# True  -> each hidden layer STRICTLY smaller than the previous (h_i <  h_{i-1})
# False -> each hidden layer no larger than the previous        (h_i <= h_{i-1})
STRICTLY_DECREASING   = True

# --- spline search bounds --------------------------------------------------
GRID_LOW, GRID_HIGH = 3, 20   # number of spline grid intervals ("nodes")
K_LOW,    K_HIGH    = 2, 4     # spline order (set K_LOW == K_HIGH to fix, e.g. 3, 3)

# --- training (fixed) ------------------------------------------------------
OPTIMIZER    = "LBFGS"                 # "LBFGS" (pykan default) or "Adam"
UPDATE_GRID  = True                    # adapt grid-node positions during early training
REG_METRIC   = "edge_forward_spline_n" # KAN spline-edge sparsity metric
LAMB_L1      = 1.0                     # pykan default
LAMB_ENTROPY = 2.0                     # pykan default
TUNE_BATCH   = False                   # False -> full batch (-1); True -> tune (clamped)

# --- search-space bounds (tweak here without touching the objective) -------
BATCH_SIZE_LOW, BATCH_SIZE_HIGH = 512,  4096   # only used if TUNE_BATCH; clamped to data size
LR_LOW,         LR_HIGH         = 1e-2, 1.0     # learning rate (log-uniform; LBFGS range)
LAMB_LOW,       LAMB_HIGH       = 1e-8, 1e-0    # regularisation strength (log-uniform)

CSV_FIELDNAMES = ["trial", "n_layers", "width", "grid", "k",
                  "batch", "lr", "lamb", "val_loss"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def compute_tail(steps: int) -> int:
    """Length of the validation-loss window: last 1/20th of the steps,
    but never less than one step."""
    return max(1, steps // 20)


def normalize_split_keys(data: dict) -> dict:
    """Map the data dict to the keys pykan KAN.fit() expects.

    Stock pykan KAN reads 'train_input'/'train_label' and
    'test_input'/'test_label'. ``load_training_data`` may return the validation
    split as ('x_val','y_val') or ('x_test','y_test'); both are accepted.

    NB: if you have ALSO modified MultKAN's key scheme (as you did the MLP),
    change the output keys here to match.
    """
    out = {"train_input": data["x_train"], "train_label": data["y_train"]}
    if "x_val" in data and "y_val" in data:
        out["test_input"], out["test_label"] = data["x_val"], data["y_val"]
    elif "x_test" in data and "y_test" in data:
        out["test_input"], out["test_label"] = data["x_test"], data["y_test"]
    else:
        raise KeyError(
            "data must contain a validation split: either "
            "('x_val', 'y_val') or ('x_test', 'y_test')"
        )
    return out


def to_device(data: dict, device: str) -> dict:
    """KAN.fit() does not move data, so place every tensor on the model device
    as float32."""
    return {
        k: (v.to(device).float() if torch.is_tensor(v) else v)
        for k, v in data.items()
    }


def suggest_hidden_widths(trial: optuna.Trial) -> list:
    """Suggest a funnel of hidden-layer widths (identical rule to the ANN/MLP
    studies). The per-layer lower bound is raised just enough to guarantee the
    remaining (strictly decreasing) layers can still fit, so no infeasible
    configuration is ever proposed."""
    if STRICTLY_DECREASING:
        span = MAX_NEURONS_PER_LAYER - MIN_NEURONS_PER_LAYER + 1
        max_layers = min(MAX_HIDDEN_LAYERS, span)
    else:
        max_layers = MAX_HIDDEN_LAYERS

    n_layers = trial.suggest_int("n_layers", 1, max_layers)

    hidden = []
    upper = MAX_NEURONS_PER_LAYER
    for i in range(n_layers):
        remaining = n_layers - i - 1
        if STRICTLY_DECREASING:
            low = MIN_NEURONS_PER_LAYER + remaining
        else:
            low = MIN_NEURONS_PER_LAYER
        high = upper
        low = min(low, high)
        n = trial.suggest_int(f"n_neurons_l{i}", low, high)
        hidden.append(n)
        upper = (n - 1) if STRICTLY_DECREASING else n
        upper = max(upper, MIN_NEURONS_PER_LAYER)
    return hidden


def width_from_params(params: dict) -> list:
    """Reconstruct the full KAN width list from a trial's params dict."""
    n_layers = params["n_layers"]
    hidden = [params[f"n_neurons_l{i}"] for i in range(n_layers)]
    return [INPUT_DIM, *hidden, OUTPUT_DIM]


def resolve_device(use_gpu: bool) -> str:
    if use_gpu and torch.cuda.is_available():
        return "cuda"
    if use_gpu:
        print("WARNING: --use_gpu was set but CUDA is unavailable; using CPU.")
    return "cpu"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Optuna HPO for the pykan KAN (MultKAN) architecture."
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
        "--steps", "-e",
        type=int,
        default=100,
        help="Number of optimiser (LBFGS) steps per trial. Default: 100.",
    )
    parser.add_argument(
        "--trials", "-t",
        type=int,
        default=200,
        help="Number of Optuna trials. Default: 200.",
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default=HPO_OUTPUT_DIR,
        help="Directory where best_params.json, hpo_results.csv and "
             "hpo_log.txt are saved. Default: 'hpo_output'.",
    )
    parser.add_argument(
        "--use_gpu",
        action="store_true",
        help="Train on CUDA if available (falls back to CPU otherwise).",
    )
    return parser


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------
def make_objective(data, steps, n_tail, trial_records, device="cpu", ckpt_path="./model"):
    """Build the Optuna objective.

    ``data`` must already use the KAN keys and live on ``device`` (see
    normalize_split_keys / to_device). Completed trials are appended to
    ``trial_records`` for the CSV export.
    """
    n_train = data["train_input"].shape[0]
    n_val = data["test_input"].shape[0]
    # KAN draws train AND validation mini-batches of size `batch` without
    # replacement, so batch must not exceed either split.
    batch_cap = max(1, min(n_train, n_val))

    def objective(trial: optuna.Trial) -> float:
        """Return the mean validation MSE over the last n_tail steps."""

        # --- suggest hyperparameters ------------------------------------
        hidden = suggest_hidden_widths(trial)
        width = [INPUT_DIM, *hidden, OUTPUT_DIM]
        # KAN.__init__ rewrites the width list in place to MultKAN's internal
        # [n_sum, n_mult] format, so snapshot the readable string now.
        width_str = "-".join(str(w) for w in width)

        grid = trial.suggest_int("grid", GRID_LOW, GRID_HIGH)
        k = trial.suggest_int("k", K_LOW, K_HIGH)

        if TUNE_BATCH:
            batch_high = min(BATCH_SIZE_HIGH, batch_cap)
            batch_low = min(BATCH_SIZE_LOW, batch_high)
            batch = trial.suggest_int("batch", batch_low, batch_high, log=True)
        else:
            batch = -1  # full batch (standard for LBFGS)

        lr = trial.suggest_float("lr", LR_LOW, LR_HIGH, log=True)
        lamb = trial.suggest_float("lamb", LAMB_LOW, LAMB_HIGH, log=True)

        # --- build & train model -----------------------------------------
        model = KAN(
            width=list(width),
            grid=grid,
            k=k,
            seed=SEED,
            device=device,
            auto_save=False,
            ckpt_path=ckpt_path,
            save_act=True,
        )
        results = model.fit(
            data,
            opt=OPTIMIZER,
            steps=steps,
            lr=lr,
            lamb=lamb,
            lamb_l1=LAMB_L1,
            lamb_entropy=LAMB_ENTROPY,
            reg_metric=REG_METRIC,
            update_grid=UPDATE_GRID,
            batch=batch,
        )

        # --- validation metric --------------------------------------------
        # KAN records RMSE per step; square to MSE, then average over the tail
        # so the objective matches the ANN/MLP studies (mean validation MSE).
        test_rmse = results.get("test_loss")
        if not test_rmse:
            raise RuntimeError(
                "KAN.fit() recorded no validation losses. Check that the data "
                "dict contains 'test_input'/'test_label' keys "
                "(see normalize_split_keys)."
            )
        tail_rmse = np.asarray(test_rmse[-n_tail:], dtype=float)
        val_loss = float(np.mean(tail_rmse ** 2))
        if not np.isfinite(val_loss):
            # Diverged trial (NaN/inf loss, e.g. a spline singularity): give it
            # the worst possible score instead of poisoning the study.
            val_loss = float("inf")

        # --- record for CSV ------------------------------------------------
        trial_records.append(
            {
                "trial":    trial.number,
                "n_layers": len(hidden),
                "width":    width_str,
                "grid":     grid,
                "k":        k,
                "batch":    batch,
                "lr":       lr,
                "lamb":     lamb,
                "val_loss": val_loss,
            }
        )
        return val_loss

    return objective


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save_results(study, trial_records, best_params_path, csv_path):
    best_params_out = {
        **study.best_trial.params,
        "width": "-".join(str(w) for w in width_from_params(study.best_trial.params)),
        "val_loss": study.best_value,
    }
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
    csv_path = os.path.join(
        args.output_dir,
        HPO_CSV_FILENAME.format(kan_or_ann=MODEL_TAG, target=args.target),
    )
    best_params_path = os.path.join(
        args.output_dir,
        HPO_BEST_PARAMS_FILENAME.format(kan_or_ann=MODEL_TAG, target=args.target),
    )
    log_path = os.path.join(
        args.output_dir,
        HPO_LOG_FILENAME.format(kan_or_ann=MODEL_TAG, target=args.target),
    )

    n_tail = compute_tail(args.steps)
    device = resolve_device(args.use_gpu)
    ckpt_dir = tempfile.mkdtemp(prefix="kan_hpo_")  # throwaway; auto_save is off

    set_seed(SEED, deterministic=True)

    # redirect_stdout restores sys.stdout even if the study crashes
    with open(log_path, "w") as log_file, contextlib.redirect_stdout(log_file):
        print(f"Loading dataset from: {os.path.join(DATA_DIR, args.data_path)}")
        print(f"Model               : pykan KAN (MultKAN)")
        print(f"Target variable     : {args.target}")
        print(f"Optimiser           : {OPTIMIZER}")
        print(f"Device              : {device}")
        print(f"Steps per trial     : {args.steps}")
        print(f"Optuna trials       : {args.trials}")
        print(f"Val-loss tail length: {n_tail} steps")
        print(f"Grid search         : [{GRID_LOW}, {GRID_HIGH}]   k: [{K_LOW}, {K_HIGH}]")
        print(
            f"Funnel limits       : <= {MAX_HIDDEN_LAYERS} layers, "
            f"<= {MAX_NEURONS_PER_LAYER} neurons/layer, "
            f"{'strictly decreasing' if STRICTLY_DECREASING else 'non-increasing'}\n"
        )

        data = normalize_split_keys(
            load_training_data(
                os.path.join(DATA_DIR, args.data_path),
                target=args.target,
                seed=SEED,
            )
        )
        data = to_device(data, device)
        print(
            f"Dataset sizes  ->  "
            f"train: {data['train_input'].shape[0]}  |  "
            f"val: {data['test_input'].shape[0]}"
        )

        trial_records = []
        sampler = TPESampler(seed=SEED)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(
            make_objective(data, args.steps, n_tail, trial_records,
                           device=device, ckpt_path=ckpt_dir),
            n_trials=args.trials,
            show_progress_bar=True,  # the bar goes to stderr, not the log
        )

        print("\n" + "=" * 70)
        print("HPO complete.")
        print(f"  Best validation MSE  : {study.best_value:.6f}")
        print(f"  Best hyperparameters : {study.best_trial.params}")
        print(f"  Best width           : {width_from_params(study.best_trial.params)}")
        print(f"  Val-loss tail length : {n_tail} steps")
        print("=" * 70 + "\n")

        save_results(study, trial_records, best_params_path, csv_path)

    return study


if __name__ == "__main__":
    main()