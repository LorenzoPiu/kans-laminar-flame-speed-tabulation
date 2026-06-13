"""
Self-contained generator for a fake laminar flame dataset.

Structure (same columns as the original CSV):
  Inputs  : Pressure [Pa], Temperature [K], 6 unburnt mass fractions (H2, CO, CO2, H2O, O2, N2)
  Outputs : 6 burnt mass fractions (set to 0, not of interest for now),
            Laminar Flame Speed S_L [m/s], Density Ratio (Burnt/Unburnt)

S_L and the density ratio are computed from two smooth analytical functions of the
8 inputs, so the data has a learnable structure (useful e.g. for testing surrogate
models / regressors).

Usage:
  python generate_fake_dataset.py -n 1000 -o fake_flame_dataset.csv
"""

import argparse
import numpy as np
import pandas as pd

# ----------------------------- configuration --------------------------------
SEED = 42
DATA_DIR = "data"

# Sampling ranges for the inputs
P_RANGE = (1.0e5, 4.0e5)      # Pa
T_RANGE = (300.0, 700.0)      # K
Y_H2_RANGE = (0.0, 0.10)      # fuel
Y_CO_RANGE = (0.0, 0.05)      # fuel
Y_CO2_RANGE = (0.0, 0.40)     # diluent
Y_H2O_RANGE = (0.0, 0.15)     # diluent

# The remainder of the mixture is "air", split into O2/N2 with the
# usual air mass fractions (this matches the original dataset, where
# Y_O2 / Y_N2 = 0.233 / 0.767 on every row).
AIR_YO2, AIR_YN2 = 0.233, 0.767

# Reference conditions for the analytical laws
P0, T0 = 1.0e5, 300.0

COLUMNS = [
    "Pressure [Pa]", "Temperature [K]",
    "Mass Fraction (H2) Unburnt", "Mass Fraction (CO) Unburnt",
    "Mass Fraction (CO2) Unburnt", "Mass Fraction (H2O) Unburnt",
    "Mass Fraction (O2) Unburnt", "Mass Fraction (N2) Unburnt",
    "Mass Fraction (H2) Burnt", "Mass Fraction (CO) Burnt",
    "Mass Fraction (CO2) Burnt", "Mass Fraction (H2O) Burnt",
    "Mass Fraction (O2) Burnt", "Mass Fraction (N2) Burnt",
    "Laminar Flame Speed S_L [m/s]", "Density Ratio (Burnt/Unburnt)",
]


# --------------------------- input sampling ---------------------------------
def sample_inputs(n, rng):
    """Random grid of P, T and unburnt composition (mass fractions sum to 1)."""
    P = rng.uniform(*P_RANGE, n)
    T = rng.uniform(*T_RANGE, n)

    y_h2 = rng.uniform(*Y_H2_RANGE, n)
    y_co = rng.uniform(*Y_CO_RANGE, n)
    y_co2 = rng.uniform(*Y_CO2_RANGE, n)
    y_h2o = rng.uniform(*Y_H2O_RANGE, n)

    # If fuel + diluents exceed ~0.85, rescale so some air always remains
    total = y_h2 + y_co + y_co2 + y_h2o
    scale = np.where(total > 0.85, 0.85 / total, 1.0)
    y_h2, y_co, y_co2, y_h2o = (y * scale for y in (y_h2, y_co, y_co2, y_h2o))

    rest = 1.0 - (y_h2 + y_co + y_co2 + y_h2o)
    y_o2 = AIR_YO2 * rest
    y_n2 = AIR_YN2 * rest
    return P, T, y_h2, y_co, y_co2, y_h2o, y_o2, y_n2


# ------------------------- analytical functions -----------------------------
def equivalence_ratio(y_h2, y_co, y_o2):
    """Mass-based equivalence ratio for an H2/CO fuel blend.

    Stoichiometric O2 demand: 8 kg O2 per kg H2, 4/7 kg O2 per kg CO.
    """
    o2_needed = 8.0 * y_h2 + (4.0 / 7.0) * y_co
    return o2_needed / np.maximum(y_o2, 1e-12)


def laminar_flame_speed(P, T, y_h2, y_co, y_co2, y_h2o, y_o2):
    """Analytical model:  S_L = S_ref(phi) * dilution * (T/T0)^a * (P/P0)^b."""
    phi = equivalence_ratio(y_h2, y_co, y_o2)

    # Bell-shaped dependence on equivalence ratio, peaking slightly rich,
    # with amplitude growing with the H2 share of the fuel.
    fuel = y_h2 + y_co
    h2_share = np.divide(y_h2, np.maximum(fuel, 1e-12))
    s_ref = (0.4 + 2.6 * h2_share) * np.exp(-((phi - 1.4) ** 2) / (2 * 0.55 ** 2))

    # Diluents slow the flame down (CO2 stronger than H2O)
    dilution = np.exp(-3.5 * y_co2 - 2.0 * y_h2o)

    s_l = s_ref * dilution * (T / T0) ** 1.8 * (P / P0) ** (-0.4)

    # Flammability limits: no fuel, no oxygen, or phi out of range -> no flame
    flammable = (fuel > 1e-4) & (y_o2 > 1e-4) & (phi > 0.3) & (phi < 5.0)
    return np.where(flammable, s_l, 0.0), flammable


def density_ratio(T, y_h2, y_co, y_co2, y_h2o, y_o2, flammable):
    """Analytical model:  sigma ~ T_ad / T_u, driven by released heat.

    Heat release scales with the limiting reactant (fuel- or O2-limited),
    is weighted by the fuels' heating values, and is damped by diluents.
    """
    phi = equivalence_ratio(y_h2, y_co, y_o2)
    burn_eff = np.minimum(1.0, 1.0 / np.maximum(phi, 1e-12))  # lean: 1, rich: 1/phi

    q = (120.0 * y_h2 + 10.1 * y_co) * burn_eff          # MJ/kg-mixture proxy
    q *= np.exp(-1.2 * y_co2 - 0.8 * y_h2o)              # thermal ballast of diluents

    sigma = 1.0 + 1500.0 * q / T                          # ~ (T_u + dT_ad) / T_u
    sigma = np.clip(sigma, 1.0, 8.0)

    # Match original convention: non-flammable rows get 0 everywhere
    return np.where(flammable, sigma, 0.0)


# ------------------------------- main ---------------------------------------
def generate(n, path):
    rng = np.random.default_rng(SEED)
    P, T, y_h2, y_co, y_co2, y_h2o, y_o2, y_n2 = sample_inputs(n, rng)

    s_l, flammable = laminar_flame_speed(P, T, y_h2, y_co, y_co2, y_h2o, y_o2)
    sigma = density_ratio(T, y_h2, y_co, y_co2, y_h2o, y_o2, flammable)

    zeros = np.zeros(n)  # burnt mass fractions: not of interest -> 0
    data = np.column_stack([
        P, T, y_h2, y_co, y_co2, y_h2o, y_o2, y_n2,
        zeros, zeros, zeros, zeros, zeros, zeros,
        s_l, sigma,
    ])

    df = pd.DataFrame(data, columns=COLUMNS)
    os.makedirs(DATA_DIR, exists_ok=True)
    df.to_csv(os.path.join(DATA_DIR, path), index=False)
    print(f"Wrote {n} samples to '{path}'")
    print(f"  flammable rows : {int(flammable.sum())}/{n}")
    print(f"  S_L    range   : [{s_l.min():.4f}, {s_l.max():.4f}] m/s")
    print(f"  sigma  range   : [{sigma[flammable].min():.3f}, "
          f"{sigma[flammable].max():.3f}] (flammable rows)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a fake laminar flame dataset (CSV)."
    )
    parser.add_argument(
        "-n", "--n-samples", type=int, default=1000,
        help="number of samples to generate (default: 1000)",
    )
    parser.add_argument(
        "-o", "--out", type=str, default="mock_dataset.csv",
        help="output CSV path (default: mock_dataset.csv)",
    )

    args = parser.parse_args()
    generate(args.n_samples, args.out)