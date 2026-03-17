from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from ml_utils import FEATURE_COLUMNS, build_feature_row, load_models
from simulator import SimulationConfig, run_simulation


TARGET_COLUMNS = [
    "peak_infected_population",
    "outbreak_duration",
    "total_infected_population",
]

MODEL_DIR = Path(__file__).parent / "outputs" / "models"
DATA_PATH = Path(__file__).parent / "outputs" / "simulation_training_data.csv"

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"


def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print("=" * 60)


def check(label: str, passed: bool, detail: str = "") -> None:
    status = PASS if passed else FAIL
    line = f"  [{status}] {label}"
    if detail:
        line += f"  ({detail})"
    print(line)
    return passed


# ── 1. Held-out accuracy ──────────────────────────────────────────
def check_accuracy(models: dict, data: pd.DataFrame) -> None:
    section("1. Held-out test accuracy")

    X = data[FEATURE_COLUMNS]
    all_passed = True

    for target in TARGET_COLUMNS:
        y = data[target]
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        preds = models[target].predict(X_test)

        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        rmse = mean_squared_error(y_test, preds) ** 0.5

        passed = r2 > 0.40
        check(
            f"{target}",
            passed,
            f"R²={r2:.3f}  MAE={mae:.1f}  RMSE={rmse:.1f}",
        )
        all_passed = all_passed and passed

    if not all_passed:
        print("  → Consider generating more data with generate_dataset.py and retraining.")


# ── 2. Behavioral monotonicity ────────────────────────────────────
def check_behavior(models: dict) -> None:
    section("2. Behavioral sanity checks")

    base = {
        "population_size": 400,
        "infection_probability": 0.30,
        "recovery_time": 14,
        "movement_speed": 2.0,
        "interaction_radius": 2.5,
        "vaccination_rate": 0.10,
        "initial_infected": 8,
    }

    def predict(params: dict) -> dict[str, float]:
        row = build_feature_row(params)
        return {
            target: float(models[target].predict(row)[0])
            for target in TARGET_COLUMNS
        }

    base_pred = predict(base)

    # High vs low infection probability
    high_p = predict({**base, "infection_probability": 0.55})
    low_p = predict({**base, "infection_probability": 0.05})
    check(
        "Higher infection probability → higher peak infected",
        high_p["peak_infected_population"] > low_p["peak_infected_population"],
        f"high={high_p['peak_infected_population']:.1f} > low={low_p['peak_infected_population']:.1f}",
    )
    check(
        "Higher infection probability → more total infected",
        high_p["total_infected_population"] > low_p["total_infected_population"],
        f"high={high_p['total_infected_population']:.1f} > low={low_p['total_infected_population']:.1f}",
    )

    # High vs low vaccination rate
    high_vax = predict({**base, "vaccination_rate": 0.55})
    low_vax = predict({**base, "vaccination_rate": 0.0})
    check(
        "Higher vaccination → lower peak infected",
        high_vax["peak_infected_population"] < low_vax["peak_infected_population"],
        f"high_vax={high_vax['peak_infected_population']:.1f} < low_vax={low_vax['peak_infected_population']:.1f}",
    )
    check(
        "Higher vaccination → fewer total infected",
        high_vax["total_infected_population"] < low_vax["total_infected_population"],
        f"high_vax={high_vax['total_infected_population']:.1f} < low_vax={low_vax['total_infected_population']:.1f}",
    )

    # Long vs short recovery
    long_rec = predict({**base, "recovery_time": 28})
    short_rec = predict({**base, "recovery_time": 5})
    check(
        "Longer recovery time → longer outbreak duration",
        long_rec["outbreak_duration"] > short_rec["outbreak_duration"],
        f"long={long_rec['outbreak_duration']:.1f} > short={short_rec['outbreak_duration']:.1f}",
    )

    # More initial infected → larger outbreak
    more_init = predict({**base, "initial_infected": 25})
    few_init = predict({**base, "initial_infected": 2})
    check(
        "More initial infected → higher peak",
        more_init["peak_infected_population"] >= few_init["peak_infected_population"],
        f"more={more_init['peak_infected_population']:.1f} >= few={few_init['peak_infected_population']:.1f}",
    )


# ── 3. ML prediction vs actual simulation ─────────────────────────
def check_prediction_vs_simulation(models: dict) -> None:
    section("3. ML predictions vs actual simulation outcomes")

    test_cases = [
        {"population_size": 300, "infection_probability": 0.25, "recovery_time": 14,
         "movement_speed": 1.5, "interaction_radius": 2.0, "vaccination_rate": 0.0,
         "initial_infected": 5, "seed": 7},
        {"population_size": 500, "infection_probability": 0.45, "recovery_time": 10,
         "movement_speed": 2.5, "interaction_radius": 3.0, "vaccination_rate": 0.2,
         "initial_infected": 12, "seed": 19},
        {"population_size": 200, "infection_probability": 0.10, "recovery_time": 20,
         "movement_speed": 1.0, "interaction_radius": 1.5, "vaccination_rate": 0.3,
         "initial_infected": 3, "seed": 55},
    ]

    print(f"\n  {'Metric':<30} {'Predicted':>12} {'Actual':>12} {'Abs Error':>12}")
    print("  " + "-" * 70)

    total_error_pct = 0.0
    count = 0

    for case in test_cases:
        seed = case.pop("seed")
        params = {**case}

        config = SimulationConfig(
            population_size=params["population_size"],
            initial_infected=params["initial_infected"],
            vaccinated_fraction=params["vaccination_rate"],
            movement_step=params["movement_speed"],
            infection_radius=params["interaction_radius"],
            infection_probability=params["infection_probability"],
            recovery_time=params["recovery_time"],
            max_steps=300,
            seed=seed,
        )
        sim_result = run_simulation(config)["summary"]
        case["seed"] = seed

        row = build_feature_row(params)
        for target in TARGET_COLUMNS:
            pred = float(models[target].predict(row)[0])
            actual = float(sim_result[target])
            err = abs(pred - actual)
            err_pct = (err / max(actual, 1)) * 100
            total_error_pct += err_pct
            count += 1
            label = f"pop={params['population_size']} p={params['infection_probability']} → {target[:14]}"
            print(f"  {label:<30} {pred:>12.1f} {actual:>12.1f} {err:>11.1f}")
        print()

    avg_err_pct = total_error_pct / count
    passed = avg_err_pct < 60
    check(
        "Average relative error across all test cases",
        passed,
        f"{avg_err_pct:.1f}%  (acceptable for stochastic simulation)",
    )
    print()
    print("  Note: agent-based simulations are stochastic so some prediction error is expected.")
    print("  Models predict expected trends, not exact run outcomes.")


# ── Main ──────────────────────────────────────────────────────────
def main() -> None:
    print("\nModel Verification Report")
    print("=========================")

    if not DATA_PATH.exists():
        print(f"Dataset not found at {DATA_PATH}. Run generate_dataset.py first.")
        sys.exit(1)

    models = load_models(MODEL_DIR)
    data = pd.read_csv(DATA_PATH)
    print(f"Dataset loaded: {len(data)} rows")

    check_accuracy(models, data)
    check_behavior(models)
    check_prediction_vs_simulation(models)

    print(f"\n{'='*60}")
    print("  Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
