from __future__ import annotations

import argparse

from ml_utils import build_feature_row, load_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict outbreak metrics from simulation parameters")
    parser.add_argument("--model-dir", default="outputs/models", help="Directory containing trained model files")
    parser.add_argument("--population-size", type=int, required=True)
    parser.add_argument("--infection-probability", type=float, required=True)
    parser.add_argument("--recovery-time", type=int, required=True)
    parser.add_argument("--movement-speed", type=float, required=True)
    parser.add_argument("--interaction-radius", type=float, required=True)
    parser.add_argument("--vaccination-rate", type=float, required=True)
    parser.add_argument("--initial-infected", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = {
        "population_size": args.population_size,
        "infection_probability": args.infection_probability,
        "recovery_time": args.recovery_time,
        "movement_speed": args.movement_speed,
        "interaction_radius": args.interaction_radius,
        "vaccination_rate": args.vaccination_rate,
        "initial_infected": args.initial_infected,
    }

    features = build_feature_row(payload)
    models = load_models(args.model_dir)

    peak = float(models["peak_infected_population"].predict(features)[0])
    duration = float(models["outbreak_duration"].predict(features)[0])
    total = float(models["total_infected_population"].predict(features)[0])

    print("Predicted outbreak metrics:")
    print(f"  peak_infected_population: {peak:.3f}")
    print(f"  outbreak_duration: {duration:.3f}")
    print(f"  total_infected_population: {total:.3f}")


if __name__ == "__main__":
    main()
