from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeRegressor

from ml_utils import FEATURE_COLUMNS

TARGET_COLUMNS = [
    "peak_infected_population",
    "outbreak_duration",
    "total_infected_population",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train outbreak metric regressors")
    parser.add_argument("--data-path", default="outputs/simulation_training_data.csv", help="Input CSV from dataset generator")
    parser.add_argument("--model-dir", default="outputs/models", help="Directory to write model files")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/test split and models")
    parser.add_argument("--cv-folds", type=int, default=5, help="Cross-validation folds")
    return parser.parse_args()


def _candidate_models(seed: int) -> dict[str, object]:
    return {
        "linear_regression": LinearRegression(),
        "decision_tree": DecisionTreeRegressor(random_state=seed),
        "random_forest": RandomForestRegressor(
            n_estimators=400,
            random_state=seed,
            n_jobs=-1,
            min_samples_leaf=1,
        ),
    }


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(args.data_path)

    missing = [col for col in FEATURE_COLUMNS + TARGET_COLUMNS if col not in data.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    X = data[FEATURE_COLUMNS]

    metrics_rows: list[dict[str, float | str]] = []
    best_models: dict[str, object] = {}
    best_algorithms: dict[str, str] = {}

    for target in TARGET_COLUMNS:
        y = data[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=args.test_size,
            random_state=args.seed,
        )

        target_results: list[dict[str, float | str]] = []
        for name, model in _candidate_models(args.seed).items():
            model.fit(X_train, y_train)
            test_predictions = model.predict(X_test)

            mse = mean_squared_error(y_test, test_predictions)
            r2 = r2_score(y_test, test_predictions)
            cv_r2 = cross_val_score(model, X, y, cv=args.cv_folds, scoring="r2").mean()

            result_row = {
                "target": target,
                "model": name,
                "mse": float(mse),
                "r2": float(r2),
                "cv_r2": float(cv_r2),
            }
            target_results.append(result_row)
            metrics_rows.append(result_row)

        best_result = max(target_results, key=lambda row: (row["r2"], row["cv_r2"]))
        best_algorithm = str(best_result["model"])
        best_algorithms[target] = best_algorithm

        best_model = _candidate_models(args.seed)[best_algorithm]
        best_model.fit(X, y)
        best_models[target] = best_model

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = model_dir / "model_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    joblib.dump(best_models["peak_infected_population"], model_dir / "peak_model.pkl")
    joblib.dump(best_models["outbreak_duration"], model_dir / "duration_model.pkl")
    joblib.dump(best_models["total_infected_population"], model_dir / "total_model.pkl")

    metadata = {
        "feature_columns": FEATURE_COLUMNS,
        "target_columns": TARGET_COLUMNS,
        "best_algorithms": best_algorithms,
    }
    metadata_path = model_dir / "model_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Training complete.")
    print(f"Saved metrics: {metrics_path.as_posix()}")
    print(f"Saved metadata: {metadata_path.as_posix()}")
    print("Saved models:")
    print(f"  {(model_dir / 'peak_model.pkl').as_posix()}")
    print(f"  {(model_dir / 'duration_model.pkl').as_posix()}")
    print(f"  {(model_dir / 'total_model.pkl').as_posix()}")

    print("\nBest algorithm per target:")
    for target_name, algorithm in best_algorithms.items():
        print(f"  {target_name}: {algorithm}")


if __name__ == "__main__":
    main()
