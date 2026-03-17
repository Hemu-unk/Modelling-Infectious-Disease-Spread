from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd

FEATURE_COLUMNS = [
    "population_size",
    "infection_probability",
    "recovery_time",
    "movement_speed",
    "interaction_radius",
    "vaccination_rate",
    "initial_infected",
]

MODEL_FILES = {
    "peak_infected_population": "peak_model.pkl",
    "outbreak_duration": "duration_model.pkl",
    "total_infected_population": "total_model.pkl",
}


def build_feature_row(payload: Dict[str, Any]) -> pd.DataFrame:
    row = {
        "population_size": int(payload["population_size"]),
        "infection_probability": float(payload["infection_probability"]),
        "recovery_time": int(payload["recovery_time"]),
        "movement_speed": float(payload["movement_speed"]),
        "interaction_radius": float(payload["interaction_radius"]),
        "vaccination_rate": float(payload["vaccination_rate"]),
        "initial_infected": int(payload["initial_infected"]),
    }
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def load_models(model_dir: str | Path) -> Dict[str, Any]:
    model_path = Path(model_dir)
    models = {}

    for target_name, file_name in MODEL_FILES.items():
        full_path = model_path / file_name
        if not full_path.exists():
            raise FileNotFoundError(f"Missing model file: {full_path.as_posix()}")
        models[target_name] = joblib.load(full_path)

    return models
