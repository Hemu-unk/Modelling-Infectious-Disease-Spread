from __future__ import annotations

from pathlib import Path

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

from ml_utils import build_feature_row, load_models
from simulator import SimulationConfig, run_simulation

app = Flask(__name__)
CORS(app)

MODEL_DIR = Path(__file__).parent / "outputs" / "models"
FORM_FIELDS = [
    "population_size",
    "infection_probability",
    "recovery_time",
    "movement_speed",
    "interaction_radius",
    "vaccination_rate",
    "initial_infected",
]
DEFAULT_FORM_VALUES = {
    "population_size": "400",
    "infection_probability": "0.35",
    "recovery_time": "14",
    "movement_speed": "2.2",
    "interaction_radius": "2.8",
    "vaccination_rate": "0.15",
    "initial_infected": "10",
}


def _get_int(value, default: int) -> int:
    if value is None:
        return default
    return int(value)


def _get_float(value, default: float) -> float:
    if value is None:
        return default
    return float(value)


def _coerce_payload(raw: dict[str, object]) -> dict[str, float | int]:
    return {
        "population_size": int(raw["population_size"]),
        "infection_probability": float(raw["infection_probability"]),
        "recovery_time": int(raw["recovery_time"]),
        "movement_speed": float(raw["movement_speed"]),
        "interaction_radius": float(raw["interaction_radius"]),
        "vaccination_rate": float(raw["vaccination_rate"]),
        "initial_infected": int(raw["initial_infected"]),
    }


def _validate_payload(payload: dict[str, float | int]) -> list[str]:
    errors: list[str] = []
    if payload["population_size"] <= 0:
        errors.append("Population size must be greater than 0.")
    if payload["initial_infected"] <= 0:
        errors.append("Initial infected must be greater than 0.")
    if payload["initial_infected"] >= payload["population_size"]:
        errors.append("Initial infected must be less than population size.")
    if not (0.0 <= payload["infection_probability"] <= 1.0):
        errors.append("Infection probability must be between 0 and 1.")
    if not (0.0 <= payload["vaccination_rate"] <= 1.0):
        errors.append("Vaccination rate must be between 0 and 1.")
    if payload["recovery_time"] <= 0:
        errors.append("Recovery time must be greater than 0.")
    if payload["movement_speed"] <= 0:
        errors.append("Movement speed must be greater than 0.")
    if payload["interaction_radius"] <= 0:
        errors.append("Interaction radius must be greater than 0.")
    return errors


def _predict_metrics(payload: dict[str, float | int]) -> dict[str, float]:
    models = load_models(MODEL_DIR)
    feature_row = build_feature_row(payload)
    return {
        "peak_infected_population": float(models["peak_infected_population"].predict(feature_row)[0]),
        "outbreak_duration": float(models["outbreak_duration"].predict(feature_row)[0]),
        "total_infected_population": float(models["total_infected_population"].predict(feature_row)[0]),
    }


@app.get("/health")
def health_check():
    return jsonify({"status": "ok"})


@app.get("/")
def web_predict_form():
    return render_template(
        "index.html",
        form_values=dict(DEFAULT_FORM_VALUES),
        errors=[],
        show_simulation=False,
    )


@app.post("/web-predict")
def web_predict_results():
    form_values = {field: request.form.get(field, "") for field in FORM_FIELDS}
    try:
        payload = _coerce_payload(form_values)
        errors = _validate_payload(payload)
        if errors:
            return render_template(
                "index.html",
                form_values=form_values,
                errors=errors,
                show_simulation=False,
            ), 400

        return render_template(
            "index.html",
            form_values=form_values,
            errors=[],
            show_simulation=True,
        )
    except (TypeError, ValueError):
        return (
            render_template(
                "index.html",
                form_values=form_values,
                errors=["Enter valid numeric values for all fields."],
                show_simulation=False,
            ),
            400,
        )
    except FileNotFoundError as exc:
        return (
            render_template(
                "index.html",
                form_values=form_values,
                errors=[f"{str(exc)}. Train models first using train_models.py."],
                show_simulation=False,
            ),
            400,
        )


@app.post("/simulate")
def simulate():
    payload = request.get_json(silent=True) or {}
    include_agent_snapshots = bool(payload.get("include_agent_snapshots", False))

    config = SimulationConfig(
        population_size=_get_int(payload.get("population_size"), 300),
        initial_infected=_get_int(payload.get("initial_infected"), 3),
        vaccinated_fraction=_get_float(payload.get("vaccinated_fraction"), 0.0),
        width=_get_float(payload.get("width"), 100.0),
        height=_get_float(payload.get("height"), 100.0),
        movement_step=_get_float(payload.get("movement_step"), 1.5),
        infection_radius=_get_float(payload.get("infection_radius"), 2.0),
        infection_probability=_get_float(payload.get("infection_probability"), 0.25),
        recovery_time=_get_int(payload.get("recovery_time"), 20),
        max_steps=_get_int(payload.get("max_steps"), 300),
        seed=_get_int(payload.get("seed"), 42),
    )

    try:
        result = run_simulation(config, include_agent_snapshots=include_agent_snapshots)
        return jsonify(result)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


@app.post("/predict")
def predict_outcomes():
    payload = request.get_json(silent=True) or {}

    missing_fields = [field for field in FORM_FIELDS if field not in payload]
    if missing_fields:
        return jsonify({"error": f"Missing required fields: {missing_fields}"}), 400

    try:
        typed_payload = _coerce_payload(payload)
        validation_errors = _validate_payload(typed_payload)
        if validation_errors:
            return jsonify({"error": validation_errors}), 400

        predictions = _predict_metrics(typed_payload)
        return jsonify({"predictions": predictions})
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc), "hint": "Train models first using train_models.py"}), 400
    except (TypeError, ValueError) as exc:
        return jsonify({"error": f"Invalid input: {str(exc)}"}), 400


if __name__ == "__main__":
    app.run(debug=True)
