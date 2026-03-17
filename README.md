# Modelling Infectious Disease Spread

## Phase 1 MVP (Simulation Core)

This workspace currently includes a Flask backend with an agent-based epidemic simulator.

### Implemented in Phase 1

- Agent data model with:
  - `id`
  - `x`, `y`
  - `state` (`S`, `I`, `R`)
  - `infection_timer`
  - `vaccinated`
- 2D environment and random movement
- Infection logic using proximity radius + infection probability
- Recovery logic with fixed recovery time
- Per-timestep data collection (`S`, `I`, `R`)
- Summary statistics:
  - peak infected population
  - outbreak duration
  - total infected population

## Phase 2 MVP (Visualization and Validation)

### Added in Phase 2

- Epidemic curve plotting with Matplotlib:
  - susceptible over time
  - infected over time
  - recovered over time
- Spatial state scatter plots with color mapping:
  - blue = susceptible
  - red = infected
  - green = recovered
- Sanity checks across multiple runs to validate expected model behavior:
  - higher infection probability -> faster/larger spread
  - higher vaccination -> smaller outbreaks
  - longer recovery -> longer outbreaks
  - total population remains constant

### Backend Structure

- `backend/app.py` Flask app
- `backend/simulator/simulation.py` simulation core
- `backend/run_demo.py` quick local simulation runner
- `backend/visualize_simulation.py` plot generator
- `backend/sanity_checks.py` validation script

### Run locally

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python run_demo.py
python app.py
```

Then test API:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:5000/simulate -ContentType 'application/json' -Body '{"population_size": 200, "initial_infected": 5}'
```

Generate plots:

```powershell
python visualize_simulation.py
```

Run sanity checks:

```powershell
python sanity_checks.py
```

## Phase 3 MVP (Dataset Generation)

### Added in Phase 3

- Automated experiment runner that executes many simulations and saves ML-ready datasets.
- Parameter sweeps over:
  - population size
  - infection probability
  - recovery time
  - movement speed
  - interaction radius
  - vaccination rate
  - initial infected
- Supports both random parameter sampling and grid search.
- Exports summary metrics to CSV using pandas.
- Optional export of raw per-timestep curves for deeper analysis.

### New Script

- `backend/generate_dataset.py`

### Generate a few hundred runs (recommended start)

```powershell
cd backend
python generate_dataset.py --mode random --num-sets 300 --repeats 1 --max-steps 250
```

### Generate grid-search data

```powershell
python generate_dataset.py --mode grid --repeats 1 --max-steps 250
```

### Also export timestep curves

```powershell
python generate_dataset.py --mode random --num-sets 200 --include-timesteps
```

### Output files

- `backend/outputs/simulation_training_data.csv`
- `backend/outputs/simulation_timestep_curves.csv` (only when `--include-timesteps` is enabled)

## Phase 4 MVP (Machine Learning Training)

### Added in Phase 4

- Model training pipeline that uses simulation data to predict outbreak outcomes.
- Features (X):
  - population size
  - infection probability
  - recovery time
  - movement speed
  - interaction radius
  - vaccination rate
  - initial infected
- Targets (y):
  - peak infected population
  - outbreak duration
  - total infected population
- Baseline regressors:
  - Linear Regression
  - Decision Tree Regressor
  - Random Forest Regressor
- Evaluation metrics:
  - train/test split
  - Mean Squared Error (MSE)
  - R2 score
  - cross-validation R2
- Saves best model per target with joblib:
  - `peak_model.pkl`
  - `duration_model.pkl`
  - `total_model.pkl`

### New Scripts

- `backend/train_models.py`
- `backend/predict_outcomes.py`
- `backend/ml_utils.py`

### Train models

```powershell
cd backend
python train_models.py --data-path outputs/simulation_training_data.csv --model-dir outputs/models
```

### Make instant predictions from a parameter set (CLI)

```powershell
python predict_outcomes.py --population-size 400 --infection-probability 0.35 --recovery-time 14 --movement-speed 2.2 --interaction-radius 2.8 --vaccination-rate 0.15 --initial-infected 10
```

## Phase 5 MVP (Flask Web App)

### Added in Phase 5

- Browser form for all model inputs:
  - population size
  - infection probability
  - recovery time
  - movement speed
  - interaction radius
  - vaccination rate
  - initial infected
- Backend web prediction route that:
  - reads submitted form values
  - validates and formats inputs
  - loads trained models
  - computes predictions
- Dedicated results page displaying:
  - predicted peak infected population
  - predicted outbreak duration
  - predicted total infected population
- Includes basic validation and improved styling.

### Run web app

```powershell
cd backend
python app.py
```

Open in browser:

- `http://127.0.0.1:5000/`

### Predict via API

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:5000/predict -ContentType 'application/json' -Body '{"population_size":400,"infection_probability":0.35,"recovery_time":14,"movement_speed":2.2,"interaction_radius":2.8,"vaccination_rate":0.15,"initial_infected":10}'
```
