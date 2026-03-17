# Modelling Infectious Disease Spread

An **agent-based epidemic simulation** with a machine learning pipeline and a Flask web application for predicting outbreak outcomes based on epidemiological parameters.

---

## Overview

This project simulates how infectious diseases spread through a population using an **agent-based model (ABM)**. Each individual is represented as an agent that moves within a 2D environment and interacts with others.

The system can:

- Simulate epidemic spread dynamics
- Generate datasets from repeated simulations
- Train machine learning models to predict outbreak outcomes
- Provide predictions through a simple web interface

---

## Features

### Agent-Based Simulation

- 2D environment with moving agents
- Health states:
  - **Susceptible (S)**
  - **Infected (I)**
  - **Recovered (R)**
- Infection based on **interaction radius and infection probability**
- Fixed recovery time
- Optional vaccination flag
- Per-timestep tracking of:
  - susceptible population
  - infected population
  - recovered population

Summary statistics generated from simulations:

- Peak infected population
- Outbreak duration
- Total infected population

---

### Visualization

- Epidemic curve plots (S, I, R over time)
- Spatial scatter plots of agent states
- Sanity checks to validate model behavior

Expected behaviors verified:

- Higher infection probability → faster spread
- Higher vaccination rate → smaller outbreaks
- Longer recovery time → longer outbreaks
- Population count remains constant

---

### Dataset Generation

Automated experiment runner generates **machine learning training data** by executing many simulations.

Parameter sweeps include:

- Population size
- Infection probability
- Recovery time
- Movement speed
- Interaction radius
- Vaccination rate
- Initial infected population

Outputs:

- `simulation_training_data.csv`
- optional timestep curve dataset

---

### Machine Learning Models

Simulation data is used to train regression models that predict outbreak outcomes.

Input features:

- population size
- infection probability
- recovery time
- movement speed
- interaction radius
- vaccination rate
- initial infected

Targets:

- peak infected population
- outbreak duration
- total infected population

Models included:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

Evaluation metrics:

- Train/test split
- Mean Squared Error (MSE)
- R² score
- Cross-validation

Trained models are saved using `joblib`.

---

### Web Application (Flask)

A simple browser interface allows users to enter simulation parameters and receive predicted outbreak outcomes.

Predictions include:

- predicted peak infected population
- predicted outbreak duration
- predicted total infected population

The API can also be accessed programmatically.

---

## Project Structure

```
backend/
│
├── app.py
├── run_demo.py
├── visualize_simulation.py
├── sanity_checks.py
│
├── simulator/
│   └── simulation.py
│
├── generate_dataset.py
├── train_models.py
├── predict_outcomes.py
├── ml_utils.py
│
└── outputs/
```

---

## Setup

Create and activate a Python virtual environment.

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## Run Simulation

```
python run_demo.py
```

---

## Generate Dataset

Example: generate random parameter simulations.

```
python generate_dataset.py --mode random --num-sets 300 --repeats 1 --max-steps 250
```

Optional: export timestep curves.

```
python generate_dataset.py --mode random --num-sets 200 --include-timesteps
```

---

## Train Machine Learning Models

```
python train_models.py --data-path outputs/simulation_training_data.csv --model-dir outputs/models
```

---

## Make Predictions (CLI)

```
python predict_outcomes.py \
--population-size 400 \
--infection-probability 0.35 \
--recovery-time 14 \
--movement-speed 2.2 \
--interaction-radius 2.8 \
--vaccination-rate 0.15 \
--initial-infected 10
```

---

## Run Web Application

```
python app.py
```

Open in browser:

```
http://127.0.0.1:5000/
```

---

## API Example

```
POST /predict
```

Example request:

```json
{
  "population_size": 400,
  "infection_probability": 0.35,
  "recovery_time": 14,
  "movement_speed": 2.2,
  "interaction_radius": 2.8,
  "vaccination_rate": 0.15,
  "initial_infected": 10
}
```

---

## Future Work

Possible extensions include:

- real-time simulation visualization
- stochastic recovery models
- age or network-based contact structures
- reinforcement learning policy experiments
- cloud deployment for large-scale simulation experiments
