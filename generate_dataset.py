from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from random import Random
from typing import Iterable, List

import pandas as pd

from simulator import SimulationConfig, run_simulation


@dataclass
class SweepParameterSet:
    population_size: int
    infection_probability: float
    recovery_time: int
    movement_speed: float
    interaction_radius: float
    vaccination_rate: float
    initial_infected: int


def _grid_parameter_sets() -> List[SweepParameterSet]:
    population_sizes = [200, 350, 500]
    infection_probabilities = [0.1, 0.3, 0.5]
    recovery_times = [8, 14, 20]
    movement_speeds = [1.0, 2.0, 3.0]
    interaction_radii = [1.5, 2.5, 3.5]
    vaccination_rates = [0.0, 0.2, 0.4]
    initial_infected_values = [3, 8, 15]

    combinations: List[SweepParameterSet] = []
    for pop in population_sizes:
        for p_infect in infection_probabilities:
            for recovery in recovery_times:
                for move in movement_speeds:
                    for radius in interaction_radii:
                        for vax in vaccination_rates:
                            for initial in initial_infected_values:
                                if initial >= pop:
                                    continue
                                combinations.append(
                                    SweepParameterSet(
                                        population_size=pop,
                                        infection_probability=p_infect,
                                        recovery_time=recovery,
                                        movement_speed=move,
                                        interaction_radius=radius,
                                        vaccination_rate=vax,
                                        initial_infected=initial,
                                    )
                                )
    return combinations


def _random_parameter_sets(count: int, rng: Random) -> List[SweepParameterSet]:
    sets: List[SweepParameterSet] = []
    for _ in range(count):
        pop = rng.choice([200, 300, 400, 500, 700])
        init = rng.randint(2, max(2, int(pop * 0.08)))
        sets.append(
            SweepParameterSet(
                population_size=pop,
                infection_probability=round(rng.uniform(0.05, 0.60), 3),
                recovery_time=rng.randint(5, 24),
                movement_speed=round(rng.uniform(0.5, 4.0), 3),
                interaction_radius=round(rng.uniform(1.0, 4.0), 3),
                vaccination_rate=round(rng.uniform(0.0, 0.6), 3),
                initial_infected=init,
            )
        )
    return sets


def _build_simulation_config(params: SweepParameterSet, seed: int, max_steps: int) -> SimulationConfig:
    return SimulationConfig(
        population_size=params.population_size,
        initial_infected=params.initial_infected,
        vaccinated_fraction=params.vaccination_rate,
        width=100.0,
        height=100.0,
        movement_step=params.movement_speed,
        infection_radius=params.interaction_radius,
        infection_probability=params.infection_probability,
        recovery_time=params.recovery_time,
        max_steps=max_steps,
        seed=seed,
    )


def _run_experiments(parameter_sets: Iterable[SweepParameterSet], repeats: int, base_seed: int, max_steps: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset_rows = []
    timeline_rows = []

    run_index = 0
    for params in parameter_sets:
        for repeat_index in range(repeats):
            seed = base_seed + run_index
            config = _build_simulation_config(params, seed=seed, max_steps=max_steps)
            result = run_simulation(config)

            summary = result["summary"]
            timeline = result["timeline"]

            dataset_rows.append(
                {
                    **asdict(params),
                    "repeat_index": repeat_index,
                    "seed": seed,
                    "peak_infected_population": summary["peak_infected_population"],
                    "outbreak_duration": summary["outbreak_duration"],
                    "total_infected_population": summary["total_infected_population"],
                }
            )

            for row in timeline:
                timeline_rows.append(
                    {
                        "run_id": run_index,
                        "repeat_index": repeat_index,
                        "seed": seed,
                        **asdict(params),
                        "timestep": row["timestep"],
                        "susceptible": row["susceptible"],
                        "infected": row["infected"],
                        "recovered": row["recovered"],
                    }
                )

            run_index += 1

    return pd.DataFrame(dataset_rows), pd.DataFrame(timeline_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate epidemic simulation datasets")
    parser.add_argument("--mode", choices=["random", "grid"], default="random", help="How to create parameter combinations")
    parser.add_argument("--num-sets", type=int, default=250, help="Number of random parameter sets (used when mode=random)")
    parser.add_argument("--repeats", type=int, default=1, help="Repeat simulations per parameter set with different seeds")
    parser.add_argument("--max-steps", type=int, default=250, help="Max timesteps per simulation")
    parser.add_argument("--seed", type=int, default=2026, help="Base random seed for sampling and runs")
    parser.add_argument(
        "--include-timesteps",
        action="store_true",
        help="Also export raw per-timestep curves to a separate CSV",
    )
    parser.add_argument("--out-dir", default="outputs", help="Output directory for generated CSV files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sampling_rng = Random(args.seed)

    if args.mode == "grid":
        parameter_sets = _grid_parameter_sets()
    else:
        parameter_sets = _random_parameter_sets(args.num_sets, sampling_rng)

    dataset_df, timeline_df = _run_experiments(
        parameter_sets=parameter_sets,
        repeats=args.repeats,
        base_seed=args.seed,
        max_steps=args.max_steps,
    )

    dataset_path = out_dir / "simulation_training_data.csv"
    dataset_df.to_csv(dataset_path, index=False)

    print(f"Saved dataset with {len(dataset_df)} rows to {dataset_path.as_posix()}")
    print("Columns:")
    print(", ".join(dataset_df.columns.tolist()))

    if args.include_timesteps:
        timeline_path = out_dir / "simulation_timestep_curves.csv"
        timeline_df.to_csv(timeline_path, index=False)
        print(f"Saved timestep curves with {len(timeline_df)} rows to {timeline_path.as_posix()}")


if __name__ == "__main__":
    main()
