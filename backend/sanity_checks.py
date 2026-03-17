from __future__ import annotations

from statistics import mean

from simulator import SimulationConfig, run_simulation


def _run_batch(config: SimulationConfig, runs: int = 12) -> dict[str, float]:
    peaks = []
    durations = []
    total_infected = []
    final_totals = []

    for i in range(runs):
        config.seed = 1000 + i
        result = run_simulation(config)
        timeline = result["timeline"]
        summary = result["summary"]

        last_row = timeline[-1]
        final_totals.append(last_row["susceptible"] + last_row["infected"] + last_row["recovered"])
        peaks.append(summary["peak_infected_population"])
        durations.append(summary["outbreak_duration"])
        total_infected.append(summary["total_infected_population"])

    return {
        "avg_peak": mean(peaks),
        "avg_duration": mean(durations),
        "avg_total_infected": mean(total_infected),
        "avg_final_population": mean(final_totals),
    }


def main() -> None:
    base = SimulationConfig(
        population_size=300,
        initial_infected=5,
        vaccinated_fraction=0.1,
        infection_probability=0.25,
        recovery_time=16,
        max_steps=180,
        seed=42,
    )

    print("Running sanity checks (multi-run averages) ...\n")

    high_infection = _run_batch(SimulationConfig(**{**base.__dict__, "infection_probability": 0.45}))
    low_infection = _run_batch(SimulationConfig(**{**base.__dict__, "infection_probability": 0.10}))

    high_vaccination = _run_batch(SimulationConfig(**{**base.__dict__, "vaccinated_fraction": 0.45}))
    low_vaccination = _run_batch(SimulationConfig(**{**base.__dict__, "vaccinated_fraction": 0.0}))

    long_recovery = _run_batch(SimulationConfig(**{**base.__dict__, "recovery_time": 28}))
    short_recovery = _run_batch(SimulationConfig(**{**base.__dict__, "recovery_time": 8}))

    print("1) Higher infection probability should spread faster")
    print(f"   low p avg_peak={low_infection['avg_peak']:.2f}, avg_duration={low_infection['avg_duration']:.2f}")
    print(f"   high p avg_peak={high_infection['avg_peak']:.2f}, avg_duration={high_infection['avg_duration']:.2f}")
    infection_check = high_infection["avg_peak"] > low_infection["avg_peak"]
    print(f"   RESULT: {'PASS' if infection_check else 'CHECK'}")

    print("\n2) Higher vaccination should reduce outbreak size")
    print(
        f"   low vax avg_total_infected={low_vaccination['avg_total_infected']:.2f}, avg_peak={low_vaccination['avg_peak']:.2f}"
    )
    print(
        f"   high vax avg_total_infected={high_vaccination['avg_total_infected']:.2f}, avg_peak={high_vaccination['avg_peak']:.2f}"
    )
    vax_check = high_vaccination["avg_total_infected"] < low_vaccination["avg_total_infected"]
    print(f"   RESULT: {'PASS' if vax_check else 'CHECK'}")

    print("\n3) Longer recovery should sustain infections longer")
    print(
        f"   short recovery avg_peak={short_recovery['avg_peak']:.2f}, avg_duration={short_recovery['avg_duration']:.2f}"
    )
    print(
        f"   long recovery avg_peak={long_recovery['avg_peak']:.2f}, avg_duration={long_recovery['avg_duration']:.2f}"
    )
    recovery_check = long_recovery["avg_duration"] > short_recovery["avg_duration"]
    print(f"   RESULT: {'PASS' if recovery_check else 'CHECK'}")

    print("\n4) Total population should remain constant")
    print(
        f"   observed avg final total={low_vaccination['avg_final_population']:.2f} (expected {base.population_size})"
    )
    population_check = abs(low_vaccination["avg_final_population"] - base.population_size) < 1e-6
    print(f"   RESULT: {'PASS' if population_check else 'CHECK'}")

    all_pass = infection_check and vax_check and recovery_check and population_check
    print("\nMilestone verdict:")
    if all_pass:
        print("  I can run one simulation, see the outbreak evolve, and trust the results.")
    else:
        print("  Some checks need tuning before trusting results.")


if __name__ == "__main__":
    main()
