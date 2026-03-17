from simulator import SimulationConfig, run_simulation


if __name__ == "__main__":
    config = SimulationConfig(
        population_size=200,
        initial_infected=5,
        infection_probability=0.2,
        recovery_time=15,
        max_steps=200,
        seed=11,
    )

    result = run_simulation(config)
    print("Summary:")
    for key, value in result["summary"].items():
        print(f"  {key}: {value}")

    print("\nFirst 5 timeline rows:")
    for row in result["timeline"][:5]:
        print(row)
