from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from simulator import SimulationConfig, run_simulation


def _plot_epidemic_curve(timeline: list[dict[str, int]], out_dir: Path) -> None:
    timesteps = [row["timestep"] for row in timeline]
    susceptible = [row["susceptible"] for row in timeline]
    infected = [row["infected"] for row in timeline]
    recovered = [row["recovered"] for row in timeline]

    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, susceptible, color="tab:blue", label="Susceptible")
    plt.plot(timesteps, infected, color="tab:red", label="Infected")
    plt.plot(timesteps, recovered, color="tab:green", label="Recovered")
    plt.xlabel("Timestep")
    plt.ylabel("Population")
    plt.title("Epidemic Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "epidemic_curve.png", dpi=150)
    plt.close()


def _plot_spatial_snapshot(snapshot: dict[str, object], out_dir: Path, width: float, height: float) -> None:
    agents: list[dict[str, object]] = snapshot["agents"]  # type: ignore[assignment]
    timestep = snapshot["timestep"]

    susceptible_x, susceptible_y = [], []
    infected_x, infected_y = [], []
    recovered_x, recovered_y = [], []

    for agent in agents:
        state = agent["state"]
        if state == "S":
            susceptible_x.append(agent["x"])
            susceptible_y.append(agent["y"])
        elif state == "I":
            infected_x.append(agent["x"])
            infected_y.append(agent["y"])
        else:
            recovered_x.append(agent["x"])
            recovered_y.append(agent["y"])

    plt.figure(figsize=(7, 7))
    plt.scatter(susceptible_x, susceptible_y, c="blue", s=15, alpha=0.8, label="Susceptible")
    plt.scatter(infected_x, infected_y, c="red", s=20, alpha=0.9, label="Infected")
    plt.scatter(recovered_x, recovered_y, c="green", s=15, alpha=0.8, label="Recovered")
    plt.xlim(0, width)
    plt.ylim(0, height)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"Spatial State at Timestep {timestep}")
    plt.legend(loc="upper right")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_dir / f"spatial_t{int(timestep):03d}.png", dpi=150)
    plt.close()


def main() -> None:
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    config = SimulationConfig(
        population_size=300,
        initial_infected=5,
        vaccinated_fraction=0.1,
        infection_probability=0.3,
        recovery_time=18,
        max_steps=120,
        seed=42,
    )

    result = run_simulation(config, include_agent_snapshots=True)
    timeline: list[dict[str, int]] = result["timeline"]  # type: ignore[assignment]
    snapshots: list[dict[str, object]] = result["snapshots"]  # type: ignore[assignment]

    _plot_epidemic_curve(timeline, out_dir)

    sample_indices = sorted({0, len(snapshots) // 2, len(snapshots) - 1})
    for idx in sample_indices:
        _plot_spatial_snapshot(snapshots[idx], out_dir, config.width, config.height)

    print("Saved plots:")
    print("  outputs/epidemic_curve.png")
    for idx in sample_indices:
        timestep = snapshots[idx]["timestep"]
        print(f"  outputs/spatial_t{int(timestep):03d}.png")

    summary = result["summary"]
    print("\nSummary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
