from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from random import Random
from typing import Dict, List, Literal, Tuple

State = Literal["S", "I", "R"]


@dataclass
class Agent:
    id: int
    x: float
    y: float
    state: State
    infection_timer: int
    vaccinated: bool


@dataclass
class SimulationConfig:
    population_size: int = 300
    initial_infected: int = 3
    vaccinated_fraction: float = 0.0
    width: float = 100.0
    height: float = 100.0
    movement_step: float = 1.5
    infection_radius: float = 2.0
    infection_probability: float = 0.25
    recovery_time: int = 20
    max_steps: int = 300
    seed: int = 42


@dataclass
class TimeStepRecord:
    timestep: int
    susceptible: int
    infected: int
    recovered: int


@dataclass
class AgentSnapshot:
    id: int
    x: float
    y: float
    state: State


def _bounded(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


def _initialize_agents(config: SimulationConfig, rng: Random) -> List[Agent]:
    if config.initial_infected > config.population_size:
        raise ValueError("initial_infected cannot exceed population_size")
    if not (0.0 <= config.vaccinated_fraction <= 1.0):
        raise ValueError("vaccinated_fraction must be between 0 and 1")
    if config.recovery_time <= 0:
        raise ValueError("recovery_time must be greater than 0")

    agents: List[Agent] = []
    vaccinated_count = int(config.population_size * config.vaccinated_fraction)

    for i in range(config.population_size):
        state: State = "S"
        infection_timer = 0

        if i < config.initial_infected:
            state = "I"
            infection_timer = config.recovery_time

        vaccinated = i >= config.initial_infected and i < (config.initial_infected + vaccinated_count)

        agents.append(
            Agent(
                id=i,
                x=rng.uniform(0, config.width),
                y=rng.uniform(0, config.height),
                state=state,
                infection_timer=infection_timer,
                vaccinated=vaccinated,
            )
        )

    rng.shuffle(agents)

    # Reassign IDs after shuffling so IDs remain unique and sequential.
    for idx, agent in enumerate(agents):
        agent.id = idx

    return agents


def _move_agents(agents: List[Agent], config: SimulationConfig, rng: Random) -> None:
    for agent in agents:
        dx = rng.uniform(-config.movement_step, config.movement_step)
        dy = rng.uniform(-config.movement_step, config.movement_step)
        agent.x = _bounded(agent.x + dx, 0.0, config.width)
        agent.y = _bounded(agent.y + dy, 0.0, config.height)


def _spread_infection(agents: List[Agent], config: SimulationConfig, rng: Random) -> int:
    infected_agents = [a for a in agents if a.state == "I"]
    susceptible_agents = [a for a in agents if a.state == "S"]
    newly_infected: List[Agent] = []

    for susceptible in susceptible_agents:
        if susceptible.vaccinated:
            continue

        for infected in infected_agents:
            distance = hypot(susceptible.x - infected.x, susceptible.y - infected.y)
            if distance <= config.infection_radius and rng.random() < config.infection_probability:
                newly_infected.append(susceptible)
                break

    for agent in newly_infected:
        agent.state = "I"
        agent.infection_timer = config.recovery_time

    return len(newly_infected)


def _update_recovery(agents: List[Agent]) -> None:
    for agent in agents:
        if agent.state == "I":
            agent.infection_timer -= 1
            if agent.infection_timer <= 0:
                agent.state = "R"
                agent.infection_timer = 0


def _count_states(agents: List[Agent]) -> Tuple[int, int, int]:
    susceptible = 0
    infected = 0
    recovered = 0

    for agent in agents:
        if agent.state == "S":
            susceptible += 1
        elif agent.state == "I":
            infected += 1
        else:
            recovered += 1

    return susceptible, infected, recovered


def _snapshot_agents(agents: List[Agent]) -> List[AgentSnapshot]:
    return [
        AgentSnapshot(
            id=agent.id,
            x=agent.x,
            y=agent.y,
            state=agent.state,
        )
        for agent in agents
    ]


def run_simulation(config: SimulationConfig, include_agent_snapshots: bool = False) -> Dict[str, object]:
    rng = Random(config.seed)
    agents = _initialize_agents(config, rng)

    timeline: List[TimeStepRecord] = []
    snapshots: List[Dict[str, object]] = []
    ever_infected_ids = {agent.id for agent in agents if agent.state == "I"}

    for timestep in range(config.max_steps + 1):
        susceptible, infected, recovered = _count_states(agents)
        timeline.append(
            TimeStepRecord(
                timestep=timestep,
                susceptible=susceptible,
                infected=infected,
                recovered=recovered,
            )
        )
        if include_agent_snapshots:
            snapshots.append(
                {
                    "timestep": timestep,
                    "agents": [
                        {
                            "id": row.id,
                            "x": row.x,
                            "y": row.y,
                            "state": row.state,
                        }
                        for row in _snapshot_agents(agents)
                    ],
                }
            )

        if infected == 0:
            break

        _move_agents(agents, config, rng)
        _spread_infection(agents, config, rng)

        for agent in agents:
            if agent.state == "I":
                ever_infected_ids.add(agent.id)

        _update_recovery(agents)

    peak_infected = max(record.infected for record in timeline)
    outbreak_duration = timeline[-1].timestep
    total_infected_population = len(ever_infected_ids)

    result: Dict[str, object] = {
        "parameters": {
            "population_size": config.population_size,
            "initial_infected": config.initial_infected,
            "vaccinated_fraction": config.vaccinated_fraction,
            "infection_probability": config.infection_probability,
            "recovery_time": config.recovery_time,
            "max_steps": config.max_steps,
        },
        "timeline": [
            {
                "timestep": row.timestep,
                "susceptible": row.susceptible,
                "infected": row.infected,
                "recovered": row.recovered,
            }
            for row in timeline
        ],
        "summary": {
            "peak_infected_population": peak_infected,
            "outbreak_duration": outbreak_duration,
            "total_infected_population": total_infected_population,
        },
    }

    if include_agent_snapshots:
        result["snapshots"] = snapshots

    return result
