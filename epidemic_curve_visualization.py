"""
epidemic_curve_visualization.py
--------------------------------
Scientific visualization of the SIR epidemic curve for a single simulation run.

Produces a publication-quality line chart showing how Susceptible, Infected, and
Recovered/Immune populations evolve over time, with the peak infection point and
outbreak statistics clearly annotated.

Usage:
    python epidemic_curve_visualization.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import numpy as np

from simulator import SimulationConfig, run_simulation

# ── Simulation parameters ─────────────────────────────────────────────────────

CONFIG = SimulationConfig(
    population_size      = 400,
    infection_probability= 0.35,
    recovery_time        = 14,
    movement_step        = 2.2,
    infection_radius     = 2.8,
    vaccinated_fraction  = 0.15,
    initial_infected     = 10,
    max_steps            = 300,
    seed                 = 42,
)

# ── Style constants ────────────────────────────────────────────────────────────

COLOR_S   = "#2166AC"   # steel blue   – Susceptible
COLOR_I   = "#D6604D"   # brick red    – Infected
COLOR_R   = "#4DAC26"   # grass green  – Recovered / Immune
COLOR_BG  = "#F8F9FA"
COLOR_AX  = "#2D2D2D"
COLOR_GRID= "#CCCCCC"

FONT_MAIN = "DejaVu Sans"  # always available with matplotlib

# ── Main routine ──────────────────────────────────────────────────────────────

def build_epidemic_curve(out_dir: Path = Path("outputs")) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Run simulation ────────────────────────────────────────────────────────
    result   = run_simulation(CONFIG)
    timeline = result["timeline"]     # list of {timestep, susceptible, infected, recovered}
    summary  = result["summary"]

    days        = np.array([row["timestep"]   for row in timeline])
    susceptible = np.array([row["susceptible"] for row in timeline])
    infected    = np.array([row["infected"]    for row in timeline])
    recovered   = np.array([row["recovered"]   for row in timeline])

    # ── Locate peak infection ─────────────────────────────────────────────────
    peak_idx  = int(np.argmax(infected))
    peak_day  = int(days[peak_idx])
    peak_val  = int(infected[peak_idx])

    # ── Find outbreak end (infected drops to 0 after peak) ───────────────────
    post_peak = np.where((days > peak_day) & (infected == 0))[0]
    end_day   = int(days[post_peak[0]]) if len(post_peak) else int(days[-1])

    total_affected = int(summary.get("total_infected_population", recovered[-1]))
    duration       = int(summary.get("outbreak_duration", end_day))

    # ── Figure setup ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6.5))
    fig.patch.set_facecolor(COLOR_BG)
    ax.set_facecolor(COLOR_BG)

    # ── Filled area beneath each curve (subtle) ───────────────────────────────
    ax.fill_between(days, susceptible, alpha=0.07, color=COLOR_S)
    ax.fill_between(days, infected,    alpha=0.10, color=COLOR_I)
    ax.fill_between(days, recovered,   alpha=0.07, color=COLOR_R)

    # ── Main curves ───────────────────────────────────────────────────────────
    ax.plot(days, susceptible, color=COLOR_S, linewidth=2.2, label="Susceptible",
            solid_capstyle="round", zorder=3)
    ax.plot(days, infected,    color=COLOR_I, linewidth=2.5, label="Infected",
            solid_capstyle="round", zorder=4)
    ax.plot(days, recovered,   color=COLOR_R, linewidth=2.2, label="Recovered / Immune",
            solid_capstyle="round", zorder=3)

    # ── Peak annotation ───────────────────────────────────────────────────────
    ax.scatter([peak_day], [peak_val], color=COLOR_I, s=90, zorder=6,
               edgecolors="white", linewidths=1.8)

    # Dashed vertical line at peak
    ax.axvline(peak_day, color=COLOR_I, linewidth=1.0, linestyle="--",
               alpha=0.55, zorder=2)

    # Arrow + text annotation
    offset_x = max(5, (days[-1] - peak_day) * 0.05)
    ax.annotate(
        f"Peak Infection\nDay {peak_day}  –  {peak_val} individuals",
        xy=(peak_day, peak_val),
        xytext=(peak_day + offset_x + 10, peak_val + CONFIG.population_size * 0.06),
        fontsize=9.5,
        color=COLOR_AX,
        arrowprops=dict(arrowstyle="-|>", color=COLOR_I,
                        lw=1.5, connectionstyle="arc3,rad=-0.18"),
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor=COLOR_I, linewidth=1.2, alpha=0.90),
        zorder=7,
    )

    # ── Statistics inset box ──────────────────────────────────────────────────
    stats_text = (
        f"Outbreak Summary\n"
        f"─────────────────────\n"
        f"Population         {CONFIG.population_size:>6}\n"
        f"Peak infected      {peak_val:>6}\n"
        f"Outbreak duration  {duration:>5}d\n"
        f"Total affected     {total_affected:>6}\n"
        f"Vaccination rate   {int(CONFIG.vaccinated_fraction*100):>5}%\n"
        f"Infection prob.    {CONFIG.infection_probability:>6.2f}"
    )
    ax.text(
        0.985, 0.97, stats_text,
        transform=ax.transAxes,
        fontsize=8.8,
        verticalalignment="top",
        horizontalalignment="right",
        family="monospace",
        color=COLOR_AX,
        bbox=dict(boxstyle="round,pad=0.6", facecolor="white",
                  edgecolor="#BBBBBB", linewidth=1.0, alpha=0.92),
        zorder=8,
    )

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(facecolor=COLOR_S, label="Susceptible"),
        mpatches.Patch(facecolor=COLOR_I, label="Infected"),
        mpatches.Patch(facecolor=COLOR_R, label="Recovered / Immune"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        framealpha=0.90,
        edgecolor="#BBBBBB",
        fontsize=10,
        handlelength=1.4,
    )

    # ── Axis labels & title ───────────────────────────────────────────────────
    ax.set_xlabel("Time (days)", fontsize=12, color=COLOR_AX, labelpad=8)
    ax.set_ylabel("Number of individuals", fontsize=12, color=COLOR_AX, labelpad=8)

    ax.set_title(
        "SIR Epidemic Curve  —  Agent-Based Infectious Disease Simulation",
        fontsize=14, fontweight="bold", color=COLOR_AX, pad=14,
    )
    subtitle = (
        f"N = {CONFIG.population_size}   |   "
        f"β = {CONFIG.infection_probability}   |   "
        f"γ = 1/{CONFIG.recovery_time}   |   "
        f"Radius = {CONFIG.infection_radius}   |   "
        f"Speed = {CONFIG.movement_step}   |   "
        f"Vaccinated = {int(CONFIG.vaccinated_fraction*100)}%"
    )
    ax.text(0.5, 1.013, subtitle, transform=ax.transAxes,
            ha="center", va="bottom", fontsize=9, color="#555555")

    # ── Grid & spines ─────────────────────────────────────────────────────────
    ax.grid(axis="both", color=COLOR_GRID, linewidth=0.65, linestyle="--", alpha=0.8)
    ax.set_axisbelow(True)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#AAAAAA")
        ax.spines[spine].set_linewidth(0.8)

    ax.tick_params(colors=COLOR_AX, labelsize=10)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    ax.tick_params(which="minor", length=3, color=COLOR_GRID)

    ax.set_xlim(left=0, right=days[-1])
    ax.set_ylim(bottom=0, top=CONFIG.population_size * 1.07)

    # ── Save ──────────────────────────────────────────────────────────────────
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = out_dir / "epidemic_curve_scientific.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=COLOR_BG)
    plt.close()

    print(f"[OK] Saved: {out_path.resolve()}")
    print(f"     Peak: {peak_val} infected on day {peak_day}")
    print(f"     Duration: {duration} days  |  Total affected: {total_affected}")


if __name__ == "__main__":
    build_epidemic_curve()
