"""
visualize.py
------------
Plotting and animation utilities: static trajectory plots, energy-drift
comparison charts across integrators, and an animated GIF of orbital
motion via matplotlib.animation.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

from bodies import System
from simulation import SimulationResult


def plot_trajectories(system: System, result: SimulationResult, out_path: str,
                       title: str | None = None) -> None:
    """Static 2D plot of every body's full path over the simulation."""
    fig, ax = plt.subplots(figsize=(7, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, system.n_bodies))

    for i in range(system.n_bodies):
        traj = result.positions_history[:, i, :]
        ax.plot(traj[:, 0], traj[:, 1], linewidth=1.2, color=colors[i], label=system.names[i])
        ax.scatter(traj[-1, 0], traj[-1, 1], color=colors[i], s=40, zorder=5)

    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title or f"Trajectories ({result.integrator}, dt={result.dt:g})")
    if system.n_bodies <= 12:
        ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_energy_drift(results: dict[str, SimulationResult], out_path: str,
                       title: str = "Relative energy drift by integrator") -> None:
    """
    Log-scale plot of |E(t) - E(0)| / |E(0)| for each integrator, on the
    same initial conditions -- the core evidence for the README's
    stability claim (symplectic Verlet stays bounded; Euler drifts).
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, result in results.items():
        drift = result.relative_energy_drift
        drift_safe = np.clip(drift, 1e-16, None)  # avoid log(0)
        ax.plot(result.times, drift_safe, label=name, linewidth=1.4)

    ax.set_yscale("log")
    ax.set_xlabel("Simulation time")
    ax.set_ylabel("|E(t) - E(0)| / |E(0)|  (log scale)")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def animate_system(system: System, result: SimulationResult, out_path: str,
                    trail_length: int = 200, fps: int = 30, stride: int = 1,
                    title: str | None = None) -> None:
    """
    Renders an animated GIF of the orbital motion with fading trails,
    using matplotlib.animation.FuncAnimation + PillowWriter (pure-Python
    GIF writer, no ffmpeg dependency required).
    """
    history = result.positions_history[::stride]
    n_frames = history.shape[0]
    colors = plt.cm.tab10(np.linspace(0, 1, system.n_bodies))

    fig, ax = plt.subplots(figsize=(6, 6))
    all_xy = history.reshape(-1, 2)
    pad = 0.1 * (all_xy.max() - all_xy.min() + 1e-9)
    ax.set_xlim(all_xy[:, 0].min() - pad, all_xy[:, 0].max() + pad)
    ax.set_ylim(all_xy[:, 1].min() - pad, all_xy[:, 1].max() + pad)
    ax.set_aspect("equal")
    ax.set_title(title or f"{result.integrator} integration")
    ax.grid(alpha=0.2)

    points = [ax.plot([], [], "o", color=colors[i], markersize=8)[0] for i in range(system.n_bodies)]
    trails = [ax.plot([], [], "-", color=colors[i], linewidth=1.0, alpha=0.6)[0] for i in range(system.n_bodies)]

    def update(frame_idx):
        start = max(0, frame_idx - trail_length)
        for i in range(system.n_bodies):
            points[i].set_data([history[frame_idx, i, 0]], [history[frame_idx, i, 1]])
            trails[i].set_data(history[start:frame_idx + 1, i, 0], history[start:frame_idx + 1, i, 1])
        return points + trails

    anim = FuncAnimation(fig, update, frames=n_frames, blit=True, interval=1000 / fps)
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
