"""
simulation.py
-------------
Runs an N-body simulation for a chosen integrator and records the full
trajectory plus per-step energy/momentum diagnostics, used both for
visualization and for the energy-conservation stability analysis in the
README.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from bodies import System
from .forces import compute_accelerations, total_energy, total_momentum
from .integrators import INTEGRATORS


@dataclass
class SimulationResult:
    positions_history: np.ndarray   # (n_steps+1, N, D)
    energy_history: np.ndarray      # (n_steps+1,)
    momentum_history: np.ndarray    # (n_steps+1, D)
    times: np.ndarray               # (n_steps+1,)
    elapsed_seconds: float
    integrator: str
    dt: float

    @property
    def relative_energy_drift(self) -> np.ndarray:
        e0 = self.energy_history[0]
        return np.abs((self.energy_history - e0) / e0)


def run_simulation(
    system: System,
    integrator: str = "verlet",
    dt: float = 1e-3,
    n_steps: int = 10_000,
    G: float = 1.0,
    softening: float = 1e-3,
    record_every: int = 1,
) -> SimulationResult:
    if integrator not in INTEGRATORS:
        raise ValueError(f"Unknown integrator '{integrator}'. Choose from {list(INTEGRATORS)}")
    step_fn = INTEGRATORS[integrator]

    accel_fn = lambda p, m: compute_accelerations(p, m, G=G, softening=softening)

    pos = system.positions.copy()
    vel = system.velocities.copy()
    masses = system.masses
    acc_cache = None

    n_recorded = n_steps // record_every + 1
    positions_history = np.empty((n_recorded, *pos.shape))
    energy_history = np.empty(n_recorded)
    momentum_history = np.empty((n_recorded, pos.shape[1]))
    times = np.empty(n_recorded)

    positions_history[0] = pos
    energy_history[0] = total_energy(pos, vel, masses, G, softening)
    momentum_history[0] = total_momentum(vel, masses)
    times[0] = 0.0

    t0 = time.perf_counter()
    record_idx = 1
    for step in range(1, n_steps + 1):
        pos, vel, acc_cache = step_fn(pos, vel, masses, dt, accel_fn, acc_cache)
        if step % record_every == 0:
            positions_history[record_idx] = pos
            energy_history[record_idx] = total_energy(pos, vel, masses, G, softening)
            momentum_history[record_idx] = total_momentum(vel, masses)
            times[record_idx] = step * dt
            record_idx += 1
    elapsed = time.perf_counter() - t0

    return SimulationResult(
        positions_history=positions_history,
        energy_history=energy_history,
        momentum_history=momentum_history,
        times=times,
        elapsed_seconds=elapsed,
        integrator=integrator,
        dt=dt,
    )
