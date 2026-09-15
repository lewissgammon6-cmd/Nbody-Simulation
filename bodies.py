"""
bodies.py
---------
Initial-condition generators for the N-body simulation, plus a small
System container that carries positions, velocities, masses, and names.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class System:
    positions: np.ndarray   # (N, D)
    velocities: np.ndarray  # (N, D)
    masses: np.ndarray      # (N,)
    names: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.names:
            self.names = [f"body_{i}" for i in range(len(self.masses))]

    @property
    def n_bodies(self) -> int:
        return len(self.masses)

    @property
    def n_dim(self) -> int:
        return self.positions.shape[1]


def figure_eight() -> System:
    """
    The Chenciner-Montgomery figure-eight three-body choreography: three
    equal masses chase each other around a single figure-eight-shaped
    orbit forever (in exact arithmetic). It's a famous stress-test for
    an integrator because any solution that isn't extremely accurate
    visibly falls off the choreography within a few periods -- making it
    an excellent way to *see* numerical error, not just measure it.

    Constants are the standard published initial conditions (period T ~= 6.3259).
    """
    m = np.array([1.0, 1.0, 1.0])

    p1 = np.array([-0.97000436, 0.24308753])
    p2 = np.array([0.97000436, -0.24308753])
    p3 = np.array([0.0, 0.0])
    positions = np.array([p1, p2, p3])

    v3 = np.array([-0.93240737, -0.86473146])
    v1 = -v3 / 2.0
    v2 = -v3 / 2.0
    velocities = np.array([v1, v2, v3])

    return System(positions, velocities, m, names=["A", "B", "C"])


def solar_system_like(n_planets: int = 4, central_mass: float = 1000.0,
                       seed: int = 0, G: float = 1.0) -> System:
    """
    A synthetic "star + planets" system: one heavy central body at rest,
    and n_planets lighter bodies on near-circular orbits at increasing
    radii, each given the exact circular-orbit speed v = sqrt(G*M/r) so
    the system starts in a physically sensible configuration rather than
    an arbitrary one.

    Note: because every planet also pulls on every other planet (this is
    a full N-body integration, not a fixed-central-force approximation),
    mutual perturbations can occasionally grow into a genuine close
    encounter between two planets over enough orbits -- this is real
    orbital chaos, not a bug, but it does mean some random seeds produce
    a system that goes unstable partway through a long run. Seed 0 is
    stable over the horizons used in this project's examples/README.
    """
    rng = np.random.default_rng(seed)
    n = n_planets + 1
    positions = np.zeros((n, 2))
    velocities = np.zeros((n, 2))
    masses = np.zeros(n)

    masses[0] = central_mass
    names = ["Star"]

    for i in range(1, n):
        radius = 2.0 + 1.8 * i + rng.uniform(-0.2, 0.2)
        angle = rng.uniform(0, 2 * np.pi)
        positions[i] = radius * np.array([np.cos(angle), np.sin(angle)])

        orbital_speed = np.sqrt(G * central_mass / radius)
        # velocity perpendicular to the radius vector -> circular orbit
        direction = np.array([-np.sin(angle), np.cos(angle)])
        velocities[i] = orbital_speed * direction

        masses[i] = rng.uniform(0.5, 5.0)
        names.append(f"Planet{i}")

    return System(positions, velocities, masses, names=names)


def random_cluster(n_bodies: int = 20, radius: float = 5.0, seed: int = 3,
                    mass_range: tuple[float, float] = (0.1, 2.0)) -> System:
    """
    A cluster of n_bodies scattered uniformly in a sphere/circle with
    random velocities, then recentered so total momentum is zero (a
    system with net momentum just drifts off-screen, which is
    physically fine but visually useless for demos).
    """
    rng = np.random.default_rng(seed)
    positions = rng.uniform(-radius, radius, size=(n_bodies, 2))
    velocities = rng.uniform(-0.3, 0.3, size=(n_bodies, 2))
    masses = rng.uniform(mass_range[0], mass_range[1], size=n_bodies)

    # Zero out net momentum in the center-of-mass frame.
    com_velocity = np.sum(masses[:, np.newaxis] * velocities, axis=0) / np.sum(masses)
    velocities -= com_velocity

    names = [f"body_{i}" for i in range(n_bodies)]
    return System(positions, velocities, masses, names=names)


SCENARIOS = {
    "figure8": figure_eight,
    "solar": solar_system_like,
    "random": random_cluster,
}
