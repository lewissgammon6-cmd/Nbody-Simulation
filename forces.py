"""
forces.py
---------
Vectorized N-body gravitational force computation. All pairwise
interactions are computed at once via NumPy array broadcasting -- no
Python-level loop over body pairs -- which is what makes it practical to
integrate systems of many bodies over long time horizons.

A Plummer softening length is used to prevent the force (and therefore
the acceleration) from diverging when two bodies pass very close to each
other, which would otherwise force integrators down to absurdly small
time steps or produce numerical blow-ups.
"""

from __future__ import annotations

import numpy as np

G_DEFAULT = 1.0          # gravitational constant in simulation units
SOFTENING_DEFAULT = 1e-3  # Plummer softening length, in position units


def pairwise_separations(positions: np.ndarray) -> np.ndarray:
    """
    positions: (N, D) array.
    Returns diff: (N, N, D) array where diff[i, j] = positions[j] - positions[i],
    i.e. the vector pointing from body i toward body j (the direction of
    the gravitational pull body i feels from body j).
    """
    return positions[np.newaxis, :, :] - positions[:, np.newaxis, :]


def compute_accelerations(
    positions: np.ndarray,
    masses: np.ndarray,
    G: float = G_DEFAULT,
    softening: float = SOFTENING_DEFAULT,
) -> np.ndarray:
    """
    Compute the gravitational acceleration on every body due to every
    other body, fully vectorized.

    positions: (N, D) array of body positions.
    masses:    (N,)   array of body masses.
    Returns:   (N, D) array of accelerations, a_i = G * sum_j m_j * (r_j - r_i) / (|r_j - r_i|^2 + eps^2)^{3/2}
    """
    diff = pairwise_separations(positions)                      # (N, N, D)
    dist_sq = np.sum(diff * diff, axis=-1) + softening**2        # (N, N)
    np.fill_diagonal(dist_sq, np.inf)                             # a body doesn't act on itself;
                                                                   # inf**-1.5 == 0, no div-by-zero warning
    inv_dist_cubed = dist_sq ** (-1.5)                            # (N, N)

    # acc[i] = G * sum_j masses[j] * diff[i, j] * inv_dist_cubed[i, j]
    weighted = masses[np.newaxis, :, np.newaxis] * inv_dist_cubed[:, :, np.newaxis]  # (N, N, 1)
    acc = G * np.sum(weighted * diff, axis=1)                     # (N, D)
    return acc


def potential_energy(
    positions: np.ndarray,
    masses: np.ndarray,
    G: float = G_DEFAULT,
    softening: float = SOFTENING_DEFAULT,
) -> float:
    """Total gravitational potential energy, summed once per unique pair."""
    diff = pairwise_separations(positions)
    dist = np.sqrt(np.sum(diff * diff, axis=-1) + softening**2)
    n = positions.shape[0]
    iu = np.triu_indices(n, k=1)  # each pair counted once
    pair_pe = -G * masses[iu[0]] * masses[iu[1]] / dist[iu]
    return float(np.sum(pair_pe))


def kinetic_energy(velocities: np.ndarray, masses: np.ndarray) -> float:
    speed_sq = np.sum(velocities * velocities, axis=-1)
    return float(0.5 * np.sum(masses * speed_sq))


def total_energy(
    positions: np.ndarray,
    velocities: np.ndarray,
    masses: np.ndarray,
    G: float = G_DEFAULT,
    softening: float = SOFTENING_DEFAULT,
) -> float:
    return kinetic_energy(velocities, masses) + potential_energy(positions, masses, G, softening)


def total_momentum(velocities: np.ndarray, masses: np.ndarray) -> np.ndarray:
    return np.sum(masses[:, np.newaxis] * velocities, axis=0)
