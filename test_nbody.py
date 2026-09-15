import numpy as np
import pytest

import bodies, forces
from src.integrators import euler_step, rk4_step, velocity_verlet_step
from src.simulation import run_simulation


def test_no_self_interaction():
    positions = np.array([[0.0, 0.0], [1.0, 0.0]])
    masses = np.array([1.0, 1.0])
    acc = forces.compute_accelerations(positions, masses, softening=0.0)
    # body 0 should be pulled toward body 1 (positive x), and vice versa
    assert acc[0, 0] > 0
    assert acc[1, 0] < 0
    assert not np.any(np.isnan(acc))


def test_newtons_third_law_two_body():
    """For a 2-body system, the force on each body must be equal and
    opposite when masses are equal (Newton's third law + symmetry)."""
    positions = np.array([[0.0, 0.0], [2.0, 0.0]])
    masses = np.array([3.0, 3.0])
    acc = forces.compute_accelerations(positions, masses)
    # equal masses -> equal-magnitude, opposite-direction accelerations
    np.testing.assert_allclose(acc[0], -acc[1], atol=1e-12)


def test_softening_prevents_singularity():
    positions = np.array([[0.0, 0.0], [1e-9, 0.0]])  # nearly coincident
    masses = np.array([1.0, 1.0])
    acc = forces.compute_accelerations(positions, masses, softening=1e-3)
    assert np.all(np.isfinite(acc))


def test_total_momentum_conserved_by_construction():
    system = bodies.random_cluster(n_bodies=10, seed=1)
    momentum = forces.total_momentum(system.velocities, system.masses)
    np.testing.assert_allclose(momentum, 0.0, atol=1e-10)


def test_figure_eight_masses_equal():
    system = bodies.figure_eight()
    assert system.n_bodies == 3
    np.testing.assert_allclose(system.masses, 1.0)


def test_solar_system_circular_orbit_speed():
    """Each planet's initial speed should match the analytic circular-orbit
    speed v = sqrt(G*M/r) for its initial radius (within seed randomness)."""
    system = bodies.solar_system_like(n_planets=3, central_mass=500.0, seed=2)
    G = 1.0
    for i in range(1, system.n_bodies):
        r = np.linalg.norm(system.positions[i])
        expected_speed = np.sqrt(G * 500.0 / r)
        actual_speed = np.linalg.norm(system.velocities[i])
        assert actual_speed == pytest.approx(expected_speed, rel=1e-9)


def test_integrators_agree_on_short_horizon():
    """Over a very short, well-resolved horizon, all three integrators
    should land in approximately the same place (they all approximate the
    same ODE; they just differ in how the error accumulates over time)."""
    system = bodies.figure_eight()
    accel_fn = lambda p, m: forces.compute_accelerations(p, m)
    dt = 1e-4

    pos_e, vel_e, _ = euler_step(system.positions, system.velocities, system.masses, dt, accel_fn)
    pos_v, vel_v, _ = velocity_verlet_step(system.positions, system.velocities, system.masses, dt, accel_fn)
    pos_r, vel_r, _ = rk4_step(system.positions, system.velocities, system.masses, dt, accel_fn)

    np.testing.assert_allclose(pos_e, pos_v, atol=1e-6)
    np.testing.assert_allclose(pos_e, pos_r, atol=1e-6)


def test_verlet_conserves_energy_better_than_euler_long_run():
    """The core stability claim: over a long integration, symplectic
    Verlet's relative energy drift must stay much smaller than Euler's."""
    system = bodies.figure_eight()
    n_steps = 20_000
    dt = 1e-3

    euler_result = run_simulation(system, integrator="euler", dt=dt, n_steps=n_steps)
    verlet_result = run_simulation(system, integrator="verlet", dt=dt, n_steps=n_steps)

    euler_drift = euler_result.relative_energy_drift[-1]
    verlet_drift = verlet_result.relative_energy_drift[-1]

    assert verlet_drift < euler_drift
    assert verlet_drift < 1e-2   # Verlet should stay well-bounded
    assert euler_drift > verlet_drift * 10  # Euler should be meaningfully worse


def test_energy_history_shape_matches_recording():
    system = bodies.figure_eight()
    result = run_simulation(system, integrator="verlet", dt=1e-3, n_steps=100, record_every=1)
    assert result.energy_history.shape[0] == 101
    assert result.positions_history.shape == (101, 3, 2)
