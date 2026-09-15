"""
integrators.py
--------------
Three numerical integration schemes for the N-body ODE system
    d(position)/dt = velocity
    d(velocity)/dt = acceleration(position)

- Explicit Euler: first-order, not symplectic. Included as the baseline
  everyone is taught first, and as the "what NOT to do for long
  integrations" comparison point in the README.
- Velocity Verlet: second-order and symplectic. Symplectic integrators
  don't conserve energy exactly at every step, but the error stays
  *bounded* (oscillates) over arbitrarily long integrations rather than
  drifting monotonically -- this is the standard method for real orbital
  simulations.
- RK4 (4th-order Runge-Kutta): much higher pointwise accuracy per step
  than Euler for a given step size, but -- being non-symplectic -- its
  energy error still drifts secularly over very long integrations,
  just far more slowly than Euler's.

Each stepper has the same signature so `simulation.py` can swap them
interchangeably: step(pos, vel, masses, dt, accel_fn, **state) -> (new_pos, new_vel, new_state)
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import numpy as np

AccelFn = Callable[[np.ndarray, np.ndarray], np.ndarray]  # (positions, masses) -> accelerations


class StepResult(NamedTuple):
    positions: np.ndarray
    velocities: np.ndarray
    # Cached acceleration at the *new* positions, so the next Verlet step
    # doesn't have to recompute the force it already knows. Other
    # integrators ignore this field (they recompute fresh each call).
    acceleration: np.ndarray | None


def euler_step(pos: np.ndarray, vel: np.ndarray, masses: np.ndarray, dt: float,
                accel_fn: AccelFn, acceleration: np.ndarray | None = None) -> StepResult:
    acc = accel_fn(pos, masses)
    new_pos = pos + vel * dt
    new_vel = vel + acc * dt
    return StepResult(new_pos, new_vel, None)


def velocity_verlet_step(pos: np.ndarray, vel: np.ndarray, masses: np.ndarray, dt: float,
                          accel_fn: AccelFn, acceleration: np.ndarray | None = None) -> StepResult:
    acc = acceleration if acceleration is not None else accel_fn(pos, masses)
    new_pos = pos + vel * dt + 0.5 * acc * dt**2
    new_acc = accel_fn(new_pos, masses)
    new_vel = vel + 0.5 * (acc + new_acc) * dt
    return StepResult(new_pos, new_vel, new_acc)


def rk4_step(pos: np.ndarray, vel: np.ndarray, masses: np.ndarray, dt: float,
             accel_fn: AccelFn, acceleration: np.ndarray | None = None) -> StepResult:
    def deriv(p, v):
        return v, accel_fn(p, masses)

    k1p, k1v = deriv(pos, vel)
    k2p, k2v = deriv(pos + 0.5 * dt * k1p, vel + 0.5 * dt * k1v)
    k3p, k3v = deriv(pos + 0.5 * dt * k2p, vel + 0.5 * dt * k2v)
    k4p, k4v = deriv(pos + dt * k3p, vel + dt * k3v)

    new_pos = pos + (dt / 6.0) * (k1p + 2 * k2p + 2 * k3p + k4p)
    new_vel = vel + (dt / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)
    return StepResult(new_pos, new_vel, None)


INTEGRATORS = {
    "euler": euler_step,
    "verlet": velocity_verlet_step,
    "rk4": rk4_step,
}
