![Tests](https://github.com/lewissgammon6-cmd/Nbody-Simulation/actions/workflows/tests.yml/badge.svg)
# N-Body Gravitational Simulation

A vectorized N-body gravitational simulator in Python, built to compare
numerical integration schemes on the metric that actually matters for
orbital mechanics: **long-term energy conservation**, not just
per-step accuracy.

![Figure-eight three-body orbit](README_assets/figure8_orbit.gif)

## Why this project exists

Any integrator can look fine for a few steps. The real test of a
numerical method for orbital dynamics is whether it stays physically
sensible over thousands of orbits — this project makes that comparison
directly, with real measured numbers, not just a claim.

## Architecture

```
nbody-simulation/
├── main.py                # CLI entry point
├── src/
│   ├── bodies.py           # System container + initial-condition generators
│   ├── forces.py           # vectorized gravitational force/energy computation
│   ├── integrators.py      # Euler, Velocity Verlet, RK4
│   ├── simulation.py       # orchestrates the integration loop, records history
│   └── visualize.py        # trajectory plots, energy-drift plots, GIF animation
└── tests/
    └── test_nbody.py       # physics correctness + integrator stability regression tests
```

**Force computation (`forces.py`)** — All pairwise gravitational
interactions are computed in one vectorized NumPy broadcast (an `(N, N,
D)` displacement tensor), not a Python loop over pairs. A Plummer
softening length prevents the force from diverging when two bodies pass
very close to each other.

**Integrators (`integrators.py`)** — Three schemes sharing one
interface so they're interchangeable in `simulation.py`:
- **Explicit Euler** — first-order, *not* symplectic. Included as the
  "what everyone learns first, and why it's not enough" baseline.
- **Velocity Verlet** — second-order and symplectic. Energy error stays
  *bounded* (oscillates) rather than drifting away, which is why this
  is the standard method for real orbital simulations.
- **RK4** — much higher pointwise accuracy per step, but being
  non-symplectic, its energy error still drifts secularly over very
  long integrations — just far more slowly than Euler's.

**Initial conditions (`bodies.py`)**:
- `figure_eight()` — the famous Chenciner-Montgomery three-body
  choreography, a well-known stress test where any inaccurate
  integrator visibly falls off the figure-eight within a few periods.
- `solar_system_like()` — a central mass plus planets on
  circular orbits, each given the exact analytic circular-orbit speed
  `v = sqrt(GM/r)`.
- `random_cluster()` — N random bodies in a zero-net-momentum frame.

## Installation

```bash
git clone <this-repo-url>
cd nbody-simulation
pip install -r requirements.txt
```

## Usage

Run a single simulation:

```bash
python main.py --scenario figure8 --integrator verlet --n-steps 20000 --dt 0.001
```

```
Running figure8 scenario with verlet integrator (3 bodies, 20000 steps, dt=0.001)...
Done in 1.324s. Final relative energy drift: 3.165e-09
Saved trajectories.png
```

Generate an animated GIF:

```bash
python main.py --scenario figure8 --integrator verlet --n-steps 20000 --dt 0.001 --animate
```

![Figure-eight trajectory](README_assets/trajectories_figure8.png)

Try the solar-system scenario (planets on near-circular orbits, full
mutual N-body gravity between all bodies, not just star-planet):

```bash
python main.py --scenario solar --n-planets 4 --seed 0 --integrator verlet --n-steps 15000 --dt 0.002 --animate
```

![Solar system orbits](README_assets/trajectories_solar.png)

Note the visible orbital precession — a real effect of planet-planet
perturbation that a fixed-central-force (two-body) model would miss
entirely.

> **On seeds and stability**: `solar_system_like` is a genuine
> N-body integration — every planet pulls on every other planet, not
> just on the central star. With some random seeds, those mutual
> perturbations compound over enough orbits into an actual close
> encounter between two planets, which is real orbital chaos, not a
> bug. Seed 0 (the default) is stable over the horizons used above;
> feel free to try others and watch what an unstable configuration
> looks like.

Compare all three integrators head-to-head on the same system:

```bash
python main.py --compare-integrators --scenario figure8 --n-steps 20000 --dt 0.001
```

```
Running euler...
Running verlet...
Running rk4...
euler   : final relative energy drift = 8.386e-02   (1.316s)
verlet  : final relative energy drift = 3.165e-09   (1.314s)
rk4     : final relative energy drift = 8.574e-14   (2.387s)
```

![Energy drift comparison](README_assets/energy_drift.png)

## The headline result: stability over long integrations

Pushed to **1,000,000 integration steps** on the figure-eight system
(`dt = 1e-3`, so 1000 simulation-time units — roughly 158 periods of the
underlying choreography):

```
euler   : final relative energy drift over 1,000,000 steps = 6.028e-01   (59.8s)
verlet  : final relative energy drift over 1,000,000 steps = 5.841e-07   (63.6s)
```

Euler's error compounds to **60% of the system's total energy** —
completely unphysical. Velocity Verlet, at essentially the same
computational cost, stays at **5.8 × 10⁻⁷** relative error six orders of
magnitude better, because its energy error is *bounded* by
construction rather than accumulating step after step.

(All numbers above are real measurements from this repository, run on
the machine that built it — reproduce them yourself with the commands
shown; they'll vary a little by hardware but the qualitative gap
between Euler and Verlet will not.)

## Running tests

```bash
pip install pytest
pytest tests/ -v
```

Covers: no self-interaction, Newton's third law symmetry, softening
preventing singularities, momentum conservation by construction,
circular-orbit initial speeds matching the analytic formula, short-
horizon agreement between all three integrators, and — the key
regression test — that Verlet's long-run energy drift stays
meaningfully smaller than Euler's on the same system.

## Design choices worth calling out in an interview

- **Softening length, not exact point-mass gravity**: without it, any
  two bodies that pass close together produce a force spike that no
  fixed time step can resolve, and the simulation blows up. This is a
  standard N-body technique, not a simplification unique to this repo.
- **Symplectic vs. non-symplectic is the real dividing line**, not
  "high order vs. low order." RK4 is *more accurate per step* than
  Verlet but still drifts over very long horizons because it isn't
  symplectic — Verlet's structural property (bounded energy error) is
  what actually matters for a simulation meant to run for a long time.
- **Fully vectorized force computation**: the `(N, N, D)` broadcast in
  `forces.py` is what makes it practical to push to 10⁶+ steps at all;
  a naive double loop over pairs would be orders of magnitude slower in
  Python.
- **Zero-net-momentum initial conditions** for the random cluster and
  analytically-correct circular-orbit velocities for the solar system
  scenario — physically sensible starting points rather than arbitrary
  ones, so any drift you see later is attributable to the integrator,
  not to a system that was unphysical from step zero.

## Possible extensions

- Adaptive time-stepping (e.g., reduce `dt` automatically when two
  bodies get close) to handle close encounters gracefully instead of
  needing a lucky seed.
- Barnes-Hut or Fast Multipole approximation to scale beyond O(N²)
  force evaluation for large N.
- 3D visualization with `mpl_toolkits.mplot3d` or an interactive
  WebGL viewer.
- A symplectic higher-order integrator (e.g., Forest-Ruth or Yoshida)
  for even better long-term accuracy at Verlet's stability class.
