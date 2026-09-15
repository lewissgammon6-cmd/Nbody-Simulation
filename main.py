#!/usr/bin/env python3
"""
main.py
-------
Command-line entry point for the N-body gravitational simulation.

Examples
--------
    python main.py --scenario figure8 --integrator verlet --n-steps 20000 --dt 0.001
    python main.py --scenario solar --n-planets 5 --integrator rk4 --animate
    python main.py --compare-integrators --scenario figure8
"""

from __future__ import annotations

import argparse
import sys

from src.bodies import SCENARIOS
from src.simulation import run_simulation
from src.visualize import animate_system, plot_energy_drift, plot_trajectories


def build_system(args: argparse.Namespace):
    if args.scenario == "figure8":
        return SCENARIOS["figure8"]()
    if args.scenario == "solar":
        return SCENARIOS["solar"](n_planets=args.n_planets, central_mass=args.central_mass, seed=args.seed)
    if args.scenario == "random":
        return SCENARIOS["random"](n_bodies=args.n_bodies, radius=args.radius, seed=args.seed)
    raise ValueError(args.scenario)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="N-Body Gravitational Simulation")
    p.add_argument("--scenario", default="figure8", choices=list(SCENARIOS))
    p.add_argument("--integrator", default="verlet", choices=["euler", "verlet", "rk4"])
    p.add_argument("--dt", type=float, default=1e-3)
    p.add_argument("--n-steps", type=int, default=20_000)
    p.add_argument("--softening", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=7)

    # scenario-specific
    p.add_argument("--n-planets", type=int, default=4)
    p.add_argument("--central-mass", type=float, default=1000.0)
    p.add_argument("--n-bodies", type=int, default=20)
    p.add_argument("--radius", type=float, default=5.0)

    p.add_argument("--plot-out", default="trajectories.png")
    p.add_argument("--animate", action="store_true")
    p.add_argument("--anim-out", default="orbit.gif")
    p.add_argument("--anim-stride", type=int, default=10)

    p.add_argument("--compare-integrators", action="store_true",
                    help="Run euler/verlet/rk4 on the same system and plot relative energy drift")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    system = build_system(args)

    if args.compare_integrators:
        results = {}
        for name in ["euler", "verlet", "rk4"]:
            print(f"Running {name}...")
            results[name] = run_simulation(
                system, integrator=name, dt=args.dt, n_steps=args.n_steps, softening=args.softening
            )
        for name, result in results.items():
            final_drift = result.relative_energy_drift[-1]
            print(f"{name:8s}: final relative energy drift = {final_drift:.3e}   ({result.elapsed_seconds:.3f}s)")
        plot_energy_drift(results, "energy_drift.png")
        print("Saved energy_drift.png")
        return 0

    print(f"Running {args.scenario} scenario with {args.integrator} integrator "
          f"({system.n_bodies} bodies, {args.n_steps} steps, dt={args.dt})...")
    result = run_simulation(
        system, integrator=args.integrator, dt=args.dt, n_steps=args.n_steps, softening=args.softening
    )
    final_drift = result.relative_energy_drift[-1]
    print(f"Done in {result.elapsed_seconds:.3f}s. Final relative energy drift: {final_drift:.3e}")

    plot_trajectories(system, result, args.plot_out)
    print(f"Saved {args.plot_out}")

    if args.animate:
        animate_system(system, result, args.anim_out, stride=args.anim_stride)
        print(f"Saved {args.anim_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
