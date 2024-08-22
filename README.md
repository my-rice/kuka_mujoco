# Kuka Mujoco Plotter

## Overview

The Kuka Mujoco is a Python-based tool designed to visualize and analyze data generated from simulations using the Mujoco physics engine. This tool is particularly useful for robotics researchers and engineers working with the Kuka robotic arm. The $trajectory_interopolator.py$ module allows users to generate smooth trajectories for the Kuka arm, while the $logger.py$ module enables users to visualize and analyze the generated data.
## Features
- **Trajectory Interpolator**: Generate smooth trajectories for the Kuka arm using cubic splines or different strategies. The trajectories can be customized by specifying the desired joint positions, velocities, and accelerations at the start and end of the trajectory. The generated trajecotries will be followed by the Kuka arm in the simulation, thanks to the Mujoco physics engine inversion (`mj_inverse(model,data)`) feature.
- **Data Visualization**: Plot joint positions (`qpos`), velocities (`qvel`), accelerations (`qacc`), and control signals (`ctrl`) over time. It also save plots as PNG files automatically for easy sharing and documentation.

## Installation

To install the necessary dependencies, you can use `pip`:

```bash
pip install -r requirements.txt