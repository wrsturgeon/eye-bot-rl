#!/usr/bin/env python3

from sys import stderr

print("Importing NumPy...")
try:
    import numpy as np

    np.set_printoptions(precision=3, suppress=True, linewidth=100)
except ModuleNotFoundError:
    print(
        "ERROR: NumPy is not installed. It's installable via `pip` as `numpy`.",
        file=stderr,
    )
    exit(1)

print("Importing JAX...")
try:
    import jax

    jax.numpy.set_printoptions(precision=3, suppress=True, linewidth=100)
except ModuleNotFoundError:
    print(
        "ERROR: JAX is not installed. It's installable at `https://docs.jax.dev/en/latest/quickstart.html`.",
        file=stderr,
    )
    exit(1)

print("Importing MuJoCo...")
try:
    import mujoco
except ModuleNotFoundError:
    print(
        "ERROR: MuJoCo is not installed. It's installable via `pip` as `mujoco`.",
        file=stderr,
    )
    exit(1)

print("Importing MJX...")
try:
    from mujoco import mjx
except ModuleNotFoundError:
    print(
        "ERROR: MJX (JAX rewrite of MuJoCo) is not installed. It's installable via `pip` as `mujoco_mjx`.",
        file=stderr,
    )
    exit(1)

print("Importing Brax...")
try:
    import brax
except ModuleNotFoundError:
    print(
        "ERROR: Brax is not installed. It's installable via `pip` as `brax`.",
        file=stderr,
    )
    exit(1)

# print("Importing MuJoCo Playground...")
# try:
#     import mujoco_playground
# except ModuleNotFoundError:
#     print(
#         "ERROR: MuJoCo Playground is not installed. It's installable via `pip` as `playground`.",
#         file=stderr,
#     )
#     exit(1)

print("Importing MediaPy...")
try:
    import mediapy as media
except ModuleNotFoundError:
    print(
        "ERROR: MediaPy is not installed. It's installable via `pip` as `mediapy`.",
        file=stderr,
    )
    # exit(1)

print("Importing MatPlotLib...")
try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    print(
        "ERROR: MatPlotLib is not installed. It's installable via `pip` as `matplotlib`.",
        file=stderr,
    )
    # exit(1)

import model
