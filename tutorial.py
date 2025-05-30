#!/usr/bin/env python3


N_ENVS = 10

SCREEN_WIDTH = 2560
SCREEN_HEIGHT = 1440


# Local imports:
import envs

# Library imports:
import brax

# import cv2
from jax import jit, vmap, random as jrnd, tree_util as jtree
import mujoco as mj
from mujoco import mjx
import numpy as np

# import time


np.set_printoptions(precision=3, suppress=True, linewidth=100)


rng = jrnd.PRNGKey(0)


rng = jrnd.split(rng, N_ENVS)
env = brax.envs.get_environment("eye")
batched_sys, _ = envs.domain_randomize(env.sys, rng)


mj_model = mj.MjModel.from_xml_path("generated-mjcf.xml")
mj_model.opt.solver = mj.mjtSolver.mjSOL_CG
mj_model.opt.iterations = 6
mj_model.opt.ls_iterations = 6

mj_data = mj.MjData(mj_model)
renderer = mj.Renderer(mj_model, SCREEN_HEIGHT, SCREEN_WIDTH)

# enable joint visualization option:
scene_option = mj.MjvOption()
scene_option.flags[mj.mjtVisFlag.mjVIS_JOINT] = True

mj.mj_resetData(mj_model, mj_data)
mjx_model = mjx.put_model(mj_model)
mjx_data = mjx.put_data(mj_model, mj_data)

rng = jrnd.PRNGKey(42)
rng = jrnd.split(rng, 4096)
batch = vmap(lambda rng: mjx_data.replace(qpos=jrnd.uniform(rng, (1,))))(rng)

jit_step = jit(vmap(mjx.step, in_axes=(None, 0)))
batch = jit_step(mjx_model, batch)

print(batch.qpos)
