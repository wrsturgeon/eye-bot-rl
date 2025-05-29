#!/usr/bin/env python3

SCREEN_WIDTH = 2560
SCREEN_HEIGHT = 1440
ENABLE_MJX = True

import cv2
from jax import jit, vmap, random as jrnd
import mujoco as mj
from mujoco import mjx
import numpy as np
import time

if ENABLE_MJX:
    step = jit(mjx.step)
    get_data = mjx.get_data
else:

    def step(mj_model, mj_data):
        mj.mj_step(mj_model, mj_data)
        return mj_data

    get_data = lambda mj_model, mj_data: mj_data

np.set_printoptions(precision=3, suppress=True, linewidth=100)

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
if ENABLE_MJX:
    mjx_model = mjx.put_model(mj_model)
    mjx_data = mjx.put_data(mj_model, mj_data)
else:
    mjx_model = mj_model
    mjx_data = mj_data

duration = 5.0  # seconds
fps = 60  # actual is 500
video_writer = cv2.VideoWriter(
    "render.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    renderer.render().shape[:2][::-1],
)
next_frame = 0.0

while mjx_data.time < duration:
    mjx_data = step(mjx_model, mjx_data)
    if mjx_data.time > next_frame:
        next_frame += 1.0 / fps
        mj_data = get_data(mj_model, mjx_data)
        renderer.update_scene(mj_data, scene_option=scene_option)
        frame = renderer.render()
        video_writer.write(frame)
video_writer.release()

# rng = jrnd.PRNGKey(42)
# rng = jrnd.split(rng, 4096)
# batch = vmap(lambda rng: mjx_data.replace(qpos=jrnd.uniform(rng, (1,))))(rng)
#
# jit_step = jit(vmap(mjx.step, in_axes=(None, 0)))
# batch = jit_step(mjx_model, batch)
#
# print(batch.qpos)
