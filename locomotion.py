#!/usr/bin/env python3


import dependencies

# Local imports:
import consts
from joystick import Joystick

# Library imports:
from brax.training.agents.ppo import networks as ppo_networks, train as ppo
from copy import deepcopy
from datetime import datetime
import functools
from mujoco_playground import registry, wrapper
from mujoco_playground.config import locomotion_params

# from IPython.display import clear_output


########################################


# env_name = "Go1JoystickFlatTerrain"
# env = registry.load(env_name)
# env_cfg = registry.get_default_config(env_name)


env_cfg = registry.get_default_config(consts.COMPARABLE_MUJOCO_PLAYGROUND_ENV)
# env = registry.load(consts.COMPARABLE_MUJOCO_PLAYGROUND_ENV, config=env_cfg)
env = Joystick(task="rough_terrain", config=env_cfg)


########################################


ppo_params = locomotion_params.brax_ppo_config(consts.COMPARABLE_MUJOCO_PLAYGROUND_ENV)
ppo_params.num_timesteps = ppo_params.num_timesteps // 100_000_000
ppo_params


########################################


# x_data, y_data, y_dataerr = [], [], []
# times = [datetime.now()]
#
#
# def progress(num_steps, metrics):
#     # clear_output(wait=True)
#
#     times.append(datetime.now())
#     x_data.append(num_steps)
#     y_data.append(metrics["eval/episode_reward"])
#     y_dataerr.append(metrics["eval/episode_reward_std"])
#
#     # plt.xlim([0, ppo_params["num_timesteps"] * 1.25])
#     # plt.xlabel("# environment steps")
#     # plt.ylabel("reward per episode")
#     # plt.title(f"y={y_data[-1]:.3f}")
#     # plt.errorbar(x_data, y_data, yerr=y_dataerr, color="blue")
#     #
#     # display(plt.gcf())


x_data, y_data, y_err = [], [], []
time = datetime.now()
iteration_counter = 0


def progress(num_steps, metrics):
    global iteration_counter
    global time

    # clear_output(wait=True)

    if iteration_counter == 0:
        now = datetime.now()
        print(f"Time to JIT: {now - time}")
        time = now
        exit()
    iteration_counter += 1
    print(f"Iteration #{iteration_counter}/{consts.TRAINING_STEPS}")

    x_data.append(num_steps)
    y_data.append(metrics["eval/episode_reward"])
    y_err.append(metrics["eval/episode_reward_std"])

    # plt.xlim([0, consts.TRAINING_STEPS * 1.25])
    # plt.xlabel("# environment steps")
    # plt.ylabel("reward per episode")
    # plt.title(f"y={y_data[-1]:.3f}")
    # plt.errorbar(x_data, y_data, yerr=y_err, color="blue")
    #
    # display(plt.gcf())


# randomizer = registry.get_domain_randomizer(consts.COMPARABLE_MUJOCO_PLAYGROUND_ENV)
from randomizer import make_randomization_fn

randomizer = make_randomization_fn(env)


ppo_training_params = dict(ppo_params)
network_factory = ppo_networks.make_ppo_networks
if "network_factory" in ppo_params:
    del ppo_training_params["network_factory"]
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks, **ppo_params.network_factory
    )

train_fn = functools.partial(
    ppo.train,
    **dict(ppo_training_params),
    network_factory=network_factory,
    randomization_fn=randomizer,
    progress_fn=progress,
)


########################################


eval_env = deepcopy(env)
make_inference_fn, params, metrics = train_fn(
    environment=env,
    eval_env=eval_env,
    wrap_env_fn=wrapper.wrap_for_brax_training,
)
print(f"Time to train: {datetime.now() - time}")


trajectory = [TODO for TODO in TODO]
print(eval_env.render(trajectory))
# import cv2
# import mujoco as mj
#
# mj_data = env._mj_data
# mj_model = env._mj_model
#
# duration = 5.0  # seconds
# fps = 60  # actual is 500
# renderer = mj.Renderer(mj_model, SCREEN_HEIGHT, SCREEN_WIDTH)
# video_writer = cv2.VideoWriter(
#     "render.mp4",
#     cv2.VideoWriter_fourcc(*"mp4v"),
#     fps,
#     renderer.render().shape[:2][::-1],
# )
# next_frame = 0.0
#
# while mj_data.time < duration:
#     mj_data = step(mj_model, mj_data)
#     if mj_data.time > next_frame:
#         next_frame += 1.0 / fps
#         mj_data = get_data(mj_model, mj_data)
#         renderer.update_scene(mj_data, scene_option=scene_option)
#         frame = renderer.render()
#         video_writer.write(frame)
# video_writer.release()
