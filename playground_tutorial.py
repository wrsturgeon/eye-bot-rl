#!/usr/bin/env python3

# Based on <https://colab.research.google.com/github/google-deepmind/mujoco_playground/blob/main/learning/notebooks/locomotion.ipynb>.


# Check dependencies and emit helpful error messages if any are missing:
import dependencies


# Local imports:
import consts
from env import Env
from joystick import Joystick
from training_setup import make_training_loop

# Library imports:
from datetime import datetime
import mujoco_playground
from mujoco_playground import wrapper


# The main training function, `brax.training.agents.ppo.train.train`, is defined at
# <https://github.com/google/brax/blob/d59e4db582e98da1734c098aed7219271c940bda/brax/training/agents/ppo/train.py#L192>.
# It has two positional arguments: `environment: envs.Env` and `num_timesteps: int`.


print(
    f"Importing MuJoCo Playground's `{consts.COMPARABLE_MUJOCO_PLAYGROUND_ENV}` environment configuration..."
)
env_cfg = mujoco_playground.registry.get_default_config(
    consts.COMPARABLE_MUJOCO_PLAYGROUND_ENV
)


print(f"Configuring our custom environment...")


def make_env() -> Joystick:
    return mujoco_playground.registry.load(
        consts.COMPARABLE_MUJOCO_PLAYGROUND_ENV, config=env_cfg
    )
    # return Joystick(task="rough_terrain", config=env_cfg)


env = make_env()


# x_data, y_data, y_err = [], [], []
# time = datetime.now()
# iteration_counter = 1
#
#
# def progress(num_steps, metrics):
#     global iteration_counter
#     global time
#
#     clear_output(wait=True)
#
#     if iteration_counter == 1:
#         now = datetime.now()
#         print(f"Time to JIT: {now - time}")
#         time = now
#     iteration_counter += 1
#     print(f"Iteration #{iteration_counter}/{consts.TRAINING_STEPS}")
#
#     x_data.append(num_steps)
#     y_data.append(metrics["eval/episode_reward"])
#     y_err.append(metrics["eval/episode_reward_std"])
#
#     # plt.xlim([0, consts.TRAINING_STEPS * 1.25])
#     # plt.xlabel("# environment steps")
#     # plt.ylabel("reward per episode")
#     # plt.title(f"y={y_data[-1]:.3f}")
#     # plt.errorbar(x_data, y_data, yerr=y_err, color="blue")
#     #
#     # display(plt.gcf())


x_data, y_data, y_dataerr = [], [], []
times = [datetime.now()]


def progress(num_steps, metrics):
    clear_output(wait=True)

    times.append(datetime.now())
    x_data.append(num_steps)
    y_data.append(metrics["eval/episode_reward"])
    y_dataerr.append(metrics["eval/episode_reward_std"])

    plt.xlim([0, ppo_params["num_timesteps"] * 1.25])
    plt.xlabel("# environment steps")
    plt.ylabel("reward per episode")
    plt.title(f"y={y_data[-1]:.3f}")
    plt.errorbar(x_data, y_data, yerr=y_dataerr, color="blue")

    display(plt.gcf())


train = make_training_loop(env)


make_inference_fn, params, metrics = train(
    environment=env,
    eval_env=mujoco_playground.registry.load(
        consts.COMPARABLE_MUJOCO_PLAYGROUND_ENV, config=env_cfg
    ),  # make_env(),
    wrap_env_fn=wrapper.wrap_for_brax_training,
)


print(f"Time to train (after JIT): {datatime.now() - time}")
