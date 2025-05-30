#!/usr/bin/env python3


import dependencies

# Local imports:
import consts

# Library imports:
from brax.training.agents.ppo import networks as ppo_networks, train as ppo
from datetime import datetime
import functools
from mujoco_playground import registry, wrapper

# from IPython.display import clear_output


########################################


env_name = "Go1JoystickFlatTerrain"
env = registry.load(env_name)
env_cfg = registry.get_default_config(env_name)


########################################


from mujoco_playground.config import locomotion_params

ppo_params = locomotion_params.brax_ppo_config(env_name)
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
iteration_counter = 1


def progress(num_steps, metrics):
    global iteration_counter
    global time

    # clear_output(wait=True)

    if iteration_counter == 1:
        now = datetime.now()
        print(f"Time to JIT: {now - time}")
        time = now
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


# randomizer = registry.get_domain_randomizer(env_name)
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


make_inference_fn, params, metrics = train_fn(
    environment=env,
    eval_env=registry.load(env_name, config=env_cfg),
    wrap_env_fn=wrapper.wrap_for_brax_training,
)
print(f"time to jit: {times[1] - times[0]}")
print(f"time to train: {times[-1] - times[1]}")
