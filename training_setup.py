import dependencies

# Local imports:
import consts
from env import Env
from randomizer import make_randomization_fn

# Library imports:
from brax.training.agents.ppo.train import train as train_ppo
from brax.training.agents.ppo.networks import make_ppo_networks
from brax.training.types import Metrics
import functools
import mujoco_playground
from mujoco_playground.config import locomotion_params
from textwrap import indent


# The main training function, `brax.training.agents.ppo.train.train`, is defined at
# <https://github.com/google/brax/blob/d59e4db582e98da1734c098aed7219271c940bda/brax/training/agents/ppo/train.py#L192>.
# It has two positional arguments: `environment: envs.Env` and `num_timesteps: int`.


def make_training_loop(env: Env):

    kwargs = locomotion_params.brax_ppo_config(consts.COMPARABLE_MUJOCO_PLAYGROUND_ENV)
    kwargs.num_timesteps = consts.TRAINING_STEPS
    kwargs.randomization_fn = mujoco_playground.registry.get_domain_randomizer(
        consts.COMPARABLE_MUJOCO_PLAYGROUND_ENV
    )  # make_randomization_fn(env)

    if "network_factory" in kwargs:
        network_factory = kwargs.network_factory
        del kwargs.network_factory
    else:
        network_factory = dict()
    kwargs.network_factory = functools.partial(make_ppo_networks, **network_factory)

    def progress_fn(num_steps: int, metrics: Metrics):
        clear_output(wait=True)

        times.append(datetime.now())
        x_data.append(num_steps)
        y_data.append(metrics["eval/episode_reward"])
        y_dataerr.append(metrics["eval/episode_reward_std"])

        plt.xlim([0, kwargs.num_timesteps * 1.25])
        plt.xlabel("# environment steps")
        plt.ylabel("reward per episode")
        plt.title(f"y={y_data[-1]:.3f}")
        plt.errorbar(x_data, y_data, yerr=y_dataerr, color="blue")

        display(plt.gcf())

    kwargs.progress_fn = progress_fn

    print("Using the following keyword arguments for the training loop:")
    print(indent(str(kwargs), "    "))

    return functools.partial(train_ppo, **kwargs)
