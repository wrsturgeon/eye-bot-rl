#!/usr/bin/env python3

# Local imports:
import envs

# Library imports:
import brax
from brax.training.agents.ppo import networks as ppo_networks, train as ppo
from datetime import datetime
from etils import epath
from flax.training import orbax_utils
import functools
from orbax import checkpoint as ocp


def progress(num_steps, metrics):
    times.append(datetime.now())
    x_data.append(num_steps)
    y_data.append(metrics["eval/episode_reward"])
    ydataerr.append(metrics["eval/episode_reward_std"])

    plt.xlim([0, train_fn.keywords["num_timesteps"] * 1.25])
    plt.ylim([min_y, max_y])

    plt.xlabel("# environment steps")
    plt.ylabel("reward per episode")
    plt.title(f"y={y_data[-1]:.3f}")

    plt.errorbar(x_data, y_data, yerr=ydataerr)
    plt.show()


checkpoint_path = epath.Path("tmp/quadrupred_joystick/checkpoints")
checkpoint_path.mkdir(parents=True, exist_ok=True)


def policy_params_fn(current_step, make_policy, params):
    # save checkpoints
    orbax_checkpointer = ocp.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(params)
    path = checkpoint_path / f"{current_step}"
    orbax_checkpointer.save(path, params, force=True, save_args=save_args)


make_networks_factory = functools.partial(
    ppo_networks.make_ppo_networks, policy_hidden_layer_sizes=(128, 128, 128, 128)
)
train_fn = functools.partial(
    ppo.train,
    num_timesteps=100_000_000,
    num_evals=10,
    reward_scaling=1,
    episode_length=1000,
    normalize_observations=True,
    action_repeat=1,
    unroll_length=20,
    num_minibatches=32,
    num_updates_per_batch=4,
    discounting=0.97,
    learning_rate=3.0e-4,
    entropy_cost=1e-2,
    num_envs=8192,
    batch_size=256,
    network_factory=make_networks_factory,
    randomization_fn=envs.domain_randomize,
    policy_params_fn=policy_params_fn,
    seed=0,
)

x_data = []
y_data = []
ydataerr = []
times = [datetime.now()]
max_y, min_y = 40, 0

# Reset environments since internals may be overwritten by tracers from the
# domain randomization function.
env = brax.envs.get_environment("eye")
eval_env = brax.envs.get_environment("eye")
make_inference_fn, params, _ = train_fn(
    environment=env, progress_fn=progress, eval_env=eval_env
)

print(f"time to jit: {times[1] - times[0]}")
print(f"time to train: {times[-1] - times[1]}")
