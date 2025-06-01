#!/usr/bin/env python3


################################################################################################################################


# Based on [https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/mjx/tutorial.ipynb](this Brax tutorial).


################################################################################################################################


import dependencies


################################################################################################################################


print("Local imports...")
import consts

print("Library imports...")
from brax import envs
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf, model
from brax.training.agents.ppo import train as ppo
import cv2
from datetime import datetime
import functools
import jax
from jax import numpy as jp
import mujoco
from mujoco import mjx


################################################################################################################################


print("Building a MuJoCo model...")
mj_model = mujoco.MjModel.from_xml_path(consts.GENERATED_MJCF_XML_PATH)
mj_data = mujoco.MjData(mj_model)
renderer = mujoco.Renderer(mj_model)


################################################################################################################################


print("Porting model to MJX...")
mjx_model = mjx.put_model(mj_model)
mjx_data = mjx.put_data(mj_model, mj_data)


################################################################################################################################


# enable joint visualization option:
print("Visualing with MuJoCo...")
scene_option = mujoco.MjvOption()
scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True

duration = 3.8  # (seconds)
framerate = 60  # (Hz)

video_filename = "mj.mp4"
frames = []
video_writer = cv2.VideoWriter(
    video_filename,
    cv2.VideoWriter_fourcc(*"mp4v"),
    framerate,
    renderer.render().shape[:2][::-1],
)
mujoco.mj_resetData(mj_model, mj_data)
while mj_data.time < duration:
    mujoco.mj_step(mj_model, mj_data)
    if len(frames) < mj_data.time * framerate:
        renderer.update_scene(mj_data, scene_option=scene_option)
        pixels = renderer.render()
        frames.append(pixels)
        video_writer.write(pixels[..., ::-1])
video_writer.release()

# Simulate and display video.
try:
    media.show_video(frames, fps=framerate)
except:
    print("`media` failed. If you're not in a Jupyter notebook, this is expected.")
del frames


################################################################################################################################


print("Visualing with MJX...")
jit_step = jax.jit(mjx.step)

video_filename = "mjx.mp4"
frames = []
video_writer = cv2.VideoWriter(
    video_filename,
    cv2.VideoWriter_fourcc(*"mp4v"),
    framerate,
    renderer.render().shape[:2][::-1],
)
mujoco.mj_resetData(mj_model, mj_data)
mjx_data = mjx.put_data(mj_model, mj_data)
while mjx_data.time < duration:
    mjx_data = jit_step(mjx_model, mjx_data)
    if len(frames) < mjx_data.time * framerate:
        mj_data = mjx.get_data(mj_model, mjx_data)
        renderer.update_scene(mj_data, scene_option=scene_option)
        pixels = renderer.render()
        frames.append(pixels)
        video_writer.write(pixels[..., ::-1])
video_writer.release()

try:
    media.show_video(frames, fps=framerate)
except:
    print("`media` failed. If you're not in a Jupyter notebook, this is expected.")
del frames


################################################################################################################################


print("Randomizing across batches...")
rng = jax.random.PRNGKey(0)
rng = jax.random.split(rng, 4096)
batch = jax.vmap(
    lambda rng: mjx_data.replace(qpos=jax.random.uniform(rng, (mjx_model.nq,)))
)(rng)

jit_step = jax.jit(jax.vmap(mjx.step, in_axes=(None, 0)))
batch = jit_step(mjx_model, batch)


################################################################################################################################


print("Porting randomized batches to MJX...")
batched_mj_data = mjx.get_data(mj_model, batch)


################################################################################################################################


print("Defining a MuJoCo/Brax pipeline environment...")


class Humanoid(PipelineEnv):

    def __init__(
        self,
        forward_reward_weight=1.25,
        ctrl_cost_weight=0.1,
        healthy_reward=5.0,
        terminate_when_unhealthy=True,
        reset_noise_scale=1e-2,
        exclude_current_positions_from_observation=True,
        **kwargs,
    ):
        #
        mj_model = mujoco.MjModel.from_xml_path(consts.GENERATED_MJCF_XML_PATH)
        mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG
        mj_model.opt.iterations = 6
        mj_model.opt.ls_iterations = 6

        sys = mjcf.load_model(mj_model)

        physics_steps_per_control_step = 5
        kwargs["n_frames"] = kwargs.get("n_frames", physics_steps_per_control_step)
        kwargs["backend"] = "mjx"

        super().__init__(sys, **kwargs)

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

    def reset(self, rng: jp.ndarray) -> State:
        """Resets the environment to an initial state."""
        rng, rng1, rng2 = jax.random.split(rng, 3)

        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        qpos = self.sys.qpos0 + jax.random.uniform(
            rng1, (self.sys.nq,), minval=low, maxval=hi
        )
        qvel = jax.random.uniform(rng2, (self.sys.nv,), minval=low, maxval=hi)

        data = self.pipeline_init(qpos, qvel)

        obs = self._get_obs(data, jp.zeros(self.sys.nu))
        reward, done, zero = jp.zeros(3)
        metrics = {
            "forward_reward": zero,
            "reward_linvel": zero,
            "reward_quadctrl": zero,
            "reward_alive": zero,
            "x_position": zero,
            "y_position": zero,
            "distance_from_origin": zero,
            "x_velocity": zero,
            "y_velocity": zero,
        }
        return State(data, obs, reward, done, metrics)

    def step(self, state: State, action: jp.ndarray) -> State:
        """Runs one timestep of the environment's dynamics."""
        data0 = state.pipeline_state
        data = self.pipeline_step(data0, action)

        com_before = data0.subtree_com[1]
        com_after = data.subtree_com[1]
        velocity = (com_after - com_before) / self.dt
        forward_reward = self._forward_reward_weight * velocity[0]

        is_healthy = jp.where(data.q[2] < consts.SPHERE_RADIUS, 0.0, 1.0)
        if self._terminate_when_unhealthy:
            healthy_reward = self._healthy_reward
        else:
            healthy_reward = self._healthy_reward * is_healthy

        ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(action))

        obs = self._get_obs(data, action)
        reward = forward_reward + healthy_reward - ctrl_cost
        done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0
        state.metrics.update(
            forward_reward=forward_reward,
            reward_linvel=forward_reward,
            reward_quadctrl=-ctrl_cost,
            reward_alive=healthy_reward,
            x_position=com_after[0],
            y_position=com_after[1],
            distance_from_origin=jp.linalg.norm(com_after),
            x_velocity=velocity[0],
            y_velocity=velocity[1],
        )

        return state.replace(pipeline_state=data, obs=obs, reward=reward, done=done)

    def _get_obs(self, data: mjx.Data, action: jp.ndarray) -> jp.ndarray:
        """Observes humanoid body position, velocities, and angles."""
        position = data.qpos
        if self._exclude_current_positions_from_observation:
            position = position[2:]

        # external_contact_forces are excluded
        return jp.concatenate(
            [
                position,
                data.qvel,
                data.cinert[1:].ravel(),
                data.cvel[1:].ravel(),
                data.qfrc_actuator,
            ]
        )


print("Registering that environment...")
env_name = "humanoid"
envs.register_environment(env_name, Humanoid)


################################################################################################################################


# instantiate the environment
print("Instantiating that environment...")
env = envs.get_environment(env_name)

# define the jit reset/step functions
print("Defining (but not yet compiling) JIT-compiled `reset` and `swap`...")
jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)


################################################################################################################################


# initialize the state
print("Initializing state...")
state = jit_reset(jax.random.PRNGKey(0))
rollout = [state.pipeline_state]

# grab a trajectory
print("Computing a trajectory...")
for i in range(10):
    ctrl = -0.1 * jp.ones(env.sys.nu)
    state = jit_step(state, ctrl)
    rollout.append(state.pipeline_state)

print("Visualizing that trajectory...")
trajectory_visualized = env.render(rollout)
framerate = int(1.0 / env.dt)
try:
    media.show_video(trajectory_visualized, fps=framerate)
except:
    print("`media` failed. If you're not in a Jupyter notebook, this is expected.")
video_filename = "trajectory.mp4"
video_writer = cv2.VideoWriter(
    video_filename,
    cv2.VideoWriter_fourcc(*"mp4v"),
    framerate,
    trajectory_visualized[0].shape[:2][::-1],
)
for pixels in trajectory_visualized:
    video_writer.write(pixels[..., ::-1])
video_writer.release()
del trajectory_visualized


################################################################################################################################


print("Setting up a training loop...")
train_fn = functools.partial(
    ppo.train,
    num_timesteps=20_000_000,
    num_evals=5,
    reward_scaling=0.1,
    episode_length=1000,
    normalize_observations=True,
    action_repeat=1,
    unroll_length=10,
    num_minibatches=24,
    num_updates_per_batch=8,
    discounting=0.97,
    learning_rate=3e-4,
    entropy_cost=1e-3,
    num_envs=3072,
    batch_size=512,
    seed=0,
)


x_data = []
y_data = []
ydataerr = []
start_time = datetime.now()
jit_time = None

max_y, min_y = 13000, 0


def progress(num_steps, metrics):
    global jit_time
    global x_data
    global y_data
    global y_dataerr

    if jit_time is None:
        jit_time = datetime.now()
        print(f"Time to JIT: {jit_time - start_time}")

    x_data.append(num_steps)
    y_data.append(metrics["eval/episode_reward"])
    ydataerr.append(metrics["eval/episode_reward_std"])

    try:
        plt.xlim([0, train_fn.keywords["num_timesteps"] * 1.25])
        plt.ylim([min_y, max_y])

        plt.xlabel("# environment steps")
        plt.ylabel("reward per episode")
        plt.title(f"y={y_data[-1]:.3f}")

        plt.errorbar(x_data, y_data, yerr=ydataerr)
        plt.show()
    except:
        print("`plt` failed. If you're not in a Jupyter notebook, this is expected.")


print("Training...")
make_inference_fn, params, _ = train_fn(environment=env, progress_fn=progress)

print(f"Time to train (after JIT-compiling): {datetime.now() - jit_time}")


################################################################################################################################


model_path = "./mjx_brax_policy"
model.save_params(model_path, params)


################################################################################################################################


params = model.load_params(model_path)

inference_fn = make_inference_fn(params)
jit_inference_fn = jax.jit(inference_fn)


################################################################################################################################


eval_env = envs.get_environment(env_name)

jit_reset = jax.jit(eval_env.reset)
jit_step = jax.jit(eval_env.step)


################################################################################################################################


# initialize the state
rng = jax.random.PRNGKey(0)
state = jit_reset(rng)
rollout = [state.pipeline_state]

# grab a trajectory
n_steps = 500
render_every = 2

for i in range(n_steps):
    act_rng, rng = jax.random.split(rng)
    ctrl, _ = jit_inference_fn(state.obs, act_rng)
    state = jit_step(state, ctrl)
    rollout.append(state.pipeline_state)

    if state.done:
        break

try:
    media.show_video(
        env.render(rollout[::render_every]),
        fps=1.0 / env.dt / render_every,
    )
except:
    print("`media` failed. If you're not in a Jupyter notebook, this is expected.")


################################################################################################################################


mj_model = eval_env.sys.mj_model
mj_data = mujoco.MjData(mj_model)

renderer = mujoco.Renderer(mj_model)
ctrl = jp.zeros(mj_model.nu)

images = []
for i in range(n_steps):
    act_rng, rng = jax.random.split(rng)

    obs = eval_env._get_obs(mjx.put_data(mj_model, mj_data), ctrl)
    ctrl, _ = jit_inference_fn(obs, act_rng)

    mj_data.ctrl = ctrl
    for _ in range(eval_env._n_frames):
        mujoco.mj_step(mj_model, mj_data)  # Physics step using MuJoCo mj_step.

    if i % render_every == 0:
        renderer.update_scene(mj_data)
        images.append(renderer.render())

try:
    media.show_video(images, fps=1.0 / eval_env.dt / render_every)
except:
    print("`media` failed. If you're not in a Jupyter notebook, this is expected.")
