import dependencies

# Local imports:
import consts
from env import Env

# Library imports:
import jax
from mujoco import mjx


# We're trying to emulate `mujoco_playground.registry.get_domain_randomizer(ENV)`.
# `get_domain_randomizer` defined at <https://github.com/google-deepmind/mujoco_playground/blob/ff0ea5629bd89662f6ffa54464e247653737ea45/mujoco_playground/_src/registry.py#L64>
# immediately calls `locomotion.get_domain_randomizer` with the same arguments, at <https://github.com/google-deepmind/mujoco_playground/blob/ff0ea5629bd89662f6ffa54464e247653737ea45/mujoco_playground/_src/locomotion/__init__.py#L192>
# looks up `mujoco_playground._src.locomotion.go1.randomize.domain_randomize`, at <https://github.com/google-deepmind/mujoco_playground/blob/ff0ea5629bd89662f6ffa54464e247653737ea45/mujoco_playground/_src/locomotion/go1/randomize.py#L24>


IGNORE_FIRST_N_FRICTIONLOSS = 6
IGNORE_FIRST_N_ARMATURE = 6
IGNORE_FIRST_N_QPOS0 = 7


def make_randomization_fn(env: Env):
    floor_geom_id = env._mj_model.geom("floor").id
    torso_body_id = env._mj_model.body(consts.ROOT_BODY).id

    def randomization_fn(
        model: mjx.Model,
        rng: jax.Array,
    ):

        @jax.vmap
        def rand_dynamics(rng):
            # Floor friction: =U(0.4, 1.0).
            rng, key = jax.random.split(rng)
            geom_friction = model.geom_friction.at[floor_geom_id, 0].set(
                jax.random.uniform(key, minval=0.4, maxval=1.0)
            )

            # Scale static friction: *U(0.9, 1.1).
            rng, key = jax.random.split(rng)
            dof_frictionloss = model.dof_frictionloss[IGNORE_FIRST_N_FRICTIONLOSS:]
            frictionloss = dof_frictionloss * jax.random.uniform(
                key, shape=dof_frictionloss.shape, minval=0.9, maxval=1.1
            )
            dof_frictionloss = model.dof_frictionloss.at[
                IGNORE_FIRST_N_FRICTIONLOSS:
            ].set(frictionloss)

            # Scale armature: *U(1.0, 1.05).
            rng, key = jax.random.split(rng)
            dof_armature = model.dof_armature[IGNORE_FIRST_N_ARMATURE:]
            armature = dof_armature * jax.random.uniform(
                key, shape=dof_armature.shape, minval=1.0, maxval=1.05
            )
            dof_armature = model.dof_armature.at[IGNORE_FIRST_N_ARMATURE:].set(armature)

            # Jitter center of mass positiion: +U(-0.05, 0.05).
            rng, key = jax.random.split(rng)
            dpos = jax.random.uniform(key, (3,), minval=-0.05, maxval=0.05)
            body_ipos = model.body_ipos.at[torso_body_id].set(
                model.body_ipos[torso_body_id] + dpos
            )

            # Scale all link masses: *U(0.9, 1.1).
            rng, key = jax.random.split(rng)
            dmass = jax.random.uniform(
                key, shape=(model.nbody,), minval=0.9, maxval=1.1
            )
            body_mass = model.body_mass.at[:].set(model.body_mass * dmass)

            # Add mass to torso: +U(-1.0, 1.0).
            rng, key = jax.random.split(rng)
            dmass = jax.random.uniform(key, minval=-1.0, maxval=1.0)
            body_mass = body_mass.at[torso_body_id].set(
                body_mass[torso_body_id] + dmass
            )

            # Jitter qpos0: +U(-0.05, 0.05).
            rng, key = jax.random.split(rng)
            qpos0 = model.qpos0
            qpos0 = qpos0.at[IGNORE_FIRST_N_QPOS0:].set(
                qpos0[IGNORE_FIRST_N_QPOS0:]
                + jax.random.uniform(
                    key,
                    shape=qpos0[IGNORE_FIRST_N_QPOS0:].shape,
                    minval=-0.05,
                    maxval=0.05,
                )
            )

            return (
                geom_friction,
                body_ipos,
                body_mass,
                qpos0,
                dof_frictionloss,
                dof_armature,
            )

        (
            friction,
            body_ipos,
            body_mass,
            qpos0,
            dof_frictionloss,
            dof_armature,
        ) = rand_dynamics(rng)

        in_axes = jax.tree_util.tree_map(lambda x: None, model)
        in_axes = in_axes.tree_replace(
            {
                "geom_friction": 0,
                "body_ipos": 0,
                "body_mass": 0,
                "qpos0": 0,
                "dof_frictionloss": 0,
                "dof_armature": 0,
            }
        )

        model = model.tree_replace(
            {
                "geom_friction": friction,
                "body_ipos": body_ipos,
                "body_mass": body_mass,
                "qpos0": qpos0,
                "dof_frictionloss": dof_frictionloss,
                "dof_armature": dof_armature,
            }
        )

        return model, in_axes

    return randomization_fn
