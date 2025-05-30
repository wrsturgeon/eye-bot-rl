import dependencies

# Local imports:
import consts

# Library imports:
import jax
from jax import numpy as jp
from ml_collections.config_dict import ConfigDict
from mujoco import mjx, MjModel
from mujoco_playground._src import mjx_env
from typing import Any, Dict, Optional, Union

# We're mirroring the functionality of these two commands:
#   - `env_cfg = mujoco_playground.registry.get_default_config(ENV)`
#   - `env = mujoco_playground.registry.load(ENV, config=env_cfg)`
# `mujoco_playground.registry.load` lives at <https://github.com/google-deepmind/mujoco_playground/blob/ff0ea5629bd89662f6ffa54464e247653737ea45/mujoco_playground/_src/registry.py#L49>
# then calls `locomotion.load` with the same arguments, at <https://github.com/google-deepmind/mujoco_playground/blob/ff0ea5629bd89662f6ffa54464e247653737ea45/mujoco_playground/_src/locomotion/__init__.py#L168>
# that returns `_envs[env_name](config=env_cfg)`
# `_envs` is defined at <https://github.com/google-deepmind/mujoco_playground/blob/ff0ea5629bd89662f6ffa54464e247653737ea45/mujoco_playground/_src/locomotion/__init__.py#L47>
# `_envs["Go1JoystickRoughTerrain"]` is `functools.partial(go1_joystick.Joystick, task="rough_terrain")`
# `Joystick` is defined at <https://github.com/google-deepmind/mujoco_playground/blob/ff0ea5629bd89662f6ffa54464e247653737ea45/mujoco_playground/_src/locomotion/go1/joystick.py#L96>
# `Joystick.__init__` calls `go1_base.Go1Env.__init__(xml_path=consts.task_to_xml(task).as_posix(), config=env_cfg)` then `Joystick._post_init()` then returns
# `go1_base.Go1Env.__init__` is defined at <https://github.com/google-deepmind/mujoco_playground/blob/ff0ea5629bd89662f6ffa54464e247653737ea45/mujoco_playground/_src/locomotion/go1/base.py#L43>
# ^^^ it's longer, so let's use bullet-points:
#   - first calls `mjx_env.MjxEnv.__init__(env_cfg)`, defined at <https://github.com/google-deepmind/mujoco_playground/blob/ff0ea5629bd89662f6ffa54464e247653737ea45/mujoco_playground/_src/mjx_env.py#L213>
#       - entire body is `self._config = env_cfg.lock(); self._ctrl_dt = env_cfg.ctrl_dt; self._sim_dt = env_cfg.sim_dt`
#   - then sets up an `MjModel`:
#       - `self._mj_model = mujoco.MjModel.from_xml_string(epath.Path(xml_path).read_text(), assets=get_assets())`
#       - `self._mj_model.opt.timestep = self._config.sim_dt`
#
#       - `# Modify PD gains.`
#       - `self._mj_model.dof_damping[6:] = config.Kd`
#       - `self._mj_model.actuator_gainprm[:, 0] = config.Kp`
#       - `self._mj_model.actuator_biasprm[:, 1] = -config.Kp`
#
#       - `# Increase offscreen framebuffer size to render at higher resolutions.`
#       - `self._mj_model.vis.global_.offwidth = 3840`
#       - `self._mj_model.vis.global_.offheight = 2160`
#   - ports it to MJX: `self._mjx_model = mjx.put_model(self._mj_model)`
#   - and caches the IMU site ID: `self._imu_site_id = self._mj_model.site("imu").id`
# last call of the constructor is `_post_init`, which does a lot, defined at <https://github.com/google-deepmind/mujoco_playground/blob/ff0ea5629bd89662f6ffa54464e247653737ea45/mujoco_playground/_src/locomotion/go1/joystick.py#L112>:
#   - make a JAX NumPy array of the model's home position:
#       - `self._init_q = jp.array(self._mj_model.keyframe("home").qpos)`
#   - I guess the first seven positions are somehow irrelevant?
#       - `self._default_pose = jp.array(self._mj_model.keyframe("home").qpos[7:])`
#   - split all joints into lower and upper (somehow):
#       - `# Note: First joint is freejoint.`
#       - `self._lowers, self._uppers = self.mj_model.jnt_range[1:].T`
#       - `self._soft_lowers = self._lowers * self._config.soft_joint_pos_limit_factor`
#       - `self._soft_uppers = self._uppers * self._config.soft_joint_pos_limit_factor`
#   - find various parts (torso, feet, floor):
#       - `self._torso_body_id = self._mj_model.body(consts.ROOT_BODY).id`
#       - `self._torso_mass = self._mj_model.body_subtreemass[self._torso_body_id]`
#       - `self._feet_site_id = np.array([self._mj_model.site(name).id for name in consts.FEET_SITES])`
#       - `self._floor_geom_id = self._mj_model.geom("floor").id`
#       - `self._feet_geom_id = np.array([self._mj_model.geom(name).id for name in consts.FEET_GEOMS])`
#   - find all the sensors for the linear velocity of the feet:
#       - `foot_linvel_sensor_adr = []`
#       - `for site in consts.FEET_SITES:`
#       - `  sensor_id = self._mj_model.sensor(f"{site}_global_linvel").id`
#       - `  sensor_adr = self._mj_model.sensor_adr[sensor_id]`
#       - `  sensor_dim = self._mj_model.sensor_dim[sensor_id]`
#       - `  foot_linvel_sensor_adr.append(`
#       - `      list(range(sensor_adr, sensor_adr + sensor_dim))`
#       - `  )`
#       - `self._foot_linvel_sensor_adr = jp.array(foot_linvel_sensor_adr)`
#   - somehow separate commands:
#       - `self._cmd_a = jp.array(self._config.command_config.a)`
#       - `self._cmd_b = jp.array(self._config.command_config.b)`


def get_assets() -> Dict[str, bytes]:
    assets = {}
    mjx_env.update_assets(assets, consts.ROOT_PATH / "xmls", "*.xml")
    mjx_env.update_assets(assets, consts.ROOT_PATH / "xmls" / "assets")
    path = mjx_env.MENAGERIE_PATH / "unitree_go1"
    mjx_env.update_assets(assets, path, "*.xml")
    mjx_env.update_assets(assets, path / "assets")
    return assets


# Largely a copy of `Go1Env` at <https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/_src/locomotion/go1/base.py#L40>:
class Env(mjx_env.MjxEnv):
    def __init__(
        self,
        config: ConfigDict,
    ) -> None:
        super().__init__(config)

        self._mj_model = MjModel.from_xml_path(
            "generated-mjcf.xml", assets=get_assets()
        )
        self._mj_model.opt.timestep = self._config.sim_dt

        # # Modify PD gains.
        # self._mj_model.dof_damping[6:] = config.Kd
        # self._mj_model.actuator_gainprm[:, 0] = config.Kp
        # self._mj_model.actuator_biasprm[:, 1] = -config.Kp

        # # Increase offscreen framebuffer size to render at higher resolutions.
        # self._mj_model.vis.global_.offwidth = 3840
        # self._mj_model.vis.global_.offheight = 2160

        self._mjx_model = mjx.put_model(self._mj_model)
        self._imu_site_id = self._mj_model.site("imu").id

    @property
    def xml_path(self) -> str:
        return "/go/fuck/yourself.xml"

    @property
    def action_size(self) -> int:
        return self._mjx_model.nu

    @property
    def mj_model(self) -> MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model

    def get_gyro(self, data: mjx.Data) -> jax.Array:
        return mjx_env.get_sensor_data(self.mj_model, data, consts.GYRO_SENSOR)

    def get_gravity(self, data: mjx.Data) -> jax.Array:
        return data.site_xmat[self._imu_site_id].T @ jp.array([0, 0, -1])

    def get_local_linvel(self, data: mjx.Data) -> jax.Array:
        return mjx_env.get_sensor_data(self.mj_model, data, consts.LOCAL_LINVEL_SENSOR)

    def get_accelerometer(self, data: mjx.Data) -> jax.Array:
        return mjx_env.get_sensor_data(self.mj_model, data, consts.ACCELEROMETER_SENSOR)

    def get_global_angvel(self, data: mjx.Data) -> jax.Array:
        return mjx_env.get_sensor_data(self.mj_model, data, consts.GLOBAL_ANGVEL_SENSOR)
