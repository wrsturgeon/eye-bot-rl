import dependencies

# Library imports:
from etils import epath
from mujoco_playground._src import mjx_env

N_LEGS = 3
COMPARABLE_MUJOCO_PLAYGROUND_ENV = "Go1JoystickRoughTerrain"
TRAINING_STEPS = 1_000_000

ROOT_PATH = mjx_env.ROOT_PATH / "locomotion" / "go1"
FEET_ONLY_FLAT_TERRAIN_XML = ROOT_PATH / "xmls" / "scene_mjx_feetonly_flat_terrain.xml"
FEET_ONLY_ROUGH_TERRAIN_XML = (
    ROOT_PATH / "xmls" / "scene_mjx_feetonly_rough_terrain.xml"
)
FULL_FLAT_TERRAIN_XML = ROOT_PATH / "xmls" / "scene_mjx_flat_terrain.xml"
FULL_COLLISIONS_FLAT_TERRAIN_XML = (
    ROOT_PATH / "xmls" / "scene_mjx_fullcollisions_flat_terrain.xml"
)


def task_to_xml(task_name: str) -> epath.Path:
    return {
        "flat_terrain": FEET_ONLY_FLAT_TERRAIN_XML,
        "rough_terrain": FEET_ONLY_ROUGH_TERRAIN_XML,
    }[task_name]


FEET_SITES = [f"foot #{i + 1}" for i in range(0, N_LEGS)]

FEET_GEOMS = FEET_SITES

FEET_POS_SENSOR = [f"{site} pos" for site in FEET_SITES]


ROOT_BODY = "trunk"  # "torso"

UPVECTOR_SENSOR = "upvector"
GLOBAL_LINVEL_SENSOR = "global_linvel"
GLOBAL_ANGVEL_SENSOR = "global_angvel"
LOCAL_LINVEL_SENSOR = "local_linvel"
ACCELEROMETER_SENSOR = "accelerometer"
GYRO_SENSOR = "gyro"
