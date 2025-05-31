import dependencies

print("Defining constants...")

# Library imports:
from etils import epath

# from mujoco_playground._src import mjx_env

N_LEGS = 3
COMPARABLE_MUJOCO_PLAYGROUND_ENV = "Go1JoystickRoughTerrain"
GENERATED_MJCF_XML_PATH = "generated-mjcf.xml"
TRAINING_STEPS = 1_000_000

# ROOT_PATH = mjx_env.ROOT_PATH / "locomotion" / "go1"
# FEET_ONLY_FLAT_TERRAIN_XML = ROOT_PATH / "xmls" / "scene_mjx_feetonly_flat_terrain.xml"
# FEET_ONLY_ROUGH_TERRAIN_XML = (
#     ROOT_PATH / "xmls" / "scene_mjx_feetonly_rough_terrain.xml"
# )
# FULL_FLAT_TERRAIN_XML = ROOT_PATH / "xmls" / "scene_mjx_flat_terrain.xml"
# FULL_COLLISIONS_FLAT_TERRAIN_XML = (
#     ROOT_PATH / "xmls" / "scene_mjx_fullcollisions_flat_terrain.xml"
# )


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


def inches(murica):
    return murica * 0.0254


def grams(g):
    return g * 0.001


def kg_cm(torque):
    return torque * 0.0980665


EYE_RADIUS = inches(2.0)
UPPER_LEG_LENGTH = inches(2.0)
LOWER_LEG_LENGTH = inches(4.0)

HIP_MIN_DEGREES = -90
HIP_MAX_DEGREES = 90
KNEE_MIN_DEGREES = -60
KNEE_MAX_DEGREES = 60
LEG_YAW_MIN_DEGREES = -30
LEG_YAW_MAX_DEGREES = 30

PUSH_ROD_SPACING = inches(0.5)
LEG_DIAMETER = inches(0.05)
FOOT_DIAMETER = inches(0.25)
KNEE_DIAMETER = LEG_DIAMETER * 1.5

LEG_DENSITY = grams(1.0) / inches(1.0)
SPHERE_MASS = grams(10.0)
SERVO_MASS = grams(19.0)
FOOT_CAP_MASS = grams(1.0)
EXTRA_SPHERE_MASS_PERCENTAGE_IM_FORGETTING = 0.1

PUPIL_SIZE_RELATIVE = None  # 0.75
PUPIL_SIZE_PROTRUSION = 0.05

SERVO_TORQUE_NM = 10 * kg_cm(2.7)
