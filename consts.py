print("Defining constants...")

# Library imports:
from etils import epath

# from mujoco_playground._src import mjx_env

N_LEGS = 3
COMPARABLE_MUJOCO_PLAYGROUND_ENV = "Go1JoystickRoughTerrain"
GENERATED_ROBOT_MJCF_XML_PATH = "generated-robot-mjcf.xml"
GENERATED_SCENE_MJCF_XML_PATH = "generated-scene-mjcf.xml"
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


SPHERE_RADIUS = inches(2.0)
LENGTH_HIP_TO_KNEE = inches(2.0)
LENGTH_KNEE_TO_FOOT = inches(4.0)

HIP_MIN_DEGREES = -90
HIP_MAX_DEGREES = 90
KNEE_MIN_DEGREES = -60
KNEE_MAX_DEGREES = 60
LEG_YAW_MIN_DEGREES = -30
LEG_YAW_MAX_DEGREES = 30

PUSH_ROD_SPACING = inches(0.5)
LEG_RADIUS = inches(0.1)
FOOT_RADIUS = inches(0.125)
KNEE_RADIUS = LEG_RADIUS * 1.5

LEG_DENSITY = grams(1.0) / inches(1.0)
SPHERE_MASS = grams(10.0)
SERVO_MASS = grams(19.0)
FOOT_CAP_MASS = grams(1.0)
EXTRA_SPHERE_MASS_PERCENTAGE_IM_FORGETTING = 0.1

PUPIL_SIZE_RELATIVE = None  # 0.75
PUPIL_SIZE_PROTRUSION = 0.05

SERVO_TORQUE_NM = 5  # kg_cm(2.7)
SERVO_KP = 21.1  # from <https://github.com/google-deepmind/mujoco/issues/1075>: see line <https://github.com/google-deepmind/mujoco_menagerie/blob/cfd91c5605e90f0b77860ae2278ff107366acc87/robotis_op3/op3.xml#L62>

JOINT_DAMPING = 1.084
JOINT_STIFFNESS = None
JOINT_ARMATURE = 0.045
JOINT_FRICTION_LOSS = 0.03
