#!/usr/bin/env python3


# Based on the ROBOTIS op3 spec: <https://github.com/google-deepmind/mujoco_menagerie/blob/main/robotis_op3/op3.xml>


# Local imports:
import consts

# Library imports:
import xml.etree.ElementTree as XML


print("Generating an MJCF XML file...")


# Top-level declaration (structured like a Russian nesting doll):
mujoco = XML.Element("mujoco", model="Eye")

# Automatically limit movement to provided ranges:
XML.SubElement(mujoco, "compiler", autolimits="true")

# Graphics/screen/video display options:
visual = XML.SubElement(mujoco, "visual")

# 2560x1440 display:
XML.SubElement(
    visual,
    "global",
    offwidth="2560",
    offheight="1440",
    elevation="-20",
    azimuth="120",
)

# Not sure what this does yet, but I'd guess it does something to the camera:
XML.SubElement(visual, "map", force="0.1", zfar="16")

# Far-away haze:
XML.SubElement(visual, "rgba", haze="0.25 0.25 0.25 1")

# What point should the camera focus on?
XML.SubElement(mujoco, "statistic", center=f"0 0 {consts.LENGTH_KNEE_TO_FOOT}")

# Define ambient textures:
asset = XML.SubElement(mujoco, "asset")
XML.SubElement(
    asset,
    "texture",
    type="skybox",
    builtin="gradient",
    rgb1="0.5 0.5 0.5",
    rgb2="0 0 0",
    width="32",
    height="512",
)
XML.SubElement(
    asset,
    "texture",
    name="body",
    type="cube",
    builtin="flat",
    mark="cross",
    width="128",
    height="128",
    rgb1="0.6 0.6 0.6",
    rgb2="0.6 0.6 0.6",
    markrgb="1 1 1",
    random="0.01",
)
XML.SubElement(
    asset,
    "material",
    name="body",
    texture="body",
    texuniform="true",
    rgba="0.6 0.6 0.6 1",
)
XML.SubElement(
    asset,
    "texture",
    name="grid",
    type="2d",
    builtin="checker",
    width="512",
    height="512",
    rgb1="0.7 0.7 0.7",
    rgb2="0.8 0.8 0.8",
)
XML.SubElement(
    asset,
    "material",
    name="grid",
    texture="grid",
    texrepeat="1 1",
    texuniform="true",
    reflectance=".2",
)

# Defaults:
default = XML.SubElement(mujoco, "default")

# Defaults for geometries/`geom`s/physical parts:
XML.SubElement(
    default,
    "geom",
    type="capsule",
    condim="1",
    solref=".004 1",
    material="body",
    rgba="0 0 0 1",
    size=f"{consts.LEG_RADIUS}",
    contype="0",
    conaffinity="0",
)

# Defaults for joints:
XML.SubElement(
    default,
    "joint",
    damping=f"{consts.JOINT_DAMPING}",
    armature=f"{consts.JOINT_ARMATURE}",
    frictionloss=f"{consts.JOINT_FRICTION_LOSS}",
)

# Defaults for servos:
XML.SubElement(
    default,
    "position",
    kp=f"{consts.SERVO_KP}",
    forcerange=f"-{consts.SERVO_TORQUE_NM} {consts.SERVO_TORQUE_NM}",
)

# Nesting doll for everything in the world (that isn't metadata):
worldbody = XML.SubElement(mujoco, "worldbody")

contact = XML.SubElement(mujoco, "contact")

# Add the ground plane/floor:
XML.SubElement(
    worldbody,
    "geom",
    name="floor",
    size=f"0 0 {consts.inches(2)}",
    type="plane",
    material="grid",
    condim="3",
)

# XML.SubElement(
#     worldbody,
#     "light",
#     name="spotlight",
#     mode="targetbodycom",
#     target=consts.ROOT_BODY,
#     diffuse=".8 .8 .8",
#     specular="0.3 0.3 0.3",
#     pos="0 -6 4",
#     cutoff="30",
# )

# Light above:
XML.SubElement(
    worldbody,
    "light",
    name="top",
    pos=f"0 0 {8 * consts.SPHERE_RADIUS}",
    mode="trackcom",
)

# And an off-axis spotlight on the torso:
XML.SubElement(
    worldbody,
    "light",
    name="spotlight",
    mode="targetbodycom",
    target=consts.ROOT_BODY,
    diffuse=".8 .8 .8",
    specular="0.3 0.3 0.3",
    pos="0 -6 4",
    cutoff="30",
)

torso = XML.SubElement(
    worldbody,
    "body",
    name=consts.ROOT_BODY,
    pos=f"0 0 {1.5 * consts.LENGTH_KNEE_TO_FOOT}",
)
# XML.SubElement(torso, "light", name="top", pos="0 0 2", mode="trackcom")
XML.SubElement(
    torso, "camera", name="back", pos="-3 0 1", xyaxes="0 -1 0 1 0 2", mode="trackcom"
)
XML.SubElement(
    torso, "camera", name="side", pos="0 -3 1", xyaxes="1 0 0 0 1 2", mode="trackcom"
)
XML.SubElement(torso, "freejoint", name="root")

XML.SubElement(
    torso,
    "geom",
    name=consts.ROOT_BODY,
    type="sphere",
    size=f"{consts.SPHERE_RADIUS}",
    mass=f"{consts.SPHERE_MASS + 9 * consts.SERVO_MASS * (1.0 + consts.EXTRA_SPHERE_MASS_PERCENTAGE_IM_FORGETTING)}",
    rgba="1 1 1 1",
)
if consts.PUPIL_SIZE_RELATIVE is not None:
    XML.SubElement(
        torso,
        "geom",
        name="pupil",
        type="sphere",
        pos=f"0 {-(1 + consts.PUPIL_SIZE_PROTRUSION - consts.PUPIL_SIZE_RELATIVE) * consts.SPHERE_RADIUS} 0",
        size=f"{consts.PUPIL_SIZE_RELATIVE * consts.SPHERE_RADIUS}",
        mass="0",
        rgba="0 0 1 1",
    )
XML.SubElement(
    torso,
    "site",
    name="imu",
)
XML.SubElement(
    torso,
    "camera",
    name="egocentric",
    pos=f"{consts.SPHERE_RADIUS} 0 0",
    xyaxes="0 -1 0 .1 0 1",
    fovy="80",
)


contact = XML.SubElement(mujoco, "contact")


sensor = XML.SubElement(mujoco, "sensor")

XML.SubElement(sensor, "gyro", site="imu", name="gyro")
# XML.SubElement(sensor, "velocimeter", site="imu", name="local_linvel")
XML.SubElement(sensor, "accelerometer", site="imu", name="accelerometer")
# XML.SubElement(sensor, "framepos", objtype="site", objname="imu", name="position")
# XML.SubElement(sensor, "framezaxis", objtype="site", objname="imu", name="upvector")
# XML.SubElement(
#     sensor, "framexaxis", objtype="site", objname="imu", name="forwardvector"
# )
# XML.SubElement(
#     sensor, "framelinvel", objtype="site", objname="imu", name="global_linvel"
# )
# XML.SubElement(
#     sensor, "frameangvel", objtype="site", objname="imu", name="global_angvel"
# )
# XML.SubElement(sensor, "framequat", objtype="site", objname="imu", name="orientation")


actuator = XML.SubElement(mujoco, "actuator")


keyframe = XML.SubElement(mujoco, "keyframe")
XML.SubElement(keyframe, "key", name="home", qpos=" ".join(["0"] * 16))


for i in range(consts.N_LEGS):
    leg_mount = XML.SubElement(
        torso,
        "body",
        name=f"leg mount #{i + 1}",
        euler=f"0 0 {360.0 * i / consts.N_LEGS}",
    )
    leg = XML.SubElement(
        leg_mount,
        "body",
        name=f"leg #{i + 1}",
        pos=f"{consts.SPHERE_RADIUS} 0 0",
    )
    XML.SubElement(
        leg,
        "joint",
        name=f"leg yaw joint #{i + 1}",
        axis="0 0 1",
        range=f"{consts.LEG_YAW_MIN_DEGREES} {consts.LEG_YAW_MAX_DEGREES}",
    )
    XML.SubElement(
        leg,
        "joint",
        name=f"hip joint #{i + 1}",
        axis="0 1 0",
        range=f"{consts.HIP_MIN_DEGREES} {consts.HIP_MAX_DEGREES}",
    )
    XML.SubElement(
        leg,
        "geom",
        name=f"upper leg #{i + 1}",
        fromto=f"0 0 0 {consts.LENGTH_HIP_TO_KNEE} 0 0",
        mass=f"{consts.LEG_DENSITY * consts.LENGTH_HIP_TO_KNEE}",
    )
    lower_leg = XML.SubElement(
        leg,
        "body",
        name=f"lower leg #{i + 1}",
        euler=f"0 90 0",
        pos=f"{consts.LENGTH_HIP_TO_KNEE} 0 0",
    )
    # XML.SubElement(
    #     lower_leg,
    #     "geom",
    #     type="sphere",
    #     name=f"knee #{i + 1}",
    #     size=f"{consts.KNEE_RADIUS}",
    # )
    XML.SubElement(
        lower_leg,
        "joint",
        name=f"knee joint #{i + 1}",
        axis="0 1 0",
        range=f"{consts.KNEE_MIN_DEGREES} {consts.KNEE_MAX_DEGREES}",
    )
    XML.SubElement(
        lower_leg,
        "geom",
        name=f"lower leg #{i + 1}",
        fromto=f"0 0 0 {consts.LENGTH_KNEE_TO_FOOT} 0 0",
        mass=f"{consts.LEG_DENSITY * consts.LENGTH_KNEE_TO_FOOT}",
    )

    foot = XML.SubElement(
        lower_leg,
        "body",
        name=f"foot #{i + 1}",
        pos=f"{consts.LENGTH_KNEE_TO_FOOT} 0 0",
    )
    XML.SubElement(
        foot,
        "site",
        name=f"foot #{i + 1}",
    )
    XML.SubElement(
        foot,
        "geom",
        type="sphere",
        name=f"foot #{i + 1}",
        size=f"{consts.FOOT_RADIUS}",
        mass=f"{consts.FOOT_CAP_MASS}",
    )

    XML.SubElement(sensor, "force", site=f"foot #{i + 1}", name=f"foot force #{i + 1}")
    XML.SubElement(
        sensor,
        "framelinvel",
        objtype="site",
        objname=f"foot #{i + 1}",
        name=f"foot #{i + 1}_global_linvel",
    )
    XML.SubElement(
        sensor,
        "framelinvel",
        objtype="site",
        objname=f"foot #{i + 1}",
        name=f"foot #{i + 1}_pos",
        reftype="site",
        refname="imu",
    )

    XML.SubElement(
        actuator,
        "position",
        name=f"hip servo #{i + 1}",
        joint=f"hip joint #{i + 1}",
        ctrlrange=f"{consts.HIP_MIN_DEGREES} {consts.HIP_MAX_DEGREES}",
    )
    XML.SubElement(
        actuator,
        "position",
        name=f"knee servo #{i + 1}",
        joint=f"knee joint #{i + 1}",
        ctrlrange=f"{consts.KNEE_MIN_DEGREES} {consts.KNEE_MAX_DEGREES}",
    )
    XML.SubElement(
        actuator,
        "position",
        name=f"leg yaw servo #{i + 1}",
        joint=f"leg yaw joint #{i + 1}",
        ctrlrange=f"{consts.LEG_YAW_MIN_DEGREES} {consts.LEG_YAW_MAX_DEGREES}",
    )

    XML.SubElement(contact, "pair", geom1=f"foot #{i + 1}", geom2="floor")


XML.indent(mujoco)
with open(consts.GENERATED_MJCF_XML_PATH, "wb") as file:
    XML.ElementTree(mujoco).write(file, encoding="utf-8")
