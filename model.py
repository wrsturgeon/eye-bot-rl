#!/usr/bin/env python3


# Written to parameterize the model and output XML inspired by this handwritten XML spec:
# <https://github.com/google-deepmind/mujoco/blob/main/model/humanoid/humanoid.xml>


# Local imports:
import consts

# Library imports:
import xml.etree.ElementTree as XML


GLOBAL_MULTIPLIER = 1

EYE_RADIUS_INCHES = 2.0
UPPER_LEG_LENGTH_INCHES = 2.0
LOWER_LEG_LENGTH_INCHES = 4.0
HIP_MIN_DEGREES = -90
HIP_MAX_DEGREES = 90
KNEE_MIN_DEGREES = -60
KNEE_MAX_DEGREES = 60
PUSH_ROD_SPACING_INCHES = 0.5
LEG_DIAMETER_INCHES = 0.125
LEG_WEIGHT_GRAMS = 10.0
FOOT_DIAMETER_INCHES = 0.25
KNEE_DIAMETER_INCHES = LEG_DIAMETER_INCHES * 1.5
PUPIL_SIZE_RELATIVE = None  # 0.75
PUPIL_SIZE_PROTRUSION = 0.05
RED_GROUND = False


def inches(murica):
    return murica * 0.0254 * GLOBAL_MULTIPLIER


def gram(g):
    return g * 0.001 * GLOBAL_MULTIPLIER


# Top-level declaration (structured like a Russian nesting doll):
mujoco = XML.Element("mujoco", model="Eye")

# Simulation refresh rate:
XML.SubElement(
    mujoco, "option", timestep="0.001", gravity="0 0 -9.80665", iterations="100"
)

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
XML.SubElement(mujoco, "statistic", center=f"0 0 {inches(3 * EYE_RADIUS_INCHES)}")

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

# Not sure what this does yet:
XML.SubElement(default, "motor", ctrlrange="-1 1", ctrllimited="true")

# Defaults for all bodies:
body_default = XML.SubElement(default, "default", **{"class": "body"})

# Defaults for all physical parts of the body:
XML.SubElement(
    body_default,
    "geom",
    type="capsule",
    condim="1",
    friction="0.8 0.2 0.1",
    solimp=".9 .95 .001",
    solref=".005 1",
    # material="body",
    group="1",
    rgba="0 0 0 1",
    size=f"{inches(LEG_DIAMETER_INCHES)}",
    mass=f"{gram(LEG_WEIGHT_GRAMS)}",
)

# Defaults for all joints in the body:
XML.SubElement(
    body_default,
    "joint",
    type="hinge",
    damping="0.0005",
    stiffness="0.001",
    limited="true",
    # armature=".01",
    # solimplimit="0 .99 .01",
)

# Nesting doll for everything in the world (that isn't metadata):
worldbody = XML.SubElement(mujoco, "worldbody")

# Add the ground plane/floor:
XML.SubElement(
    worldbody,
    "geom",
    name="floor",
    size=f"0 0 {inches(2)}",
    type="plane",
    material="grid",
    condim="3",
)

# Light above:
XML.SubElement(
    worldbody,
    "light",
    name="top",
    pos=f"0 0 {inches(8 * EYE_RADIUS_INCHES)}",
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
    pos=f"0 0 {inches(4 * EYE_RADIUS_INCHES)}",
    childclass="body",
)
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
    name="eyeball",
    type="sphere",
    size=f"{inches(EYE_RADIUS_INCHES)}",
    rgba="1 1 1 1",
)
if PUPIL_SIZE_RELATIVE is not None:
    XML.SubElement(
        torso,
        "geom",
        name="pupil",
        type="sphere",
        pos=f"0 {inches(-(1 + PUPIL_SIZE_PROTRUSION - PUPIL_SIZE_RELATIVE) * EYE_RADIUS_INCHES)} 0",
        size=f"{inches(PUPIL_SIZE_RELATIVE * EYE_RADIUS_INCHES)}",
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
    pos=f"{inches(EYE_RADIUS_INCHES)} 0 0",
    xyaxes="0 -1 0 .1 0 1",
    fovy="80",
)


contact = XML.SubElement(mujoco, "contact")


sensor = XML.SubElement(mujoco, "sensor")

XML.SubElement(sensor, "gyro", site="imu", name="gyro")
XML.SubElement(sensor, "velocimeter", site="imu", name="local_linvel")
XML.SubElement(sensor, "accelerometer", site="imu", name="accelerometer")
XML.SubElement(sensor, "framepos", objtype="site", objname="imu", name="position")
XML.SubElement(sensor, "framezaxis", objtype="site", objname="imu", name="upvector")
XML.SubElement(
    sensor, "framexaxis", objtype="site", objname="imu", name="forwardvector"
)
XML.SubElement(
    sensor, "framelinvel", objtype="site", objname="imu", name="global_linvel"
)
XML.SubElement(
    sensor, "frameangvel", objtype="site", objname="imu", name="global_angvel"
)
XML.SubElement(sensor, "framequat", objtype="site", objname="imu", name="orientation")


actuator = XML.SubElement(mujoco, "actuator")


keyframe = XML.SubElement(mujoco, "keyframe")
XML.SubElement(keyframe, "key", name="home", qpos=" ".join(["0"] * 13))


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
        pos=f"{inches(EYE_RADIUS_INCHES)} 0 0",
    )
    XML.SubElement(
        leg,
        "joint",
        name=f"hip joint #{i + 1}",
        axis="0 1 0",
        range=f"{HIP_MIN_DEGREES} {HIP_MAX_DEGREES}",
    )
    XML.SubElement(
        leg,
        "geom",
        name=f"upper leg #{i + 1}",
        fromto=f"0 0 0 {inches(UPPER_LEG_LENGTH_INCHES)} 0 0",
    )
    lower_leg = XML.SubElement(
        leg,
        "body",
        name=f"lower leg #{i + 1}",
        euler=f"0 90 0",
        pos=f"{inches(UPPER_LEG_LENGTH_INCHES)} 0 0",
    )
    XML.SubElement(
        lower_leg,
        "geom",
        type="sphere",
        name=f"knee #{i + 1}",
        size=f"{inches(KNEE_DIAMETER_INCHES)}",
    )
    XML.SubElement(
        lower_leg,
        "joint",
        name=f"knee joint #{i + 1}",
        axis="0 1 0",
        range=f"{KNEE_MIN_DEGREES} {KNEE_MAX_DEGREES}",
    )
    XML.SubElement(
        lower_leg,
        "geom",
        name=f"lower leg #{i + 1}",
        fromto=f"0 0 0 {inches(LOWER_LEG_LENGTH_INCHES)} 0 0",
    )

    foot = XML.SubElement(
        lower_leg,
        "body",
        name=f"foot #{i + 1}",
        pos=f"{inches(LOWER_LEG_LENGTH_INCHES)} 0 0",
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
        size=f"{inches(FOOT_DIAMETER_INCHES)}",
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
    )
    XML.SubElement(
        actuator,
        "position",
        name=f"knee servo #{i + 1}",
        joint=f"knee joint #{i + 1}",
    )


XML.indent(mujoco)
with open("generated-mjcf.xml", "wb") as file:
    XML.ElementTree(mujoco).write(file, encoding="utf-8")
