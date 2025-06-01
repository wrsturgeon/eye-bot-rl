#!/usr/bin/env python3


# Based on the ROBOTIS op3 spec: <https://github.com/google-deepmind/mujoco_menagerie/blob/main/robotis_op3/op3.xml>


# Local imports:
import consts

# Library imports:
import xml.etree.ElementTree as XML


print("Generating MJCF XML files...")


# Top-level robot declaration:
robot = XML.Element("mujoco", model="eye-robot")


# Robot defaults:
default = XML.SubElement(robot, "default")

XML.SubElement(
    default,
    "geom",
    type="capsule",
    size=f"{consts.LEG_RADIUS}",
    solref=".004 1",
    contype="0",
    conaffinity="0",
)

joint_kwargs = dict()
if consts.JOINT_DAMPING is not None:
    joint_kwargs["damping"] = f"{consts.JOINT_DAMPING}"
if consts.JOINT_STIFFNESS is not None:
    joint_kwargs["stiffness"] = f"{consts.JOINT_STIFFNESS}"
if consts.JOINT_ARMATURE is not None:
    joint_kwargs["armature"] = f"{consts.JOINT_ARMATURE}"
if consts.JOINT_FRICTION_LOSS is not None:
    joint_kwargs["frictionloss"] = f"{consts.JOINT_FRICTION_LOSS}"
XML.SubElement(default, "joint", **joint_kwargs)
del joint_kwargs

XML.SubElement(
    default,
    "position",
    kp=f"{consts.SERVO_KP}",
    forcerange=f"{-consts.SERVO_TORQUE_NM} {consts.SERVO_TORQUE_NM}",
)


# Top-level declaration for physical objects:
worldbody = XML.SubElement(robot, "worldbody")

root = XML.SubElement(
    worldbody, "body", name=consts.ROOT_BODY, pos=f"0 0 {consts.LENGTH_KNEE_TO_FOOT}"
)

XML.SubElement(root, "freejoint")

sphere = XML.SubElement(root, "body", name="sphere")
XML.SubElement(
    sphere,
    "geom",
    name="sphere",
    type="sphere",
    size=f"{consts.SPHERE_RADIUS}",
    rgba="1 1 1 1",
    mass=f"{(consts.SPHERE_MASS + 9 * consts.SERVO_MASS) * (1.0 + consts.EXTRA_SPHERE_MASS_PERCENTAGE_IM_FORGETTING)}",
)
XML.SubElement(
    sphere,
    "site",
    name="imu",
)


contact = XML.SubElement(robot, "contact")
actuator = XML.SubElement(robot, "actuator")


sensor = XML.SubElement(robot, "sensor")
XML.SubElement(sensor, "gyro", site="imu", name="gyro")
XML.SubElement(sensor, "accelerometer", site="imu", name="accelerometer")


for i in range(0, consts.N_LEGS):
    leg_mount_to_center = XML.SubElement(
        root,
        "body",
        name=f"leg_{i}_mount_to_center",
        euler=f"0 0 {360 * i / consts.N_LEGS}",
    )
    leg_mount = XML.SubElement(
        leg_mount_to_center,
        "body",
        name=f"leg_{i}_mount",
        pos=f"{0.5 * consts.SPHERE_RADIUS} 0 0",
    )
    leg = XML.SubElement(leg_mount, "body", name=f"leg_{i}")
    XML.SubElement(leg, "joint", axis="0 1 0", name=f"leg_{i}_hip_joint")
    XML.SubElement(leg, "joint", axis="0 0 1", name=f"leg_{i}_yaw_joint")
    hip_to_knee = XML.SubElement(
        leg,
        "geom",
        name=f"leg_{i}_hip_to_knee",
        fromto=f"0 0 0 {consts.LENGTH_HIP_TO_KNEE} 0 0",
        mass=f"{consts.LEG_DENSITY * consts.LENGTH_HIP_TO_KNEE}",
        rgba="1 0 0 1",
    )
    lower_leg = XML.SubElement(
        leg,
        "body",
        name=f"leg_{i}_lower",
        pos=f"{consts.LENGTH_HIP_TO_KNEE} 0 0",
        euler="0 90 0",
    )
    XML.SubElement(lower_leg, "joint", axis="0 1 0", name=f"leg_{i}_knee_joint")
    knee_to_foot = XML.SubElement(
        lower_leg,
        "geom",
        name=f"leg_{i}_knee_to_foot",
        fromto=f"0 0 0 {consts.LENGTH_KNEE_TO_FOOT} 0 0",
        mass=f"{consts.LEG_DENSITY * consts.LENGTH_HIP_TO_KNEE}",
        rgba="1 0 0 1",
    )
    foot = XML.SubElement(
        lower_leg,
        "body",
        name=f"leg_{i}_foot",
        pos=f"{consts.LENGTH_KNEE_TO_FOOT} 0 0",
    )
    XML.SubElement(
        foot,
        "geom",
        type="sphere",
        size=f"{consts.FOOT_RADIUS}",
        name=f"leg_{i}_foot",
        mass=f"{consts.FOOT_CAP_MASS}",
        rgba="1 0 0 1",
    )
    XML.SubElement(
        foot,
        "site",
        name=f"leg_{i}_foot",
    )

    XML.SubElement(contact, "pair", geom1=f"leg_{i}_foot", geom2="floor")

    XML.SubElement(
        actuator, "position", name=f"leg_{i}_hip_joint", joint=f"leg_{i}_hip_joint"
    )
    XML.SubElement(
        actuator, "position", name=f"leg_{i}_knee_joint", joint=f"leg_{i}_knee_joint"
    )
    XML.SubElement(
        actuator, "position", name=f"leg_{i}_yaw_joint", joint=f"leg_{i}_yaw_joint"
    )

    XML.SubElement(sensor, "force", site=f"leg_{i}_foot", name=f"leg_{i}_foot_fsr")


# Save the MJCF MXL for the robot:
XML.indent(robot)
with open(consts.GENERATED_ROBOT_MJCF_XML_PATH, "wb") as file:
    XML.ElementTree(robot).write(file, encoding="utf-8")
del robot


# Top-level scene declaration:
scene = XML.Element("mujoco", model="eye-robot")


# Focus on the center of the body. TODO: what's `extent`?
XML.SubElement(
    scene, "statistic", center=f"0 0 {consts.LENGTH_KNEE_TO_FOOT}", extent="0.6"
)


# Rendering settings:
visual = XML.SubElement(scene, "visual")

# Headlight (light from the active camera):
XML.SubElement(
    visual, "headlight", diffuse="0.6 0.6 0.6", ambient="0.3 0.3 0.3", specular="0 0 0"
)

# Haze at the render limit:
XML.SubElement(visual, "rgba", haze="0.15 0.25 0.35 1")

# Global camera orientation:
XML.SubElement(visual, "global", azimuth="160", elevation="-20")


# Textures & materials:
asset = XML.SubElement(scene, "asset")

# Sky texture:
XML.SubElement(
    asset,
    "texture",
    type="skybox",
    builtin="gradient",
    rgb1="0.3 0.5 0.7",
    rgb2="0 0 0",
    width="512",
    height="3072",
)

# Ground plane/floor grid texture:
XML.SubElement(
    asset,
    "texture",
    type="2d",
    name="groundplane",
    builtin="checker",
    mark="edge",
    rgb1="0.2 0.3 0.4",
    rgb2="0.1 0.2 0.3",
    markrgb="0.8 0.8 0.8",
    width="300",
    height="300",
)

# Ground plane/floor material:
XML.SubElement(
    asset,
    "material",
    name="groundplane",
    texture="groundplane",
    texuniform="true",
    texrepeat="5 5",
    reflectance="0.2",
)


# Then use the textures & materials we just defined on physical things:
worldbody = XML.SubElement(scene, "worldbody")

# Sunlight:
XML.SubElement(worldbody, "light", pos="0 0 1.5", dir="0 0 -1", directional="true")

# Sunlight:
XML.SubElement(
    worldbody,
    "geom",
    name="floor",
    pos="0 0 -0.05",
    size="0 0 0.05",
    type="plane",
    material="groundplane",
)


# Save the MJCF MXL for the scene:
XML.indent(scene)
with open(consts.GENERATED_SCENE_MJCF_XML_PATH, "wb") as file:
    XML.ElementTree(scene).write(file, encoding="utf-8")
del scene


# Top-level declaration for the combined model:
combined = XML.Element("mujoco", model="eye-robot")
XML.SubElement(combined, "include", file=consts.GENERATED_SCENE_MJCF_XML_PATH)
XML.SubElement(combined, "include", file=consts.GENERATED_ROBOT_MJCF_XML_PATH)


# Save the MJCF MXL for the combined model:
XML.indent(combined)
with open(consts.GENERATED_MJCF_XML_PATH, "wb") as file:
    XML.ElementTree(combined).write(file, encoding="utf-8")
del combined
