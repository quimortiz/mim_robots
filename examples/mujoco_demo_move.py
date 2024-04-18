import mujoco
import mujoco.viewer
import time
from mim_robots.robot_loader import (
    load_mujoco_model,
    get_robot_list,
    load_pinocchio_wrapper,
)
from pinocchio.visualize import MeshcatVisualizer
import pinocchio as pin
import sys
import numpy as np
import cv2
import pathlib
from scipy.spatial.transform import Rotation as RR
import yaml
import pickle
from aruco_utils import *
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import sys
import copy

import argparse

from scipy.spatial.transform import Slerp



def add_visual_capsule(scene, point1, point2, radius, rgba):
  """Adds one capsule to an mjvScene."""
  if scene.ngeom >= scene.maxgeom:
    return
  scene.ngeom += 1  # increment ngeom\n",
  # initialise a new capsule, add it to the scene using mjv_makeConnector\n",
  mujoco.mjv_initGeom(scene.geoms[scene.ngeom-1],
                      mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                      np.zeros(3), np.zeros(9), rgba.astype(np.float32))
  mujoco.mjv_makeConnector(scene.geoms[scene.ngeom-1],
                           mujoco.mjtGeom.mjGEOM_CAPSULE, radius,
                           point1[0], point1[1], point1[2],
                           point2[0], point2[1], point2[2]),



def view_image(image):
    cv2.imshow(f"tmp", image)
    while True:
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyWindow("tmp")


argp = argparse.ArgumentParser()

argp.add_argument("--callibrate_intrinsics", action="store_true")
argp.add_argument("--callibrate_extrinsics", action="store_true")
argp.add_argument("--record_movement", action="store_true")

args = argp.parse_args()

model = load_mujoco_model("iiwa_pusher")
data = mujoco.MjData(model)

mujoco.mj_forward(model, data)

geoms = [model.geom(i).name for i in range(model.ngeom)]
bodies = [model.body(i).name for i in range(model.nbody)]
print("geoms", geoms)
print("bodies", bodies)

for geom in geoms:
    if geom != "":
        print(f"name {geom} pos {data.geom(geom).xpos}")

for body in bodies:
    if body != "":
        print(f"name {body} pos {data.body(body).xpos}")


joints = ["A1", "A2", "A3", "A4", "A5", "A6", "A7"]
# pos="0 0 0" axis="0 0 1" range="-2.96706 2.96706"/>

q0 = {"A1": 0, "A2": 0.5, "A3": 0, "A4": -1.5, "A5": 0, "A6": 1.5, "A7": 0}
# q0 = { "A1":0, "A2":0, "A3":0, "A4":0, "A5":0, "A6":  0, "A7":0 }

for k, v in q0.items():
    data.joint(k).qpos[0] = v
    # data.qpos[model.joint_name2id(k)] = v

mujoco.mj_forward(model, data)

# init

position_pusher_tip_mujoco = data.geom("pusher_tip").xpos
position_pusher_mujoco = data.geom("pusher_stick").xpos
postion_pusher_tip_goal = data.geom("pusher_tip_goal").xpos

r_pusher_tip_mujoco = data.geom("pusher_tip").xmat.reshape(3,3)
r_pusher_tip_goal = data.geom("pusher_tip_goal").xmat.reshape(3,3)


viewer = mujoco.viewer.launch_passive(model, data)

while viewer.is_running():
    viewer.sync()
    time.sleep(0.01)
viewer.close()


pusher_tip = "pusher_tip"
pusher_tip_goal = "pusher_tip_goal"


x_pusher_tip = data.geom(pusher_tip).xpos
x_pusher_tip_goal = data.geom(pusher_tip_goal).xpos

r_pusher_tip = data.geom(pusher_tip).xmat.reshape(3,3)
r_pusher_tip_goal = data.geom(pusher_tip_goal).xmat.reshape(3,3)

# convert to quaternions
q_pusher_tip = RR.from_matrix(r_pusher_tip).as_quat()
q_pusher_tip_goal = RR.from_matrix(r_pusher_tip_goal).as_quat()






num_steps = 100

max_sim_time = 10  # in seconds
max_T = max_sim_time

slerp = Slerp([0, max_T] , RR.from_quat([q_pusher_tip, q_pusher_tip_goal]))
              
times = np.linspace(0, max_T, num_steps)

rotation_path = slerp(times)


# plot rotation path?








nx = len(x_pusher_tip)

path = np.zeros((100, nx))
for i in range(nx):
    path[:, i] = np.interp(times, [0, max_T], [x_pusher_tip[i], x_pusher_tip_goal[i]])


fig = plt.figure()
ax = plt.axes(projection="3d")
one_every = 5
for ii in range(0, len(rotation_path), one_every):
    R = rotation_path[ii].as_matrix()
    x = path[ii]
    print("x", x)
    print("R", R)
    draw_3d_axis_from_R(ax, x, R, length=.05)

plt.show()


displacement = np.diff(path, axis=0)
displacement = np.vstack([displacement, displacement[-1, :]])


desired_vel = displacement / (max_T / num_steps)

# print everything in matplotlib

fig, axs = plt.subplots(2, 1, sharex=True)

ax = axs[0]
ax.plot(times, path[:, 0], label="x")
ax.plot(times, path[:, 1], label="x")
ax.plot(times, path[:, 2], label="x")

ax = axs[1]
ax.plot(times, desired_vel[:, 0], label="vx")
ax.plot(times, desired_vel[:, 1], label="vy")
ax.plot(times, desired_vel[:, 2], label="vx")

plt.show()


# with mujoco.viewer.launch_passive(model, data) as viewer:
#     while viewer.is_running() :
#         # mujoco.mj_step(model, data)
#         # Pick up changes to the physics state, apply perturbations, update options from GUI.
#         viewer.sync()
#         time.sleep(0.01)

print("done")


model_pin = load_pinocchio_wrapper("iiwa_pusher")
print(model_pin)
print(model_pin.model)

viz = MeshcatVisualizer(
    model_pin.model, model_pin.collision_model, model_pin.visual_model
)

# Start a new MeshCat server and client.
# Note: the server can also be started separately using the "meshcat-server" command in a terminal:
# this enables the server to remain active after the current script ends.
#
# Option open=True pens the visualizer.
# Note: the visualizer can also be opened seperately by visiting the provided URL.
try:
    viz.initViewer(open=False)
except ImportError as err:
    print(
        "Error while initializing the viewer. It seems you should install Python meshcat"
    )
    print(err)
    sys.exit(0)
#
# # Load the robot in the viewer.
viz.loadViewerModel()
#
# # Display a robot configuration.
# q0 = pin.neutral(model_pin.model)


q0array = np.array([v for _, v in q0.items()])

viz.display(q0array)
viz.displayVisuals(True)

# IDX_TOOL = model_pin.model.getFrameId('frametool')
# IDX_BASIS = model_pin.model.getFrameId('framebasis')

print("name")
for i, n in enumerate(model_pin.model.names):
    print(i, n)

print("frames")
for f in model_pin.model.frames:
    print(f.name, "attached to joint #", f.parent)

IDX_PUSHER = model_pin.model.getFrameId("pusher_stick")
IDX_PUSHER_TIP = model_pin.model.getFrameId("pusher_tip")
IDX_PUSHER_TIP_GOAL = model_pin.model.getFrameId("pusher_tip_goal")

q = q0array
robot = model_pin
pin.framesForwardKinematics(model_pin.model, model_pin.data, q)
oMtool = robot.data.oMf[IDX_PUSHER]
print("Tool placement:", oMtool)
oMtooltip = robot.data.oMf[IDX_PUSHER_TIP]
oMtool_goal = robot.data.oMf[IDX_PUSHER_TIP_GOAL]

print("Tool tip placement:", oMtooltip)

assert np.linalg.norm(position_pusher_tip_mujoco - oMtooltip.translation) < 1e-4
assert np.linalg.norm(position_pusher_mujoco - oMtool.translation) < 1e-4
assert np.linalg.norm(postion_pusher_tip_goal - oMtool_goal.translation) < 1e-4

assert np.linalg.norm(r_pusher_tip_mujoco - oMtooltip.rotation) < 1e-4
assert np.linalg.norm(r_pusher_tip_goal - oMtool_goal.rotation) < 1e-4







# while True:
#     time.sleep(0.01)

# input("Press Enter to continue...")
#
# # q = q0.copy()
# q = q0array
# herr = []
# DT = .01
# for i in range(500):  # Integrate over 2 second of robot life
#
#     # Run the algorithms that outputs values in robot.data
#     pin.framesForwardKinematics(robot.model,robot.data,q)
#     pin.computeJointJacobians(robot.model,robot.data,q)
#
#     # Placement from world frame o to frame f oMtool  
#     oMtool = robot.data.oMf[IDX_PUSHER_TIP]
#     oMgoal = robot.data.oMf[IDX_PUSHER_TIP_GOAL]
#
#     # 6D error between the two frame
#     tool_nu = pin.log(oMtool.inverse()*oMgoal).vector
#
#     # Get corresponding jacobian
#     tool_Jtool = pin.computeFrameJacobian(robot.model, robot.data, q, IDX_PUSHER_TIP)
#
#     # Control law by least square
#     vq = np.linalg.pinv(tool_Jtool)@tool_nu
#
#     q = pin.integrate(robot.model,q, vq * DT)
#     print("new q ", q)
#     if i % 10 == 0:
#         viz.display(q)
#         time.sleep(.1)
#











tau_gravity = pin.rnea(
    robot.model, robot.data, q, np.zeros(robot.nv), np.zeros(robot.nv)
)
print("tau_gravity", tau_gravity)

viewer = mujoco.viewer.launch_passive(model, data)
data.ctrl = tau_gravity
# np.zeros(model.nu)


def get_robot_joints():
    """ """
    joints = ["A1", "A2", "A3", "A4", "A5", "A6", "A7"]
    return np.array([data.joint(j).qpos[0] for j in joints])


def get_robot_vel():
    """ """
    joints = ["A1", "A2", "A3", "A4", "A5", "A6", "A7"]
    return np.array([data.joint(j).qvel[0] for j in joints])


class PositionController:
    def __init__(self, gravity_compenstation=True):
        self.kp = 0
        self.kv = 0
        self.gravity_compenstation = gravity_compenstation
        self.q_desired = q0array

    def get_u(self):
        q = get_robot_joints()
        qvel = get_robot_vel()
        out = self.kp * (self.q_desired - q) - self.kv * qvel
        if self.gravity_compenstation:
            out += pin.rnea(
                robot.model, robot.data, q, np.zeros(robot.nv), np.zeros(robot.nv)
            )
        return out


class Trajectory_controller:

    def __init__(self):
        """ """
        self.path = path
        self.rotation_path = rotation_path 
        self.path_vel = desired_vel
        self.path_time = times
        self.kvv = 1
        self.qvl = 0.1
        self.k_invdyn = 1

    def get_u(self):
        """ 
        """
        t = data.time

        if t > self.path_time[-1]:
            idx = -1
        else:
            idx = np.argmax(self.path_time > t)
        pdes = self.path[idx]

        # Compute the error term in the end effector position

        q = get_robot_joints()
        qvel = get_robot_vel()

        only_position = False

        if only_position:

            pin.framesForwardKinematics(robot.model, robot.data, q)
            pin.computeJointJacobians(robot.model, robot.data, q)
            oMtool = robot.data.oMf[IDX_PUSHER_TIP]

            o_Jtool3 = pin.computeFrameJacobian(
                robot.model, robot.data, q, IDX_PUSHER_TIP, pin.LOCAL_WORLD_ALIGNED
            )[:3, :]

            o_TG = oMtool.translation - pdes
            v_des = -self.k_invdyn * np.linalg.pinv(o_Jtool3) @ o_TG


        else:
            # I compute now the error in local frame
           # Run the algorithms that outputs values in robot.data
            pin.framesForwardKinematics(robot.model,robot.data,q)
            pin.computeJointJacobians(robot.model,robot.data,q)

            # Placement from world frame o to frame f oMtool  
            oMtool = robot.data.oMf[IDX_PUSHER_TIP]

            _oMgoal = self.rotation_path[idx].as_matrix()
            oMgoal = pin.SE3( _oMgoal, pdes)

            # 6D error between the two frame
            tool_nu = pin.log(oMtool.inverse()*oMgoal).vector

            # Get corresponding jacobian
            tool_Jtool = pin.computeFrameJacobian(robot.model, robot.data, q, IDX_PUSHER_TIP)

            # Control law by least square
            v_des = self.k_invdyn * np.linalg.pinv(tool_Jtool)@tool_nu

        aq = self.kvv * (v_des - qvel) - self.qvl * qvel
        return pin.rnea(robot.model, robot.data, q, qvel, aq)


class Goal_controller:

    def __init__(self):
        self.idx = 0
        self.k_invdyn = .5
        self.qvl = 0.1
        # self.kvv = 1.
        self.kvv = .5

    def get_u(self):
        """
        """

        q = get_robot_joints()
        qvel = get_robot_vel()

        only_position = False
        if only_position:
            pin.framesForwardKinematics(robot.model, robot.data, q)
            pin.computeJointJacobians(robot.model, robot.data, q)

            # Placement from world frame o to frame f oMtool
            oMtool = robot.data.oMf[IDX_PUSHER_TIP]
            oMgoal = robot.data.oMf[IDX_PUSHER_TIP_GOAL]

            # 3D jacobian in world frame
            o_Jtool3 = pin.computeFrameJacobian(
                robot.model, robot.data, q, IDX_PUSHER_TIP, pin.LOCAL_WORLD_ALIGNED
            )[:3, :]

            # vector from tool to goal, in world frame
            o_TG = oMtool.translation - oMgoal.translation

            v_des = -self.k_invdyn * np.linalg.pinv(o_Jtool3) @ o_TG
        else:
            # I compute now the error in local frame
           # Run the algorithms that outputs values in robot.data
            pin.framesForwardKinematics(robot.model,robot.data,q)
            pin.computeJointJacobians(robot.model,robot.data,q)

            # Placement from world frame o to frame f oMtool  

            oMtool = robot.data.oMf[IDX_PUSHER_TIP]
            oMgoal = robot.data.oMf[IDX_PUSHER_TIP_GOAL]

            # 6D error between the two frame
            tool_nu = pin.log(oMtool.inverse()*oMgoal).vector

            # Get corresponding jacobian
            tool_Jtool = pin.computeFrameJacobian(robot.model, robot.data, q, IDX_PUSHER_TIP)

            # Control law by least square
            v_des = np.linalg.pinv(tool_Jtool)@tool_nu

        aq = self.kvv * (v_des - qvel) - self.qvl * qvel
        return pin.rnea(robot.model, robot.data, q, qvel, aq)


# c = PositionController()
# c = Goal_controller()  # TODO: test this controller!
c = Trajectory_controller()


data.time = 0  # set time to 0

limit_rate = True
# while viewer.is_running() :
last_sim_time = data.time
last_real_time = time.time()
last_viewer_sync = time.time()
scene = viewer._user_scn

# add 

Ncapusule_path = 20

for i in range(Ncapusule_path):
    p = path[int(i*len(path)/Ncapusule_path)]
# def add_visual_capsule(scene, point1, point2, radius, rgba):
    point1 = p
    point2 = np.copy(p)
    point2[2] += 0.01
    add_visual_capsule(scene, point1, point2, 0.01, np.array([0, 1, 0, .5]))

add_trace_every = .5 # s
last_trace  = -1
extra_time = 1

while data.time < max_sim_time + extra_time:
    data.ctrl = c.get_u()
    mujoco.mj_step(model, data)
    tic = time.time()
    if limit_rate:
        compute_time = tic - last_real_time
        sim_time = data.time - last_sim_time
        if compute_time < sim_time:
            print("sleeping", sim_time - compute_time)
            time.sleep(sim_time - compute_time)
    if tic - last_viewer_sync > 1.0 / 24.0:
        # update the position of the goal
        viewer.sync()
        last_viewer_sync = tic
        # print("current q")
        # print(get_robot_joints())
    if data.time - last_trace > add_trace_every:
        # get the position of pusher-tip
        p = data.geom("pusher_tip").xpos
        point2 = np.copy(p)
        point2[2] += 0.01
        add_visual_capsule(scene, p, point2, 0.01, np.array([1, 0, 0, .5]))
        last_trace = data.time
        


    last_sim_time = data.time
    last_real_time = time.time()

print("last configuration is", q)

print("sim done, closing viewer")


# lets define a position for the end effector

# <geom  type="sphere" size="0.01" rgba="1. 0. 0. .5" contype="0" conaffinity="0"/>


# time.sleep(0.01)

# mujoco.mj_step(model, data)
# viewer.sync()
# time.sleep(0.01)
viewer.close()
