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



joints = [ "A1", "A2", "A3", "A4", "A5", "A6", "A7" ]
          # pos="0 0 0" axis="0 0 1" range="-2.96706 2.96706"/>

q0 = { "A1":0, "A2":.5, "A3":0, "A4":-1.5, "A5":0, "A6":  0, "A7":0 }
# q0 = { "A1":0, "A2":0, "A3":0, "A4":0, "A5":0, "A6":  0, "A7":0 }

for k,v in q0.items():
    data.joint(k).qpos[0] = v
    # data.qpos[model.joint_name2id(k)] = v

mujoco.mj_forward(model, data)

#init

position_pusher_tip_mujoco = data.geom( "pusher_tip" ).xpos
position_pusher_mujoco = data.geom( "pusher_stick" ).xpos
postion_pusher_tip_goal = data.geom( "pusher_tip_goal" ).xpos



viewer = mujoco.viewer.launch_passive(model, data)

while viewer.is_running() :
    viewer.sync()
    time.sleep(0.01)
viewer.close()


pusher_tip = "pusher_tip"
pusher_tip_goal = "pusher_tip_goal"


x_pusher_tip = data.geom(pusher_tip).xpos
x_pusher_tip_goal = data.geom(pusher_tip_goal).xpos

max_T = 2
num_steps = 100
times = np.linspace(0, max_T, num_steps)

nx = len(x_pusher_tip)

path = np.zeros((100,nx))
for i in range(nx):
    path[:,i] = np.interp(times, [0, max_T], [x_pusher_tip[i], x_pusher_tip_goal[i]])


displacement = np.diff(path, axis=0)
displacement = np.vstack([displacement, displacement[-1,:]])


desired_vel = displacement / (max_T / num_steps)

# print everything in matplotlib

fig, axs = plt.subplots(2,1, sharex=True)

ax = axs[0]
ax.plot(times, path[:,0], label="x")
ax.plot(times, path[:,1], label="x")
ax.plot(times, path[:,2], label="x")

ax = axs[1]
ax.plot(times, desired_vel[:,0], label="vx")
ax.plot(times, desired_vel[:,1], label="vy")
ax.plot(times, desired_vel[:,2], label="vx")

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



q0array = np.array([ v for _,v in q0.items() ])

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

assert np.linalg.norm( position_pusher_tip_mujoco  - oMtooltip.translation) < 1e-6
assert np.linalg.norm( position_pusher_mujoco  - oMtool.translation) < 1e-6
assert np.linalg.norm( postion_pusher_tip_goal  - oMtool_goal.translation) < 1e-6







# while True:
#     time.sleep(0.01)

input("Press Enter to continue...")
tau_gravity = pin.rnea(robot.model, robot.data, q, np.zeros(robot.nv), np.zeros(robot.nv))
print("tau_gravity", tau_gravity)

viewer = mujoco.viewer.launch_passive(model, data)
data.ctrl = tau_gravity
# np.zeros(model.nu)


def get_robot_joints():
    """
    """
    joints = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7']
    return np.array( [ data.joint(j).qpos[0] for j in joints ])

def get_robot_vel():
    """
    """
    joints = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7']
    return np.array( [ data.joint(j).qvel[0] for j in joints ])





class PositionController:
    def __init__(self,gravity_compenstation=True):
        self.kp = 0
        self.kv = 0
        self.gravity_compenstation = gravity_compenstation
        self.q_desired = q0array 

    def get_u(self):
        q = get_robot_joints()
        qvel = get_robot_vel()
        out = self.kp * (self.q_desired - q) - self.kv * qvel
        if self.gravity_compenstation:
            out += pin.rnea(robot.model, robot.data, q, np.zeros(robot.nv), np.zeros(robot.nv))
        return out

class FollowTrajectoryController:

    def __init__(self,gravity_compenstation=True):
        self.kp = 1
        self.kv = 1
        self.gravity_compenstation = gravity_compenstation
        # self.q_desired = q0array 
        self.path = path
        self.path_vel = desired_vel
        self.path_time = times
        self.idx = 0
        self.k_invdyn  = 1.
        self.qvl = .1
        self.kvv =  1

    def get_u(self):

        t = data.time
        # get the index of the first time that is greater than the current time
        idx = np.argmax(self.path_time > t)
        qdes = self.path[idx]
        qvel = self.path_vel[idx]
        q = get_robot_joints()
        qvel = get_robot_vel()

        # # do a call to pinocchio
        # x_pusher_tip = data.geom(pusher_tip).xpos
        # x_pusher_tip_goal = data.geom(pusher_tip_goal).xpos
        # error = x_pusher_tip  - x_pusher_tip_goal 

        pin.framesForwardKinematics(robot.model,robot.data,q)
        pin.computeJointJacobians(robot.model,robot.data,q)

        # Placement from world frame o to frame f oMtool
        oMtool = robot.data.oMf[IDX_PUSHER_TIP]
        oMgoal = robot.data.oMf[IDX_PUSHER_TIP_GOAL]

        # 3D jacobian in world frame
        o_Jtool3 = pin.computeFrameJacobian(robot.model,robot.data,q,IDX_PUSHER_TIP,pin.LOCAL_WORLD_ALIGNED)[:3,:]

        # vector from tool to goal, in world frame
        o_TG = oMtool.translation-oMgoal.translation
        
        # Control law by least square
        v_des = - self.k_invdyn  * np.linalg.pinv(o_Jtool3)@o_TG

        aq = self.kvv * ( v_des  - qvel) - self.qvl  * qvel

        # print("vq", vq)
        # # out = self.k_invdyn *  vq
        # aq =  * vq
        # c

        # out = pin.rnea(robot.model, robot.data, q, qvel , np.zeros(7))
        # return out


        # if self.gravity_compenstation:
        #     # out += 
        return pin.rnea(robot.model, robot.data, q, qvel, aq)
                        # np.zeros(robot.nv), np.zeros(robot.nv))
        # return out













# c = PositionController()
c = FollowTrajectoryController() # TODO: test this controller!



max_sim_time = 10 # in seconds

data.time = 0 # set time to 0
# tic = time.time()

limit_rate = True
# while viewer.is_running() :
last_sim_time = data.time
last_real_time = time.time()
last_viewer_sync = time.time()
while data.time < max_sim_time:
    data.ctrl = c.get_u() 
    mujoco.mj_step(model, data)
    tic = time.time()
    if limit_rate:
        compute_time  =  tic  - last_real_time
        sim_time = data.time - last_sim_time
        if compute_time < sim_time:
            print("sleeping", sim_time - compute_time)
            time.sleep(sim_time - compute_time)

    if tic - last_viewer_sync > 1. / 24.:
        viewer.sync()
        last_viewer_sync = tic
    last_sim_time = data.time
    last_real_time = time.time()


print("sim done, closing viewer")




# lets define a position for the end effector

            # <geom  type="sphere" size="0.01" rgba="1. 0. 0. .5" contype="0" conaffinity="0"/>




        # time.sleep(0.01)

    # mujoco.mj_step(model, data)
    # viewer.sync()
    # time.sleep(0.01)
viewer.close()





