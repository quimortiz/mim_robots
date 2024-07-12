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
# import cv2
import pathlib
from scipy.spatial.transform import Rotation as RR
import yaml
import pickle
from aruco_utils import *
from pin_croco_utils import *
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import sys
import copy
import math

import argparse
import crocoddyl
import numpy as np
import example_robot_data
import copy

from scipy.spatial.transform import Slerp















def q_array_to_dict(q):
    joints = ["A1", "A2", "A3", "A4", "A5", "A6", "A7"]
    return {j: q[i] for i, j in enumerate(joints)}


def q_dict_to_array(q):
    return np.array([v for _, v in q.items()])


# fix the seed.
np.random.seed(0)


argp = argparse.ArgumentParser()

argp.add_argument("--callibrate_intrinsics", action="store_true")
argp.add_argument("--callibrate_extrinsics", action="store_true")
argp.add_argument("--record_movement", action="store_true")

args = argp.parse_args()


model = load_mujoco_model("iiwa_pusher")
data = mujoco.MjData(model)

mujoco.mj_forward(model, data)


desired_z = 0.05

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

q0 = {"A1": 0, "A2": 0.5, "A3": 0, "A4": -1.5, "A5": 0, "A6": 1.5, "A7": 0}

for k, v in q0.items():
    data.joint(k).qpos[0] = v

mujoco.mj_forward(model, data)


position_pusher_tip_mujoco = data.geom("pusher_tip").xpos
position_pusher_mujoco = data.geom("pusher_stick").xpos
postion_pusher_tip_goal = data.geom("pusher_tip_goal").xpos

r_pusher_tip_mujoco = data.geom("pusher_tip").xmat.reshape(3, 3)
r_pusher_tip_goal = data.geom("pusher_tip_goal").xmat.reshape(3, 3)


pusher_tip = "pusher_tip"
pusher_tip_goal = "pusher_tip_goal"


x_pusher_tip = data.geom(pusher_tip).xpos
x_pusher_tip_goal = data.geom(pusher_tip_goal).xpos

r_pusher_tip = data.geom(pusher_tip).xmat.reshape(3, 3)
r_pusher_tip_goal = data.geom(pusher_tip_goal).xmat.reshape(3, 3)

# convert to quaternions
q_pusher_tip = RR.from_matrix(r_pusher_tip).as_quat()
q_pusher_tip_goal = RR.from_matrix(r_pusher_tip_goal).as_quat()


num_steps = 100

max_sim_time = 20  # in seconds
max_T = max_sim_time

slerp = Slerp([0, max_T], RR.from_quat([q_pusher_tip, q_pusher_tip_goal]))

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
    draw_3d_axis_from_R(ax, x, R, length=0.05)

plt.show()

# sys.exit()

displacement = np.diff(path, axis=0)
displacement = np.vstack([displacement, displacement[-1, :]])


desired_vel = displacement / (max_T / num_steps)

# print everything in matplotlib

fig, axs = plt.subplots(2, 1, sharex=True)

ax = axs[0]
ax.plot(times, path[:, 0], label="x")
ax.plot(times, path[:, 1], label="y")
ax.plot(times, path[:, 2], label="z")

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

q0array_orig = np.copy(q0array)

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
print("oMtool_goal", oMtool_goal)

assert np.linalg.norm(position_pusher_tip_mujoco - oMtooltip.translation) < 1e-4
assert np.linalg.norm(position_pusher_mujoco - oMtool.translation) < 1e-4
assert np.linalg.norm(postion_pusher_tip_goal - oMtool_goal.translation) < 1e-4

assert np.linalg.norm(r_pusher_tip_mujoco - oMtooltip.rotation) < 1e-4
assert np.linalg.norm(r_pusher_tip_goal - oMtool_goal.rotation) < 1e-4


# let

# lb = np.array([.2,-.7])
# ub = np.array([1,+.7])
# num_points = 10

# x should go from .3 to .6
# y should go from -.4 to .4


lb = np.array([0.4, -0.30])
ub = np.array([0.75, +0.30])


center = (lb + ub) / 2.0


# generate a path

p1 = np.array([ub[0], ub[1]])
p2 = np.array([lb[0], ub[1]])
p3 = np.array([lb[0], lb[1]])
p4 = np.array([ub[0], lb[1]])

solve_mpc = False
if solve_mpc:

    # # # # # # # # # # # # # # #
    ###       LOAD ROBOT      ###
    # # # # # # # # # # # # # # #

    example_seb = False
    example_carlos = True

    if example_seb:
        id_endeff = model_pin.model.getFrameId("pusher_tip")
        nq = robot.model.nq
        nv = robot.model.nv
        robot_model = robot.model

        # Reset the robot to some initial state, get initial placement and define goal placement
        # q0 = pin.utils.zero(nq)
        q0 = q0array
        dq0 = pin.utils.zero(nv)
        M0 = robot.data.oMf[id_endeff]
        M_des = robot.data.oMf[IDX_PUSHER_TIP_GOAL]
        # pin.SE3(np.eye(3), np.matrix([.5, .5, .5]).T)

        # Params
        dt = 5e-2  # Integration step for DDP (s)         # 5e-3 #1e-2
        N = 10  # Number of knots in the MPC horizon   # for [1e-1, 1e-3, 1e-6, 1.] 2e-2, 30
        cost_weights = [
            2.0,
            1e0,
            2e-4,
            1e3,
        ]  # running cost, x_reg, u_reg, terinal_cost [10., 1e-1, 1e-4, 1e4] dt=1e-2, N = 50
        x0 = np.concatenate([q0, dq0])

        # cost models
        # Mref = crocoddyl.FramePlacement(id_endeff, M_des)
        state = crocoddyl.StateMultibody(robot.model)
        # goalTrackingCost = crocoddyl.CostModelFramePlacement(state, Mref)
        # xRegCost = crocoddyl.CostModelState(state)
        # uRegCost = crocoddyl.CostModelControl(state)

        nu = 7
        framePlacementResidual = crocoddyl.ResidualModelFramePlacement(
            state, robot_model.getFrameId("pusher_tip"), oMtool_goal, nu
        )
        uResidual = crocoddyl.ResidualModelControl(state, nu)
        xResidual = crocoddyl.ResidualModelState(state, x0, nu)
        goalTrackingCost = crocoddyl.CostModelResidual(state, framePlacementResidual)
        xRegCost = crocoddyl.CostModelResidual(state, xResidual)
        uRegCost = crocoddyl.CostModelResidual(state, uResidual)

        # nu = 7
        # framePlacementResidual = crocoddyl.ResidualModelFramePlacement(
        #     state,
        #     robot_model.getFrameId("pusher_tip"),  oMtool_goal , nu)
        #
        # goalTrackingCost = crocoddyl.CostModelResidual(state, framePlacementResidual)
        #

        # Create cost model per each action model
        runningCostModel = crocoddyl.CostModelSum(state)
        terminalCostModel = crocoddyl.CostModelSum(state)

        #  Running and terminal cost functions
        runningCostModel.addCost("endeff", goalTrackingCost, cost_weights[0])  # 5
        runningCostModel.addCost("stateReg", xRegCost, cost_weights[1])  # 1e-2
        runningCostModel.addCost("ctrlReg", uRegCost, cost_weights[2])  # 1e-4
        terminalCostModel.addCost("endeff", goalTrackingCost, cost_weights[3])  # 1e3

        # Create the action model
        actuation = crocoddyl.ActuationModelFull(state)
        runningModel = crocoddyl.IntegratedActionModelEuler(
            crocoddyl.DifferentialActionModelFreeFwdDynamics(
                state, actuation, runningCostModel
            ),
            dt,
        )
        terminalModel = crocoddyl.IntegratedActionModelEuler(
            crocoddyl.DifferentialActionModelFreeFwdDynamics(
                state, actuation, terminalCostModel
            )
        )

        # Create the problem
        problem = crocoddyl.ShootingProblem(x0, [runningModel] * N, terminalModel)

        # Creating the DDP solver for this OC problem, defining a logger
        ddp = crocoddyl.SolverDDP(problem)
        ddp.setCallbacks(
            [
                crocoddyl.CallbackVerbose(),
                crocoddyl.CallbackLogger(),
            ]
        )
        ddp.solve(ddp.xs, ddp.us, maxiter=10)

        print(
            "Finally reached = ",
            ddp.problem.terminalData.differential.multibody.pinocchio.oMf[
                robot_model.getFrameId("pusher_tip")
                # IDX_PUSHER_TIP
                # robot.getFrameId("ctor")
            ].translation.T,
        )

        # Plotting the solution and the solver convergence

        WITHPLOT = True
        WITHDISPLAY = True

        if WITHPLOT:
            log = ddp.getCallbacks()[1]
            crocoddyl.plotOCSolution(log.xs, log.us, figIndex=1, show=False)
            crocoddyl.plotConvergence(
                log.costs,
                log.pregs,
                log.dregs,
                log.grads,
                log.stops,
                log.steps,
                figIndex=2,
            )

        # Visualizing the solution in gepetto-viewer
        WITHDISPLAY = True
        if WITHDISPLAY:
            try:
                import gepetto

                cameraTF = [2.0, 2.68, 0.54, 0.2, 0.62, 0.72, 0.22]
                gepetto.corbaserver.Client()
                display = crocoddyl.GepettoDisplay(kinova, 4, 4, cameraTF, floor=False)
            except Exception:
                display = crocoddyl.MeshcatDisplay(robot)

            display.rate = -1
            display.freq = 1
            while True:
                display.displayFromSolver(ddp)
                time.sleep(1.0)

    if example_carlos:
        # robot = example_robot_data.load("ur5")
        robot = model_pin
        robot_model = robot.model
        nq = robot_model.nq
        nv = robot_model.nv
        nu = nv
        # q0 = np.array([0, 0, 0, 0, 0, 0, 0.])
        q0 = q0array
        v0 = np.zeros(nv)
        print("q0", q0)
        print("v0", v0)
        x0 = np.concatenate([q0, v0]).copy()
        print("x0", x0)

        state = crocoddyl.StateMultibody(robot_model)
        actuation = crocoddyl.ActuationModelFull(state)
        # q0 = kinova.model.referenceConfigurations["arm_up"]
        # x0 = np.concatenate([q0, pin.utils.zero(robot_model.nv)])

        # Create a cost model per the running and terminal action model.
        nu = state.nv
        runningCostModel = crocoddyl.CostModelSum(state)
        terminalCostModel = crocoddyl.CostModelSum(state)

        # Note that we need to include a cost model (i.e. set of cost functions) in
        # order to fully define the action model for our optimal control problem.
        # For this particular example, we formulate three running-cost functions:
        # goal-tracking cost, state and control regularization; and one terminal-cost:
        # goal cost. First, let's create the common cost functions.
        framePlacementResidual = crocoddyl.ResidualModelFramePlacement(
            state, robot_model.getFrameId("pusher_tip"), oMtool_goal, nu
        )

        # uResidual = crocoddyl.ResidualModelControl(state, nu)
        uResidual = crocoddyl.ResidualModelControlGrav(state, nu)
        # ResidualModelControl(state, nu)
        xResidual = crocoddyl.ResidualModelState(state, x0, nu)

        goalTrackingCost = crocoddyl.CostModelResidual(state, framePlacementResidual)
        xRegCost = crocoddyl.CostModelResidual(state, xResidual)
        uRegCost = crocoddyl.CostModelResidual(state, uResidual)

        acc_refs = crocoddyl.ResidualModelJointAcceleration(state, nu)
        accCost = crocoddyl.CostModelResidual(state, acc_refs)

        _stateWeights = np.array([0.0] * 7 + [1.0] * 7)
        _stateResidual = crocoddyl.ResidualModelState(state, x0, nu)
        _stateActivation = crocoddyl.ActivationModelWeightedQuad(_stateWeights**2)
        _stateReg = crocoddyl.CostModelResidual(state, _stateActivation, _stateResidual)

        # pinocchio::Motion frame_motion = pinocchio::Motion::Zero();

        res_frame_vel = crocoddyl.ResidualModelFrameVelocity(
            state,
            robot_model.getFrameId("pusher_tip"),
            pin.Motion.Zero(),
            pin.ReferenceFrame.LOCAL,
            nu,
        )

        cost_frame_vel = crocoddyl.CostModelResidual(state, res_frame_vel)

        # ResidualModelFrameVelocity

        # final_stateWeights = np.array( [0.0] * 7 + [1.] * 7  )
        #
        # zero_vel = crocoddyl.ResidualModelState( state, final_stateWeights )
        #
        # stateActivation = crocoddyl.ActivationModelWeightedQuad(final_stateWeights**2)
        #
        # stateReg_only_vel = crocoddyl.CostModelResidual( state, stateActivation, xResidual)
        #

        # Then let's added the running and terminal cost functions

        runningCostModel.addCost("gripperPose", goalTrackingCost, 1e-1)
        runningCostModel.addCost("xReg", xRegCost, 1e-1)
        runningCostModel.addCost("uReg", uRegCost, 1e-1)
        runningCostModel.addCost("xdotdotReg", accCost, 1.0)
        runningCostModel.addCost("frame_final_vel", cost_frame_vel, 1e-1)

        terminalCostModel.addCost("gripperPose", goalTrackingCost, 1e4)
        terminalCostModel.addCost("stateRegVel", _stateReg, 1e3)

        # terminalCostModel.addCost("final_vel", stateReg_only_vel, 1e-1)

        # reach with zero velocity

        # Next, we need to create an action model for running and terminal knots. The
        # forward dynamics (computed using ABA) are implemented
        # inside DifferentialActionModelFreeFwdDynamics.
        dt = 1e-2
        runningModel = crocoddyl.IntegratedActionModelEuler(
            crocoddyl.DifferentialActionModelFreeFwdDynamics(
                state, actuation, runningCostModel
            ),
            dt,
        )
        terminalModel = crocoddyl.IntegratedActionModelEuler(
            crocoddyl.DifferentialActionModelFreeFwdDynamics(
                state, actuation, terminalCostModel
            ),
            0.0,
        )

        # For this optimal control problem, we define 100 knots (or running action
        # models) plus a terminal knot
        T = 100
        problem = crocoddyl.ShootingProblem(x0, [runningModel] * T, terminalModel)

        # Creating the DDP solver for this OC problem, defining a logger
        solver = crocoddyl.SolverDDP(problem)

        # Solving it with the solver algorithm
        # solver.solve(x0=[x0 for _ in range(T + 1)] )
        xs = [x0] * (len(problem.runningModels) + 1)
        us = [
            m.quasiStatic(d, x0)
            for m, d in list(zip(problem.runningModels, problem.runningDatas))
        ]

        WITHPLOT = True
        if WITHPLOT:
            solver.setCallbacks(
                [
                    crocoddyl.CallbackVerbose(),
                    crocoddyl.CallbackLogger(),
                ]
            )
        else:
            solver.setCallbacks([crocoddyl.CallbackVerbose()])

        # Solving it with the solver algorithm
        solver.solve(xs, us)

        print(
            "Finally reached = ",
            solver.problem.terminalData.differential.multibody.pinocchio.oMf[
                robot_model.getFrameId("pusher_tip")
                # IDX_PUSHER_TIP
                # robot.getFrameId("ctor")
            ].translation.T,
        )

        # Plotting the solution and the solver convergence
        if WITHPLOT:
            log = solver.getCallbacks()[1]
            crocoddyl.plotOCSolution(log.xs, log.us, figIndex=1, show=False)
            crocoddyl.plotConvergence(
                log.costs,
                log.pregs,
                log.dregs,
                log.grads,
                log.stops,
                log.steps,
                figIndex=2,
            )

        # Visualizing the solution in gepetto-viewer
        WITHDISPLAY = True
        if WITHDISPLAY:
            try:
                import gepetto

                cameraTF = [2.0, 2.68, 0.54, 0.2, 0.62, 0.72, 0.22]
                gepetto.corbaserver.Client()
                display = crocoddyl.GepettoDisplay(kinova, 4, 4, cameraTF, floor=False)
            except Exception:
                display = crocoddyl.MeshcatDisplay(robot)

            display.rate = -1
            display.freq = 1
            while True:
                display.displayFromSolver(solver)
                time.sleep(1.0)

path = np.array([p2, p3, p2])

_full_path = []
_full_time = []

num_poinst_per_line = 50
max_T = 5

for i in range(len(path) - 1):
    p0 = path[i]
    pnext = path[i + 1]
    # do a linear interpolation

    line = np.zeros((num_poinst_per_line, 2))
    times = np.linspace(i * max_T, (i + 1) * max_T, num_poinst_per_line)
    for jj in range(2):
        line[:, jj] = np.interp(times, [times[0], times[-1]], [p0[jj], pnext[jj]])

    _full_path.append(line)
    _full_time.append(times)

full_path = np.concatenate(_full_path)
full_time = np.concatenate(_full_time)

# quickly plot the path

fig, ax = plt.subplots()
ax.plot(full_path[:, 0], full_path[:, 1], "o")
plt.show()

fig, ax = plt.subplots()
ax.plot(full_time, full_path[:, 0], "o", label="x")
ax.plot(full_time, full_path[:, 1], "o", label="y")
ax.set_xlabel("time")
ax.set_ylabel("position")

plt.show()


# solve ik for the first point in the path

p = np.copy(postion_pusher_tip_goal)
p[:2] = full_path[0]
frame_goal = pin.SE3(r_pusher_tip_goal, p)

q, cost = solve_ik(robot, frame_goal, IDX_PUSHER_TIP, q0array)

qs = []
init_guess_q = q0array
for p in full_path:
    # sov
    _p = np.copy(postion_pusher_tip_goal)
    _p[:2] = np.copy(p[:2])
    frame_goal = pin.SE3(r_pusher_tip_goal, _p)
    _q, cost = solve_ik(robot, frame_goal, IDX_PUSHER_TIP, init_guess_q)
    init_guess_q = _q
    viz.display(_q)
    qs.append(_q)
dt = full_time[1] - full_time[0]
qs_vel = np.diff(qs, axis=0) / dt

# generate a full line using croccodyl
mpc_generation = False

if mpc_generation:
    robot = model_pin
    robot_model = robot.model
    nq = robot_model.nq
    nv = robot_model.nv
    nu = nv
    # q0 = np.array([0, 0, 0, 0, 0, 0, 0])
    q0 = q0array
    v0 = np.zeros(nv)
    x0 = np.concatenate([q0, v0]).copy()

    state = crocoddyl.StateMultibody(robot_model)
    actuation = crocoddyl.ActuationModelFull(state)

    # Joint Limits
    # Adding the state limits penalization\n",
    print("state bounds")
    print(state.lb)
    print(state.ub)
    import pdb

    # pdb.set_trace()

    # continue here!!

    # x_lb = np.concatenate([state.l+ 1], state.lb[-state.nv :]])
    # x_ub = np.concatenate([state.ub[0 : state.nv + 1], state.ub[-state.nv :]])

    activation_xbounds = crocoddyl.ActivationModelQuadraticBarrier(
        crocoddyl.ActivationBounds(state.lb, state.ub)
    )

    x_bounds = crocoddyl.CostModelResidual(
        state,
        activation_xbounds,
        crocoddyl.ResidualModelState(state, np.zeros(14), actuation.nu),
    )

    # q0 = kinova.model.referenceConfigurations["arm_up"]
    x0 = np.concatenate([q0, pin.utils.zero(robot_model.nv)])

    # Create a cost model per the running and terminal action model.
    nu = state.nv

    runningCostModels = []
    runningModels = []

    terminalCostModel = crocoddyl.CostModelSum(state)

    # 50 Times to go to the start, 50 time to move
    T = 100
    dt = 5 * 1e-2
    for i in range(T):
        runningCostModel = crocoddyl.CostModelSum(state)

        # Note that we need to include a cost model (i.e. set of cost functions) in
        # order to fully define the action model for our optimal control problem.
        # For this particular example, we formulate three running-cost functions:
        # goal-tracking cost, state and control regularization; and one terminal-cost:
        # goal cost. First, let's create the common cost functions.

        # uResidual = crocoddyl.ResidualModelControl(state, nu)
        uResidual = crocoddyl.ResidualModelControlGrav(state, nu)
        xResidual = crocoddyl.ResidualModelState(state, x0, nu)
        xRegCost = crocoddyl.CostModelResidual(state, xResidual)
        uRegCost = crocoddyl.CostModelResidual(state, uResidual)

        acc_refs = crocoddyl.ResidualModelJointAcceleration(state, nu)
        accCost = crocoddyl.CostModelResidual(state, acc_refs)

        runningCostModel.addCost("xReg", xRegCost, 1e1)
        runningCostModel.addCost("uReg", uRegCost, 1e-1)
        runningCostModel.addCost("xdotdotReg", accCost, 1e-1)
        runningCostModel.addCost("xBounds", x_bounds, 1e5)

        # This works!!
        # runningModels[0].differential.costs.costs["uReg"].cost.calc(problem.runningDatas[0].differential.costs.costs["uReg"] , np.zeros(14),np.zeros(14))

        if i > 50:
            p = full_path[i - 50]
            _p = np.copy(postion_pusher_tip_goal)
            _p[:2] = np.copy(p[:2])
            frame_goal = pin.SE3(r_pusher_tip_goal, _p)
            framePlacementResidual = crocoddyl.ResidualModelFramePlacement(
                state, robot_model.getFrameId("pusher_tip"), frame_goal, nu
            )
            print(f"at i {i}")
            print("frame goal", frame_goal)
            goalTrackingCost = crocoddyl.CostModelResidual(
                state, framePlacementResidual
            )
            runningCostModel.addCost("gripperPose", goalTrackingCost, 1e4)

        # runningCostModels.append(runningCostModel)

        runningModel = crocoddyl.IntegratedActionModelEuler(
            crocoddyl.DifferentialActionModelFreeFwdDynamics(
                state, actuation, runningCostModel
            ),
            dt,
        )
        runningModels.append(runningModel)

        # Then let's added the running and terminal cost functions

    p = full_path[50]
    _p = np.copy(postion_pusher_tip_goal)
    _p[:2] = np.copy(p[:2])
    frame_goal = pin.SE3(r_pusher_tip_goal, _p)

    print(f"final frame")
    print("frame goal", frame_goal)

    framePlacementResidual = crocoddyl.ResidualModelFramePlacement(
        state, robot_model.getFrameId("pusher_tip"), frame_goal, nu
    )
    goalTrackingCost = crocoddyl.CostModelResidual(state, framePlacementResidual)

    terminalCostModel.addCost("gripperPose", goalTrackingCost, 1e3)

    # Next, we need to create an action model for running and terminal knots. The
    # forward dynamics (computed using ABA) are implemented
    # inside DifferentialActionModelFreeFwdDynamics.
    terminalModel = crocoddyl.IntegratedActionModelEuler(
        crocoddyl.DifferentialActionModelFreeFwdDynamics(
            state, actuation, terminalCostModel
        ),
        0.0,
    )

    # For this optimal control problem, we define 100 knots (or running action
    # models) plus a terminal knot
    # T = 100
    print("x0 is", x0)
    problem = crocoddyl.ShootingProblem(x0, runningModels, terminalModel)

    # Creating the DDP solver for this OC problem, defining a logger
    solver = crocoddyl.SolverDDP(problem)

    # Solving it with the solver algorithm
    # solver.solve(x0=[x0 for _ in range(T + 1)] )
    xs = [x0] * (len(problem.runningModels) + 1)
    us = [
        m.quasiStatic(d, x0)
        for m, d in list(zip(problem.runningModels, problem.runningDatas))
    ]

    WITHPLOT = True
    if WITHPLOT:
        solver.setCallbacks(
            [
                crocoddyl.CallbackVerbose(),
                crocoddyl.CallbackLogger(),
            ]
        )
    else:
        solver.setCallbacks([crocoddyl.CallbackVerbose()])

    # Solving it with the solver algorithm
    solver.solve(xs, us)

    print(
        "Finally reached = ",
        solver.problem.terminalData.differential.multibody.pinocchio.oMf[
            robot_model.getFrameId("pusher_tip")
            # IDX_PUSHER_TIP
            # robot.getFrameId("ctor")
        ],
    )

    # NOTE: this works
    # (Pdb) terminalModel.differential.costs.costs["gripperPose"].weight
    # 1000.0

    # problem.runningDatas[0].differential.costs.costs.todict()["uReg"]

    _names = [
        list(data.differential.costs.costs.todict().keys())
        for data in problem.runningDatas
    ]

    names = []
    for i in _names:
        for j in i:
            names.append(j)
    names = list(set(names))

    dname_vals = {}

    for i, name in enumerate(names):
        _data = []
        for j, data in enumerate(problem.runningDatas):
            if name in data.differential.costs.costs:
                _data.append(
                    {"cost": data.differential.costs.costs[name].cost, "time": j}
                )
        dname_vals[name] = _data

    print(dname_vals)

    fig, ax = plt.subplots()

    for k, v in dname_vals.items():
        times = [d["time"] for d in v]
        cost = [d["cost"] for d in v]
        ax.plot(times, cost, label=k)
    ax.legend()
    plt.show()

    # problem.runningDatas[0].differential.costs.costs:

    # for k,v in problem.runningDatas[0].differential.costs.costs:

    # Plotting the solution and the solver convergence
    if WITHPLOT:
        log = solver.getCallbacks()[1]
        # crocoddyl.plotOCSolution(log.xs, log.us, figIndex=1, show=False)

        fig, axs = plt.subplots(3, 1, sharex=True)

        # plot  joints

        ax = axs[0]
        print("len log.xs", len(log.xs))
        Xs = log.xs.tolist()
        import pdb

        pdb.set_trace()
        # print()

        Xs = log.xs.tolist()
        Us = log.us.tolist()

        ax = axs[0]
        for i in range(7):
            ax.plot([Xs[j][i] for j in range(len(Xs))], label=f"q {i}")
        ax.legend()

        ax = axs[1]
        for i in range(7, 14):
            ax.plot([Xs[j][i] for j in range(len(Xs))], label=f"qdot {i}")
        ax.legend()

        ax = axs[2]
        for u in range(len(Us[0])):
            ax.plot([Us[i][u] for i in range(len(Us))], label=f"u{u}")
        ax.legend()

        plt.show()

        crocoddyl.plotConvergence(
            log.costs, log.pregs, log.dregs, log.grads, log.stops, log.steps, figIndex=2
        )

    # Visualizing the solution in gepetto-viewer
    WITHDISPLAY = True
    if WITHDISPLAY:
        try:
            import gepetto

            cameraTF = [2.0, 2.68, 0.54, 0.2, 0.62, 0.72, 0.22]
            gepetto.corbaserver.Client()
            display = crocoddyl.GepettoDisplay(kinova, 4, 4, cameraTF, floor=False)
        except Exception:
            display = crocoddyl.MeshcatDisplay(robot)

        display.rate = -1
        display.freq = 1
        while True:
            display.displayFromSolver(solver)
            time.sleep(1.0)


print("done")
fig, ax = plt.subplots()


for i in range(7):
    ax.plot([q[i] for q in qs], label=f"joint {i}")
ax.legend()
plt.show()


q0 = q_array_to_dict(qs[0])

q0array = q_dict_to_array(q0)

viz.display(q0array)
input("Press Enter to continue...")


# sys.exit(0)


# update q array


# lb = np.array([.3,-.4])

analyse_ik = False
if analyse_ik:

    num_points = 10

    all_points = [
        (x, y)
        for x in np.linspace(lb[0], ub[0], num_points)
        for y in np.linspace(lb[1], ub[1], num_points)
    ]

    reachable_points = []

    # print("q", q)
    # print("f", f)
    # viz.display(q)
    #
    # input("Press Enter to continue...")
    #
    #
    # sys.exit(0)
    ik_solutions = []

    for point in all_points:

        point3d = np.copy(postion_pusher_tip_goal)
        point3d[:2] = point[:2]
        frame_goal = pin.SE3(r_pusher_tip_goal, point3d)
        q, cost = solve_ik(robot, frame_goal, IDX_PUSHER_TIP, q0array)
        is_reachable = cost < 1e-4
        if is_reachable:
            reachable_points.append(point)
            ik_solution = {"point": point, "q": q}
            ik_solutions.append(ik_solution)

    print(f"num reachable points {len(reachable_points)}  /  {len(all_points)}")

    fig, ax = plt.subplots()
    ax.plot([p[0] for p in reachable_points], [p[1] for p in reachable_points], "o")

    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    plt.show()

    # x should go from .3 to .6
    # y should go from -.4 to .4

    for i, ik_solution in enumerate(ik_solutions):
        print(f"ik solution {i}")
        print(ik_solution)
        viz.display(ik_solution["q"])
        input("Press Enter to continue...")


# tau_gravity = pin.rnea(
#     robot.model, robot.data, q, np.zeros(robot.nv), np.zeros(robot.nv)
# )
# print("tau_gravity", tau_gravity)

# data.ctrl = tau_gravity


def get_robot_joints(data):
    """ """
    joints = ["A1", "A2", "A3", "A4", "A5", "A6", "A7"]
    return np.array([data.joint(j).qpos[0] for j in joints])


def get_robot_vel(data):
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
        q = get_robot_joints(data)
        qvel = get_robot_vel(data)
        out = self.kp * (self.q_desired - q) - self.kv * qvel
        if self.gravity_compenstation:
            out += pin.rnea(
                robot.model, robot.data, q, np.zeros(robot.nv), np.zeros(robot.nv)
            )
        return out


class Trajectory_controller_q:

    def __init__(self, path=None, times=None, path_vel=None):
        """ """
        self.path = path
        self.path_vel = path_vel
        self.path_time = times
        self.kp = 20.0
        self.kv = 10.0

        if self.path is not None and self.path_time is not None:
            assert len(self.path) == len(self.path_time)

    def get_u(self):
        """ """
        t = data.time

        if t > self.path_time[-1]:
            idx = -1
        else:
            idx = np.argmax(self.path_time > t)
        pdes = self.path[idx]

        # Compute the error term in the end effector position

        q = get_robot_joints(data)
        qvel = get_robot_vel(data)

        if self.path_vel is None:
            aq = self.kp * (pdes - q) - self.kv * qvel
        else:

            if idx >= len(self.path_vel):
                idx = -1
            vel_des = self.path_vel[idx]
            aq = self.kp * (pdes - q) + self.kv * (vel_des - qvel)

        return pin.rnea(robot.model, robot.data, q, qvel, aq)

    def policy(self):
        return self.get_u()


class Trajectory_controller:

    def __init__(self, path, rotation_path, times, desired_vel=None):
        """ """
        self.path = path
        self.rotation_path = rotation_path
        self.path_vel = desired_vel
        if self.path_vel is not None:
            # TODO: add this here!!
            # dp / dq = J
            # dp / dt = J dq / dt
            raise NotImplementedError("not implemented yet")
        self.path_time = times
        self.kvv = 20.0
        self.qvl = 0.05
        self.k_invdyn = 4

        assert len(self.path) == len(self.path_time)
        assert len(self.rotation_path) == len(self.path_time)

    def get_u(self):
        """ """
        t = data.time

        if t > self.path_time[-1]:
            idx = -1
        else:
            idx = np.argmax(self.path_time > t)
        pdes = self.path[idx]

        # Compute the error term in the end effector position

        q = get_robot_joints(data)
        qvel = get_robot_vel(data)

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
            pin.framesForwardKinematics(robot.model, robot.data, q)
            pin.computeJointJacobians(robot.model, robot.data, q)

            # Placement from world frame o to frame f oMtool
            oMtool = robot.data.oMf[IDX_PUSHER_TIP]

            _oMgoal = self.rotation_path[idx].as_matrix()
            oMgoal = pin.SE3(_oMgoal, pdes)

            # 6D error between the two frame
            tool_nu = pin.log(oMtool.inverse() * oMgoal).vector

            # Get corresponding jacobian
            tool_Jtool = pin.computeFrameJacobian(
                robot.model, robot.data, q, IDX_PUSHER_TIP
            )

            # Control law by least square
            v_des = (
                self.k_invdyn * np.linalg.pinv(tool_Jtool) @ tool_nu
            )  # + np.linalg.pinv(tool_Jtool) * ( error in velocity )

        aq = self.kvv * (v_des - qvel) - self.qvl * qvel
        return pin.rnea(robot.model, robot.data, q, qvel, aq)


aqs = []
aqs_time = []
ffs = []





class Goal_controller:

    def __init__(self):
        self.ee_goal = None
        self.idx = 0
        self.k_invdyn = 1.0
        self.qvl = 1.0
        self.kvv = 1.0

    def get_u(self):

        q = get_robot_joints(data)
        qvel = get_robot_vel(data)

        only_position = False
        if only_position:
            pin.framesForwardKinematics(robot.model, robot.data, q)
            pin.computeJointJacobians(robot.model, robot.data, q)

            # Placement from world frame o to frame f oMtool
            oMtool = robot.data.oMf[IDX_PUSHER_TIP]
            # oMgoal = robot.data.oMf[IDX_PUSHER_TIP_GOAL]

            # 3D jacobian in world frame
            o_Jtool3 = pin.computeFrameJacobian(
                robot.model, robot.data, q, IDX_PUSHER_TIP, pin.LOCAL_WORLD_ALIGNED
            )[:3, :]

            # vector from tool to goal, in world frame
            o_TG = oMtool.translation - self.ee_goal

            v_des = -self.k_invdyn * np.linalg.pinv(o_Jtool3) @ o_TG
        else:
            # I compute now the error in local frame
            # Run the algorithms that outputs values in robot.data
            pin.framesForwardKinematics(robot.model, robot.data, q)
            pin.computeJointJacobians(robot.model, robot.data, q)

            # Placement from world frame o to frame f oMtool

            oMtool = robot.data.oMf[IDX_PUSHER_TIP]
            # oMgoal = robot.data.oMf[IDX_PUSHER_TIP_GOAL]

            # 6D error between the two frame
            tool_nu = pin.log(oMtool.inverse() * self.ee_goal).vector

            # Get corresponding jacobian
            tool_Jtool = pin.computeFrameJacobian(
                robot.model, robot.data, q, IDX_PUSHER_TIP
            )

            # Control law by least square
            v_des = np.linalg.pinv(tool_Jtool) @ tool_nu
            # TODO: should i Regualarize a little bit? Just to be sure it does not explode!

        print("v_des", v_des)
        print("qvel", qvel)
        aq = self.kvv * (v_des - qvel) - self.qvl * qvel
        print("acc", aq)
        return pin.rnea(robot.model, robot.data, q, qvel, aq)

    def policy(self):
        return self.get_u()


# c = PositionController()
# c = Goal_controller()  # TODO: test this controller!

path_for_controller = np.stack([[p[0], p[1], desired_z] for p in full_path])
rotation_for_controller = np.stack(
    [RR.from_matrix(r_pusher_tip_goal) for _ in full_path]
)
times_for_controller = full_time

# c = Trajectory_controller( path=path_for_controller, rotation_path=rotation_for_controller, times = times_for_controller)

c = Trajectory_controller_q(path=qs, path_vel=qs_vel, times=times_for_controller)
data.time = 0  # set time to 0

print("setting q0 to", q0)
for k, v in q0.items():
    data.joint(k).qpos[0] = v
    data.joint(k).qvel[0] = 0

mujoco.mj_forward(model, data)

limit_rate = True
# while viewer.is_running() :
last_sim_time = data.time
last_real_time = time.time()
last_viewer_sync = time.time()

# sys.exit()

viewer = mujoco.viewer.launch_passive(model, data)
scene = viewer._user_scn

# add

Ncapusule_path = 50

path = path_for_controller
for i in range(Ncapusule_path):
    p = path[int(i * len(path) / Ncapusule_path)]
    point1 = p
    point2 = np.copy(p)
    point2[2] += 0.01
    add_visual_capsule(scene, point1, point2, 0.01, np.array([0, 1, 0, 0.5]))

add_trace_every = 0.5  # s
last_trace = -1
extra_time = 1

all_pos = []
all_q = []
all_times = []

max_sim_time = times_for_controller[-1]


class Position_Planner:
    """ """

    def __init__(self, pin_robot, 
                 only_random = False,
                 add_capsules = False):
        """
        Notes:
        The low level force control will generate the torque commands
        """
        # q = get_robot_joints(data)
        
        self.only_random = only_random

        self.pin_robot = pin_robot
        self.start = self.pin_robot.data.oMf[IDX_PUSHER_TIP]
        pin.framesForwardKinematics(self.pin_robot.model, self.pin_robot.data, q)

        self.low_level_force_control = Goal_controller_q(SIM=True,pin_model=pin_robot)
        self.low_level_force_control.qvl = 5.0
        self.low_level_force_control.kvv = 10.0
        self.low_level_force_control.ff_kp = 0.1
        self.low_level_force_control.ff_kv = 0.1
        self.low_level_force_control.qvgoal = np.zeros(7)
        self.low_level_force_control.qgoal = q

        self.mode = "initialized"

        self.center = np.zeros(3)
        self.center[:2] = center
        self.center[2] = desired_z

        self.plan = []  # sequence of positions and velocities
        self.reference_max_vel = 0.1

        self.p_lb = np.array([lb[0], lb[1], 0.05])
        self.p_ub = np.array([ub[0], ub[1], 0.05])

        self.z_safe = .15
        self.p_lb_safe = np.array([lb[0], lb[1], 0.05])
        self.p_ub_safe = np.array([ub[0], ub[1], self.z_safe])

        self.goal = pin.SE3().Identity()
        self.default_rotation = oMtool_goal.rotation
        self.goal.translation[:2] = center
        self.goal.translation[2] = desired_z
        self.goal.rotation = self.default_rotation

        self.scene = None  # used to add visual capsules

        self.p_lb_small = self.p_lb + 0.2 * (self.p_ub - self.p_lb)
        self.p_ub_small = self.p_ub - 0.2 * (self.p_ub - self.p_lb)

        self.add_capsules = add_capsules
        self.counter = 0
        self.dist_fail_controller = 0.2  # in cartesian space

        self.sensor_object_position = np.zeros(3)
        self.sensor_object_rotation = np.eye(3)
        self.q = np.zeros(7)
        self.qv = np.zeros(7)

        self.last_hl_time = -1
        self.hl_time = 0.1 # Time step of high level planner (check if plan is going well, recompute if necessary)

    def get_data(self):
        return {
            "ee_goal": self.plan[self.counter if self.counter < len(self.plan) else -1],
            "qgoal": self.low_level_force_control.qgoal,
            "qvgoal": self.low_level_force_control.qvgoal,
        }

    def read_sensors(self,data):
        """

        """
        self.sensor_object_position = np.copy(data.body("cube_small").xpos)
        self.sensor_object_rotation = np.copy(data.body("cube_small").xmat)


    def read_robot_info(self,data):
        """

        """
        self.q = np.copy(get_robot_joints(data))
        self.qv = np.copy(get_robot_vel(data))

    def policy(self, sim_time):
        """ """

        if sim_time - self.last_hl_time > self.hl_time:
            cube = np.copy(self.sensor_object_position)
            is_outside = np.any(cube[:2] < self.p_lb_small[:2]) or np.any(
                cube[:2] > self.p_ub_small[:2]
            )
            recompute_plan = False


            if self.mode == "initialized": 
                recompute_plan  = True

            if len(self.plan) >  0:
                pass
                # recompute_plan = True
                # xdes = self.plan[self.counter if self.counter < len(self.plan) else -1]
                #
                # is_high_tracking_error = ( np.linalg.norm(cube - xdes.translation) 
                #     > self.dist_fail_controller)
                #
                # if is_high_tracking_error:
                #     recompute_plan = True

            if is_outside and not self.mode == "recovery":
                recompute_plan = True
            
            if self.counter >= len(self.plan):
                recompute_plan = True


            if recompute_plan:

                old_mode = copy.deepcopy(self.mode)
                rgba = None
                self.plan = []
                self.counter = 0

                pin.framesForwardKinematics(self.pin_robot.model, self.pin_robot.data, self.q)
                pusher = self.pin_robot.data.oMf[IDX_PUSHER_TIP]



                # what to do if the tracking error is high?


                if is_outside:
                    self.mode = "recovery"
                    rel_cube = cube - self.center

                    print("cube is outside!")
                    print("entering recovery mode")

                    # I will generate 3 goal


                    # move UP

                    # lets compute the pusher position using pinocchio

                    goal_0 = pusher.translation
                    goal_0[2] = self.z_safe

                    # Intersect the line rel_cube with the bounds (in 2D)
                    p = self.center[:2]
                    v = rel_cube[:2]

                    intersect, tmin, tmax = ray_cube_intersection( p, v, lb, ub)

                    assert intersect == True
                    assert tmin < tmax
                    assert tmin == 0
                    intersection_point = p + tmax * v

                    goal_1 = np.array([ intersection_point[0], intersection_point[1], self.z_safe ])

                    goal_2 = np.array([ intersection_point[0], intersection_point[1], desired_z ])

                    goal_3 = self.center + .2 * rel_cube 







                    # goal_0 = self.center
                    #
                    # goal_1 = np.zeros(3)
                    #
                    # goal_1[0] = rel_cube[1]
                    # goal_1[1] = rel_cube[0]
                    #
                    # # a = 2 * np.pi / 3  # lets do "a" degrees
                    # a =  np.pi / 5  # lets do "a" degrees
                    # R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
                    # goal_1[:2] = self.center[:2] +  R @ goal_1[:2]
                    # goal_1[2] = desired_z
                    #
                    # goal_2 = self.center +  1.5 * rel_cube
                    # goal_3 = self.center

                    # take an orthogonal_direction:
                    # _big_goals = [goal_1, goal_2, goal_3]
                    _big_goals = [goal_0, goal_1, goal_2, goal_3]





                    # make sure they are in the cube

                    _big_goals = [ np.clip(g, self.p_lb_safe, self.p_ub_safe) for g in _big_goals]

                    big_goals = [ pin.SE3(self.default_rotation, g) for g in _big_goals ]


                    if self.add_capsules:
                        for g in big_goals:
                            add_visual_capsule(
                                scene,
                                g.translation,
                                g.translation + np.array([0, 0, 0.1]),
                                0.05,
                                np.array([1, 1, 0, 0.1]),
                            )

                    for i in range(len(big_goals)):

                        if i == 0:
                            # TODO: decide if I should use current orientation!
                            start = pin.SE3(self.default_rotation, pusher.translation)
                        else:
                            start = big_goals[i - 1]
                        total_time = (
                            np.linalg.norm(big_goals[i].translation - start.translation) 
                            / self.reference_max_vel
                        )
                        times = np.linspace(0, total_time, int(total_time / self.hl_time))

                        # this does not get the last one!
                        
                        print(f"i is {i}")
                        for time in times:
                            new_state = pin.SE3.Interpolate(
                                start, big_goals[i],  time / total_time
                            )

                            print(f"adding new states {new_state}")
                            self.plan.append(new_state)


                        self.mode = "recovery"
                        print("new high level mode", self.mode)
                        print("len(self.plan)", len(self.plan))

                        rgba = np.array([1, 1, 0, 0.1])
                        # if self.add_capsules:
                        #     # if len(self.plan) > 1000:
                        #     # TODO: comment this out!!
                        #     for i in range(0, len(self.plan), 1):
                        #         add_visual_capsule(
                        #             scene,
                        #             self.plan[i].translation,
                        #             self.plan[i].translation + np.array([0, 0, 0.1]),
                        #             0.05,
                        #             rgba
                        #         )

                # elif old_mode == "initialized":
                #     self.mode = "to_rand"
                #     self.start = pin.SE3(
                #         data.geom("pusher_tip").xmat.reshape(3, 3),
                #         data.geom("pusher_tip").xpos,
                #     )
                #
                #     # random goal
                #     rand_goal = self.center
                #     # self.p_lb_small + np.random.rand(3) * (
                #     #     self.p_ub_small - self.p_lb_small
                #     # )
                #     self.goal = pin.SE3(self.default_rotation, rand_goal)
                #     rgba = np.array([0, 0, 1, 0.1])
                #
                #     total_time = (
                #         np.linalg.norm(self.goal.translation - self.start.translation)
                #         / self.reference_max_vel
                #     )
                #     self.times = np.linspace(0, total_time, int(total_time / self.hl_time))
                #
                #     for time in self.times:
                #         new_state = pin.SE3.Interpolate(
                #             self.start, self.goal, time / total_time
                #         )
                #         self.plan.append(new_state)

                elif self.only_random or old_mode == "to_obj" or old_mode == "initialized" or old_mode == "recovery":
                    self.mode = "to_rand"
                    self.start = pin.SE3(
                        pusher.rotation,
                        pusher.translation
                    )

                    # random goal
                    rand_goal = self.p_lb_small + np.random.rand(3) * (
                        self.p_ub_small - self.p_lb_small
                    )
                    print("rand goal is", rand_goal)

                    if old_mode  == "initialized":
                        rand_goal =  self.center

                    self.goal = pin.SE3(self.default_rotation, rand_goal)
                    rgba = np.array([0, 0, 1, 0.1])

                    total_time = (
                        np.linalg.norm(self.goal.translation - self.start.translation)
                        / self.reference_max_vel
                    )
                    self.times = np.linspace(0, total_time, int(total_time / self.hl_time))

                    for time in self.times:
                        new_state = pin.SE3.Interpolate(
                            self.start, self.goal, time / total_time
                        )
                        self.plan.append(new_state)



                elif old_mode == "to_rand":
                    self.mode = "to_obj"
                    diff_weights = np.array([1, 1, 0])
                    # self.penetration = 0.2
                    self.penetration = 0.4
                    cube_to_ball = diff_weights * np.array(cube - pusher.translation)
                    translation = (
                        cube
                        + self.penetration * cube_to_ball / np.linalg.norm(cube_to_ball)
                    )

                    translation = np.clip(translation, self.p_lb, self.p_ub)
                    goal = pin.SE3(self.default_rotation, translation)

                    start = pin.SE3(self.default_rotation, pusher.translation)

                    rgba = np.array([1, 0, 0, 0.1])

                    total_time = (
                        np.linalg.norm(goal.translation - start.translation)
                        / self.reference_max_vel
                    )
                    self.times = np.linspace(0, total_time, int(total_time / self.hl_time))

                    for time in self.times:
                        new_state = pin.SE3.Interpolate(
                            start, goal, time / total_time
                        )
                        self.plan.append(new_state)


                # Should I follow the current plan, or change mode




                    # if self.add_capsules:
                    #     add_visual_capsule(
                    #         self.scene,
                    #         self.goal.translation,
                    #         self.goal.translation + np.array([0, 0, 0.1]),
                    #         0.05,
                    #         np.array([1, 0, 0, 0.1]),
                    #     )


                for i, s in enumerate(self.plan):
                    print(f"state {i} is {s}")

                if self.add_capsules:
                    point1 = self.plan[-1]
                    # self.goal.translation
                    point2 = copy.deepcopy(point1)
                    point2.translation[2] += 0.01

                    radius = 0.05
                    print("adding capsule!!")

                    add_visual_capsule(scene, point1.translation, point2.translation, radius, rgba)

                    # for i in range(0, len(self.plan), 2):
                    #     point1 = self.plan[i].translation
                    #     point2 = copy.deepcopy(point1)
                    #     point2[2] += 0.01
                    #     radius = 0.05
                    #     rgba = np.array([0, 0, 1, 0.1])
                    #     add_visual_capsule(scene, point1, point2, radius, rgba)

                qs = []
                init_guess_q = q0array
                for p in self.plan:
                    # sov
                    # _p = np.copy(postion_pusher_tip_goal)
                    # _p[:2] = np.copy(p[:2])
                    # frame_goal = pin.SE3(r_pusher_tip_goal, _p)
                    _q, cost = solve_ik(self.pin_robot, p, IDX_PUSHER_TIP, init_guess_q)
                    init_guess_q = _q
                    viz.display(_q)
                    qs.append(_q)
                self.qs = qs

                dt = self.hl_time  # in seconds
                self.qs_vel = np.diff(qs, axis=0) / dt
                self.counter = 0

            self.low_level_force_control.qgoal = self.qs[self.counter]
            self.low_level_force_control.qvgoal = self.qs_vel[
                self.counter if self.counter < len(self.qs_vel) else -1
            ]
            self.counter += 1
            self.last_hl_time = sim_time

        return self.low_level_force_control.get_u(self.q, self.qv)

        # lets create another plan!


planner = Position_Planner(pin_robot=robot, add_capsules=True)

exp_datas = []
# with mujoco.viewer.launch_passive(model, data) as viewer:
max_sim_time = 20
while data.time < max_sim_time + extra_time:
    exp_data = {}

    planner.read_sensors(data)
    planner.read_robot_info(data)

    sim_time = data.time
    u = planner.policy(sim_time)

    exp_data.update(planner.get_data())
    exp_data.update(
        {
            "pos": np.copy(data.geom("pusher_tip").xpos),
            "q": get_robot_joints(data),
            "qvel": get_robot_vel(data),
            "time": data.time,
        }
    )
    exp_datas.append(exp_data)

    # planner.get_data()

    print("u", u)
    data.ctrl = u
    mujoco.mj_step(model, data)
    tic = time.time()
    if limit_rate:
        compute_time = tic - last_real_time
        sim_time = data.time - last_sim_time
        if compute_time < sim_time:
            # print("sleeping", sim_time - compute_time)
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
        add_visual_capsule(scene, p, point2, 0.01, np.array([1, 0, 0, 0.5]))
        last_trace = data.time

    # get the position of the puser tip
    pos = np.copy(data.geom("pusher_tip").xpos)
    all_q.append(np.copy(get_robot_joints(data)))
    all_pos.append(pos)
    all_times.append(data.time)

    last_sim_time = data.time
    last_real_time = time.time()

viewer.close()

fig, axs = plt.subplots(2, 1)
# ax.plot( [ d["pos"][0] for d in exp_datas] , label="x")
# ax.plot( [ d["pos"][1] for d in exp_datas] , label="y")

ax = axs[0]
times = [d["time"] for d in exp_datas]
for i in range(7):
    ax.plot(times, [d["q"][i] for d in exp_datas], label=f"joint {i}")
    ax.plot(
        times,
        [d["qgoal"][i] for d in exp_datas],
        label=f"joint {i} goal",
        linestyle="--",
    )

ax = axs[1]
ax.plot(times, [d["pos"][0] for d in exp_datas], label="x")
ax.plot(times, [d["pos"][1] for d in exp_datas], label="y")
ax.plot(times, [d["ee_goal"].translation[0] for d in exp_datas], label="x goal")
ax.plot(times, [d["ee_goal"].translation[1] for d in exp_datas], label="y goal")

plt.show()


fig, ax = plt.subplots()
ax.plot([d["pos"][0] for d in exp_datas], label="x")
ax.plot([d["pos"][1] for d in exp_datas], label="y")
ax.plot([d["ee_goal"].translation[0] for d in exp_datas], label="x goal")
ax.plot([d["ee_goal"].translation[1] for d in exp_datas], label="y goal")
ax.legend()
plt.show()

fig, axs = plt.subplots(3, 1, sharex=True)
ax = axs[0]
ax.plot(all_times, [p[0] for p in all_pos], label="x", color="red", alpha=0.5)
ax.plot(all_times, [p[1] for p in all_pos], label="y", color="green", alpha=0.5)
ax.plot(all_times, [p[2] for p in all_pos], label="z", color="blue", alpha=0.5)
ax.plot(full_time, full_path[:, 0], "o", label="x", color="red", alpha=0.5)
ax.plot(full_time, full_path[:, 1], "o", label="y", color="green", alpha=0.5)
ax.plot(
    full_time, [desired_z for _ in full_time], "o", label="z", color="blue", alpha=0.5
)
ax.legend()
ax = axs[1]

colors = ["red", "green", "blue", "black", "orange", "purple", "brown"]
for i in range(7):
    ax.plot(
        all_times, [q[i] for q in all_q], label=f"joint {i}", color=colors[i], alpha=0.5
    )
    ax.plot(
        full_time,
        [q[i] for q in qs],
        ".-",
        label=f"joint {i}",
        color=colors[i],
        alpha=0.5,
    )

ax.legend()

ax = axs[2]
for i in range(7):
    ax.plot(
        aqs_time, [q[i] for q in aqs], label=f"a-joint {i}", color=colors[i], alpha=0.5
    )

ax.legend()
plt.show()

print("last configuration is", q)

print("sim done, closing viewer")


# lets define a position for the end effector

# <geom  type="sphere" size="0.01" rgba="1. 0. 0. .5" contype="0" conaffinity="0"/>


# time.sleep(0.01)

# mujoco.mj_step(model, data)
# viewer.sync()
# time.sleep(0.01)
