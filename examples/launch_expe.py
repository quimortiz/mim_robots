import pybullet as p
import numpy as np
from dynamic_graph_head import (
    ThreadHead,
    SimHead,
    HoldPDController,
    HoldPDControllerGravComp,
)
from datetime import datetime

# import dynamic_graph_manager_cpp_bindings
from mim_robots.robot_loader import load_bullet_wrapper, load_pinocchio_wrapper
from mim_robots.robot_loader import (
    load_bullet_wrapper,
    load_mujoco_model,
    load_pinocchio_wrapper,
    load_mujoco_wrapper,
)
from mim_robots.pybullet.env import BulletEnvWithGround
from mim_robots.robot_list import MiM_Robots
import pathlib
import os

# os.sys.path.insert(1, str(python_path))
import time
import pinocchio as pin
from pin_croco_utils import *
import copy


lb = np.array([0.4, -0.30])
ub = np.array([0.75, +0.30])

center = (lb + ub) / 2.0
desired_z = 0.05
default_rotation = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])


class PositionRandomPlanner:
    """ """

    def __init__(
        self, SIM: bool, head, pin_robot, IDX_PUSHER_TIP, q0, add_capsules=False
    ):
        """
        Notes:
        The low level force control will generate the torque commands
        """
        self.head = head
        self.joint_positions = self.head.get_sensor("joint_positions")
        self.joint_velocities = self.head.get_sensor("joint_velocities")

        self.pin_robot = pin_robot
        self.IDX_PUSHER_TIP = IDX_PUSHER_TIP
        self.start = self.pin_robot.data.oMf[IDX_PUSHER_TIP]
        # pin.framesForwardKinematics(self.pin_robot.model, self.pin_robot.data, q)

        self.low_level_force_control = Goal_controller_q(SIM, pin_robot)
        self.low_level_force_control.qvl = 5.0
        self.low_level_force_control.kvv = 10.0
        self.low_level_force_control.ff_kp = 0.1
        self.low_level_force_control.ff_kv = 0.1
        self.low_level_force_control.qvgoal = np.zeros(7)
        self.low_level_force_control.qgoal = q0
        self.q0 = np.copy(q0)

        self.mode = "initialized"

        self.center = np.zeros(3)

        # .rotation

        self.center[:2] = center
        self.center[2] = desired_z

        self.plan = []  # sequence of positions and velocities
        self.reference_max_vel = 0.1

        self.p_lb = np.array([lb[0], lb[1], 0.05])
        self.p_ub = np.array([ub[0], ub[1], 0.05])

        self.z_safe = 0.15
        self.p_lb_safe = np.array([lb[0], lb[1], 0.05])
        self.p_ub_safe = np.array([ub[0], ub[1], self.z_safe])

        self.goal = pin.SE3().Identity()
        self.default_rotation = default_rotation

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
        # self.q = np.zeros(7)
        # self.qv = np.zeros(7)

        self.last_hl_time = -1
        self.hl_time = 0.1  # Time step of high level planner (check if plan is going well, recompute if necessary)

        self.only_random = True

    def get_data(self):
        return {
            "ee_goal": self.plan[self.counter if self.counter < len(self.plan) else -1],
            "qgoal": self.low_level_force_control.qgoal,
            "qvgoal": self.low_level_force_control.qvgoal,
        }

    # def read_sensors(self, data):
    #     """ """
    #     self.sensor_object_position = np.copy(data.body("cube_small").xpos)
    #     self.sensor_object_rotation = np.copy(data.body("cube_small").xmat)

    # def read_robot_info(self,data):
    #     """
    #
    #     """
    #     self.q = np.copy(get_robot_joints(data))
    #     self.qv = np.copy(get_robot_vel(data))

    def warmup(self, thread):
        print("warmup -- done!")
        pass

    def run(self, thread_head):
        q = self.joint_positions
        qv = self.joint_velocities

        print("q", q)
        print("qv", qv)

        sim_time = thread_head.ti * 1.0 / config["ctrl_freq"]
        print("sim time", sim_time)

        #
        # We run the highlevel planner at very self.hl_time
        # High level planner generates and follows paths
        #

        if sim_time - self.last_hl_time > self.hl_time:

            print("inside  hl")
            recompute_plan = False

            if self.counter >= len(self.plan):
                recompute_plan = True

            if recompute_plan:

                print("recomputing the plan")
                old_mode = copy.deepcopy(self.mode)
                rgba = None
                self.plan = []
                self.counter = 0

                pin.framesForwardKinematics(
                    self.pin_robot.model, self.pin_robot.data, q
                )
                pusher = self.pin_robot.data.oMf[self.IDX_PUSHER_TIP]

                if self.only_random:

                    self.mode = "to_rand"
                    self.start = pin.SE3(pusher.rotation, pusher.translation)

                    # random goal
                    rand_goal = self.p_lb_small + np.random.rand(3) * (
                        self.p_ub_small - self.p_lb_small
                    )
                    print("rand goal is", rand_goal)

                    if old_mode == "initialized":
                        rand_goal = self.center

                    self.goal = pin.SE3(self.default_rotation, rand_goal)
                    rgba = np.array([0, 0, 1, 0.1])

                    total_time = (
                        np.linalg.norm(self.goal.translation - self.start.translation)
                        / self.reference_max_vel
                    )
                    self.times = np.linspace(
                        0, total_time, int(total_time / self.hl_time)
                    )

                    for time in self.times:
                        new_state = pin.SE3.Interpolate(
                            self.start, self.goal, time / total_time
                        )
                        self.plan.append(new_state)
                else:
                    raise NotImplemented("this planner only does random movements!")

                for i, s in enumerate(self.plan):
                    print(f"state {i} is {s}")

                if self.add_capsules:
                    raise NotImplemented("this planner cannot add capsules")
                    # point1 = self.plan[-1]
                    # # self.goal.translation
                    # point2 = copy.deepcopy(point1)
                    # point2.translation[2] += 0.01
                    #
                    # radius = 0.05
                    # print("adding capsule!!")
                    #
                    # add_visual_capsule(
                    #     scene, point1.translation, point2.translation, radius, rgba
                    # )

                qs = []
                init_guess_q = np.copy(self.q0)
                for p in self.plan:
                    # sov
                    # _p = np.copy(postion_pusher_tip_goal)
                    # _p[:2] = np.copy(p[:2])
                    # frame_goal = pin.SE3(r_pusher_tip_goal, _p)
                    _q, cost = solve_ik(self.pin_robot, p, IDX_PUSHER_TIP, init_guess_q)
                    init_guess_q = _q
                    # viz.display(_q)
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

        self.tau = self.low_level_force_control.get_u(q, qv)

        self.head.set_control("ctrl_joint_torques", self.tau)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Choose experiment, load config and import controller  #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
SIM = True
EXP_NAME = "random_move"


# # # # # # # # # # # #
# Import robot model  #
# # # # # # # # # # # #
pin_robot = load_pinocchio_wrapper("iiwa_pusher")
IDX_PUSHER_TIP = pin_robot.model.getFrameId("pusher_tip")


config = {}

config["q0"] = [
    0.5604080525397099,
    0.8670522246951389,
    0.10696889788203193,
    -1.8091919335298106,
    -0.17904272753850936,
    0.47469638420974486,
    0.7893769555580691,
]

config["dq0"] = [0, 0, 0, 0, 0, 0, 0]
config["ctrl_freq"] = 100
config["SOLVER"] = "rand"


def time_to_str_with_ms(_now):
    return "Time : %s.%s\n" % (
        time.strftime("%x %X", time.localtime(_now)),
        str("%.3f" % _now).split(".")[1],
    )  # Rounds to nearest millisecond


class EnvMJWrapper:
    """ """

    def __init__(self, robot_simulator, viewer=False):
        import mujoco
        import mujoco.viewer

        self.time_calls = []
        self.time_calls_raw = []
        self.dt = 0.001
        self.last_viewer_update = time.time()
        self.viewer_update_interval = 1.0 / 20.0
        self.robot_simulator = robot_simulator
        if viewer:
            self.viewer = mujoco.viewer.launch_passive(
                self.robot_simulator.mjmodel, self.robot_simulator.mjdata
            )
        else:
            self.viewer = None
        time.sleep(0.1)

    def step(self, sleep=None):
        """ """
        self.time_calls.append(time_to_str_with_ms(time.time()))
        self.time_calls_raw.append(time.time())
        self.robot_simulator.step_simulation()
        current_time = time.time()
        if sleep:
            time.sleep(sleep)
        if self.viewer:
            if current_time - self.last_viewer_update > self.viewer_update_interval:
                self.last_viewer_update = current_time
                self.viewer.sync()

    def close(self):
        if self.viewer:
            self.viewer.close()


# # # # # # # # # # # # #
# Setup control thread  #
# # # # # # # # # # # # #
if SIM:
    # Sim env + set initial state
    config["T_tot"] = 30

    robot_simulator = load_mujoco_wrapper("iiwa_pusher")
    env = EnvMJWrapper(robot_simulator, viewer=True)

    # env = BulletEnvWithGround(p.GUI)
    # robot_simulator = load_bullet_wrapper('iiwa')
    # env.add_robot(robot_simulator)

    # env.add_robot(robot_simulator)
    q_init = np.asarray(config["q0"])
    v_init = np.asarray(config["dq0"])
    robot_simulator.reset_state(q_init, v_init)
    robot_simulator.forward_robot(q_init, v_init)

    # <<<<< Customize your PyBullet environment here if necessary
    head = SimHead(robot_simulator, with_sliders=False)
# !!!!!!!!!!!!!!!!
# !! REAL ROBOT !!
# !!!!!!!!!!!!!!!!
else:
    config["T_tot"] = 400
    path = MiM_Robots["iiwa"].dgm_path
    print(path)
    head = dynamic_graph_manager_cpp_bindings.DGMHead(path)
    target = None
    env = None

# TODO: I don't know the robots falls down?
# TODO: just send the gravity compentation torque to the sim robot!!

ctrl = PositionRandomPlanner(
    SIM, head, pin_robot, IDX_PUSHER_TIP, q0=np.array(config["q0"]), add_capsules=False
)


thread_head = ThreadHead(
    1.0 / config["ctrl_freq"],  # dt.
    HoldPDControllerGravComp(head, robot_simulator, SIM, 5.0, 0.1, with_sliders=False),
    head,  # Heads to read / write from.
    [],
    env,  # Environment to step.
)

thread_head.switch_controllers(ctrl)
print("switch controllers done")

# import sys
# sys.exit()

# # # # # # # # #
# Data logging  #
# # # # # # # # # <<<<<<<<<<<<< Choose data save path & log config here (cf. launch_utils)
# prefix     = "/home/skleff/data_sqp_paper_croc2/constrained/circle/"
prefix = "/tmp/"
suffix = "_" + config["SOLVER"] + "_CODE_SPRINT"


LOG_FIELDS = [
    "joint_positions",
    "joint_velocities",
    "x_des",
    "tau",
    "tau_ff",
    "tau_gravity",
    "target_position_x",
    "target_position_y",
    "target_position_z",
    "lb",
    "ub",
]

# print(LOG_FIELDS)
# LOG_FIELDS = launch_utils.LOGS_NONE
# LOG_FIELDS = launch_utils.SSQP_LOGS_MINIMAL
# LOG_FIELDS = launch_utils.CSSQP_LOGS_MINIMAL


# # # # # # # # # # #
# Launch experiment #
# # # # # # # # # # #
if SIM:
    thread_head.start_logging(
        int(config["T_tot"]),
        prefix + EXP_NAME + "_SIM_" + str(datetime.now().isoformat()) + suffix + ".mds",
        LOG_FIELDS=LOG_FIELDS,
    )
    thread_head.sim_run_timed(int(config["T_tot"]))
    thread_head.stop_logging()
else:
    thread_head.start()
    thread_head.start_logging(
        50,
        prefix
        + EXP_NAME
        + "_REAL_"
        + str(datetime.now().isoformat())
        + suffix
        + ".mds",
        LOG_FIELDS=LOG_FIELDS,
    )

print("experiment done")

if type(env) == EnvMJWrapper:
    env.close()  # I want to close the mujoco viewer if opened

import matplotlib.pyplot as plt


if type(env) == EnvMJWrapper:
    times = np.array(env.time_calls_raw)
    times -= times[0]
    plt.plot(times, ".")
    plt.show()


_history_measurements = (head._history_measurements,)
import pdb

pdb.set_trace()


# robot_simulator.viewer.close()
# thread_head.plot_timing() # <<<<<<<<<<<<< Comment out to skip timings plot


from mim_data_utils import DataLogger, DataReader
from plots.plot_utils import SimpleDataPlotter

r = DataReader(thread_head.log_filename)

s = SimpleDataPlotter(dt=1.0 / config["ctrl_freq"])

ee_lb = r.data["lb"]
ee_ub = r.data["ub"]

from croco_mpc_utils.pinocchio_utils import get_p_

frameId = pin_robot.model.getFrameId("contact")
nq = pin_robot.model.nq
nv = pin_robot.model.nv

p_mea = get_p_(
    r.data["joint_positions"], pin_robot.model, pin_robot.model.getFrameId("contact")
)
p_des = get_p_(
    r.data["x_des"][:, :nq], pin_robot.model, pin_robot.model.getFrameId("contact")
)


N = r.data["absolute_time"].shape[0]

target_position = np.zeros((N, 3))
target_position[:, 0] = r.data["target_position_x"][:, 0]
target_position[:, 1] = r.data["target_position_y"][:, 0]
target_position[:, 2] = r.data["target_position_z"][:, 0]


s.plot_ee_pos(
    [p_mea, p_des, target_position, ee_lb, ee_ub],
    ["Measured", "Predicted", "Reference", "lb", "ub"],
    ["r", "b", "g", "k", "k"],
    linestyle=["solid", "solid", "dotted", "dotted", "dotted"],
    ylims=[[-0.8, -0.5, 0], [+0.8, +0.5, 1.5]],
)

plt.show()
