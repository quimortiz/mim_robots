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

import pdb
pdb.set_trace()

lb = np.array([0.35, -0.30])
ub = np.array([0.72, +0.30])

center = (lb + ub) / 2.0
desired_z = 0.05
default_rotation = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])


class PushRandomPlanner:
    """ """

    def __init__(
        self,
        SIM: bool,
        robot_head,
        pin_robot,
        IDX_PUSHER_TIP,
        q0,
        mujoco_viewer=None,
        add_capsules=False,
        perception_head=None,
    ):
        """
        Notes:
        The low level force control will generate the torque commands
        """
        self.mujoco_viewer = mujoco_viewer
        self.add_capsules = add_capsules
        self.robot_head = robot_head
        self.perception_head = perception_head

        self.joint_positions = self.robot_head.get_sensor("joint_positions")
        self.joint_velocities = self.robot_head.get_sensor("joint_velocities")
        self.object_position = self.perception_head.get_sensor("object_position")
        self.object_rotation = self.perception_head.get_sensor("object_rotation")
        self.object_velocity = self.perception_head.get_sensor("object_velocity")

        self.q0 = np.copy(q0)
        self.pin_robot = pin_robot
        self.IDX_PUSHER_TIP = IDX_PUSHER_TIP
        pin.framesForwardKinematics(self.pin_robot.model, self.pin_robot.data, self.q0)
        self.start = self.pin_robot.data.oMf[self.IDX_PUSHER_TIP]

        self.low_level_force_control = Goal_controller_q(SIM, pin_robot)
        self.low_level_force_control.qvl = 5.0
        self.low_level_force_control.kvv = 10.0
        self.low_level_force_control.ff_kp = 0.1
        self.low_level_force_control.ff_kv = 0.1
        self.low_level_force_control.qvgoal = np.zeros(7)
        self.low_level_force_control.qgoal = np.copy(q0)

        self.mode = "initialized"

        self.center = np.zeros(3)

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

        self.p_lb_small = self.p_lb + 0.3 * (self.p_ub - self.p_lb)
        self.p_ub_small = self.p_ub - 0.3 * (self.p_ub - self.p_lb)

        self.counter = 0
        self.dist_fail_controller = 0.2  # in cartesian space

        self.sensor_object_position = np.zeros(3)
        self.sensor_object_rotation = np.eye(3)

        self.last_hl_time = -1
        self.hl_time = 0.1
        self.only_random = True

    # def get_data(self):
    #     return {
    #         "ee_goal": self.plan[self.counter if self.counter < len(self.plan) else -1],
    #         "qgoal": self.low_level_force_control.qgoal,
    #         "qvgoal": self.low_level_force_control.qvgoal,
    #     }

    def warmup(self, thread):
        print("warmup -- done!")
        pass

    def recompute_plan_fn(self):
        """ """
        q = self.joint_positions
        _op = self.object_position

        pin.framesForwardKinematics(self.pin_robot.model, self.pin_robot.data, q)
        pe = self.pin_robot.data.oMf[self.IDX_PUSHER_TIP]

        cube = _op["cube_small"]

        is_outside = np.any(cube[:2] < self.p_lb_small[:2]) or np.any(
            cube[:2] > self.p_ub_small[:2]
        )

        old_mode = copy.deepcopy(self.mode)
        rgba = None
        self.plan = []
        self.counter = 0

        pin.framesForwardKinematics(self.pin_robot.model, self.pin_robot.data, q)
        pusher = self.pin_robot.data.oMf[self.IDX_PUSHER_TIP]

        if is_outside:
            self.mode = "recovery"
            rel_cube = cube - self.center

            print("cube is outside!")
            print("entering recovery mode")

            goal_0 = pusher.translation
            goal_0[2] = self.z_safe

            # Intersect the line rel_cube with the bounds (in 2D)
            p = self.center[:2]
            v = rel_cube[:2]

            intersect, tmin, tmax = ray_cube_intersection(p, v, lb, ub)

            assert intersect == True
            assert tmin < tmax
            assert tmin == 0
            intersection_point = p + tmax * v

            # goal_0 = np.array(
            #
            #     [intersection_point[0], intersection_point[1], self.z_safe]
            #
            # )
            goal_1 = np.array(
                [intersection_point[0], intersection_point[1], self.z_safe]
            )

            goal_2 = np.array([intersection_point[0], intersection_point[1], desired_z])

            goal_3 = self.center + 0.2 * rel_cube
            _big_goals = [goal_0, goal_1, goal_2, goal_3]

            _big_goals = [
                np.clip(g, self.p_lb_safe, self.p_ub_safe) for g in _big_goals
            ]

            big_goals = [pin.SE3(self.default_rotation, g) for g in _big_goals]

            if self.mujoco_viewer and self.add_capsules:
                scene = self.mujoco_viewer._user_scn
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
                    start = pin.SE3(self.default_rotation, pusher.translation)
                else:
                    start = big_goals[i - 1]
                total_time = (
                    np.linalg.norm(big_goals[i].translation - start.translation)
                    / self.reference_max_vel
                )
                times = np.linspace(0, total_time, int(total_time / self.hl_time))

                print(f"i is {i}")
                for time in times:
                    new_state = pin.SE3.Interpolate(
                        start, big_goals[i], time / total_time
                    )

                    print(f"adding new states {new_state}")
                    self.plan.append(new_state)

                print("new high level mode", self.mode)
                print("len(self.plan)", len(self.plan))
                rgba = np.array([1, 1, 0, 0.1])

        elif (
            old_mode == "to_obj" or old_mode == "initialized" or old_mode == "recovery"
        ):
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
            translation = cube + self.penetration * cube_to_ball / np.linalg.norm(
                cube_to_ball
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
                new_state = pin.SE3.Interpolate(start, goal, time / total_time)
                self.plan.append(new_state)

        for i, s in enumerate(self.plan):
            print(f"state {i} is {s}")

        if self.add_capsules and self.mujoco_viewer:
            # raise NotImplemented("this planner cannot add capsules")
            point1 = self.plan[-1]
            # self.goal.translation
            point2 = copy.deepcopy(point1)
            point2.translation[2] += 0.01
            radius = 0.05
            print("adding capsule!!")

            scene = self.mujoco_viewer._user_scn
            add_visual_capsule(
                scene, point1.translation, point2.translation, radius, rgba
            )

        qs = []
        init_guess_q = np.copy(self.q0)
        for p in self.plan:
            _q, _ = solve_ik(self.pin_robot, p, self.IDX_PUSHER_TIP, init_guess_q)
            init_guess_q = _q
            qs.append(_q)
        self.qs = qs

        dt = self.hl_time  # in seconds
        self.qs_vel = np.diff(qs, axis=0) / dt

    def run(self, thread_head):
        q = self.joint_positions
        qv = self.joint_velocities

        _op = self.object_position
        _or = self.object_rotation
        _ov = self.object_velocity

        pin.framesForwardKinematics(self.pin_robot.model, self.pin_robot.data, q)
        pe = self.pin_robot.data.oMf[self.IDX_PUSHER_TIP]

        sim_time = thread_head.ti * 1.0 / config["ctrl_freq"]
        print("sim time", sim_time)
        print("Random push controller")

        if sim_time - self.last_hl_time > self.hl_time:

            print("A")
            cube = _op["cube_small"]
            print("B")

            is_outside = np.any(cube[:2] < self.p_lb_small[:2]) or np.any(
                cube[:2] > self.p_ub_small[:2]
            )

            is_cube_outside_workspace = np.any(cube[:2] < self.p_lb[:2] - .2) or np.any(
                cube[:2] > self.p_ub_small[:2] + .2)

            if is_cube_outside_workspace: 
                raise ValueError("cube is outside the workspace. Stopping controller")

            print("D")
            recompute_plan = False

            if self.mode == "initialized":
                recompute_plan = True
                print("D1")
            elif self.counter >= len(self.plan):
                print("D4")
                recompute_plan = True
            elif len(self.plan) > 0:
                print("D2")
                xdes = self.plan[self.counter if self.counter < len(self.plan) else -1]
                print("D4")
                tracking_error = np.linalg.norm(pe.translation - xdes.translation)
                print("D5")
                if tracking_error > self.dist_fail_controller:
                    print(f"high tracking error {tracking_error} -- recompute plan")
                    recompute_plan = True
            elif is_outside and not self.mode == "recovery":
                print("D3")
                recompute_plan = True

            print("B")
            if recompute_plan:
                self.recompute_plan_fn()
                print("C")
                self.counter = 0

            print(len(self.qs))
            print(self.counter)
            self.low_level_force_control.qgoal = self.qs[self.counter]
            self.low_level_force_control.qvgoal = self.qs_vel[
                self.counter if self.counter < len(self.qs_vel) else -1
            ]
            self.counter += 1
            self.last_hl_time = sim_time

        self.tau = self.low_level_force_control.get_u(q, qv)

        self.robot_head.set_control("ctrl_joint_torques", self.tau)


class PositionRandomPlanner:
    """ """

    def __init__(
        self,
        SIM: bool,
        robot_head,
        pin_robot,
        IDX_PUSHER_TIP,
        q0,
        mujoco_viewer=None,
        add_capsules=False,
        perception_head=None,
    ):
        """
        Notes:
        The low level force control will generate the torque commands
        """
        self.mujoco_viewer = mujoco_viewer
        self.add_capsules = add_capsules
        self.robot_head = robot_head
        self.perception_head = perception_head

        self.joint_positions = self.robot_head.get_sensor("joint_positions")
        self.joint_velocities = self.robot_head.get_sensor("joint_velocities")
        self.object_position = self.perception_head.get_sensor("object_position")
        self.object_rotation = self.perception_head.get_sensor("object_rotation")
        self.object_velocity = self.perception_head.get_sensor("object_velocity")

        self.q0 = np.copy(q0)
        self.pin_robot = pin_robot
        self.IDX_PUSHER_TIP = IDX_PUSHER_TIP
        pin.framesForwardKinematics(self.pin_robot.model, self.pin_robot.data, self.q0)
        self.start = self.pin_robot.data.oMf[IDX_PUSHER_TIP]

        self.low_level_force_control = Goal_controller_q(SIM, pin_robot)
        self.low_level_force_control.qvl = 5.0
        self.low_level_force_control.kvv = 10.0
        self.low_level_force_control.ff_kp = 0.1
        self.low_level_force_control.ff_kv = 0.1
        self.low_level_force_control.qvgoal = np.zeros(7)
        self.low_level_force_control.qgoal = np.copy(q0)

        self.mode = "initialized"

        self.center = np.zeros(3)

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

        self.p_lb_small = self.p_lb + 0.3 * (self.p_ub - self.p_lb)
        self.p_ub_small = self.p_ub - 0.3 * (self.p_ub - self.p_lb)

        self.counter = 0
        self.dist_fail_controller = 0.2  # in cartesian space

        self.sensor_object_position = np.zeros(3)
        self.sensor_object_rotation = np.eye(3)

        self.last_hl_time = -1
        self.hl_time = 0.1
        self.only_random = True

    def get_data(self):
        return {
            "ee_goal": self.plan[self.counter if self.counter < len(self.plan) else -1],
            "qgoal": self.low_level_force_control.qgoal,
            "qvgoal": self.low_level_force_control.qvgoal,
        }

    def warmup(self, thread):
        print("warmup -- done!")
        pass

    def recompute_plan_fn(self):
        """ """
        q = self.joint_positions
        old_mode = copy.deepcopy(self.mode)
        rgba = None
        self.plan = []
        self.counter = 0

        pin.framesForwardKinematics(self.pin_robot.model, self.pin_robot.data, q)
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
            self.times = np.linspace(0, total_time, int(total_time / self.hl_time))

            for time in self.times:
                new_state = pin.SE3.Interpolate(
                    self.start, self.goal, time / total_time
                )
                self.plan.append(new_state)
        else:
            raise NotImplemented("this planner only does random movements!")

        for i, s in enumerate(self.plan):
            print(f"state {i} is {s}")

        if self.add_capsules and self.mujoco_viewer:
            # raise NotImplemented("this planner cannot add capsules")
            point1 = self.plan[-1]
            # self.goal.translation
            point2 = copy.deepcopy(point1)
            point2.translation[2] += 0.01
            radius = 0.05
            print("adding capsule!!")

            scene = self.mujoco_viewer._user_scn
            add_visual_capsule(
                scene, point1.translation, point2.translation, radius, rgba
            )

        qs = []
        init_guess_q = np.copy(self.q0)
        for p in self.plan:
            _q, cost = solve_ik(self.pin_robot, p, IDX_PUSHER_TIP, init_guess_q)
            init_guess_q = _q
            qs.append(_q)
        self.qs = qs

        dt = self.hl_time  # in seconds
        self.qs_vel = np.diff(qs, axis=0) / dt

    def run(self, thread_head):
        q = self.joint_positions
        qv = self.joint_velocities

        if self.perception_head:
            _op = self.object_position
            _or = self.object_rotation
            _ov = self.object_velocity

        sim_time = thread_head.ti * 1.0 / config["ctrl_freq"]
        print("sim time", sim_time)

        if sim_time - self.last_hl_time > self.hl_time:

            print("inside  hl")
            recompute_plan = False

            if self.counter >= len(self.plan):
                recompute_plan = True

            if recompute_plan:
                self.recompute_plan_fn()
                self.counter = 0

            self.low_level_force_control.qgoal = self.qs[self.counter]
            self.low_level_force_control.qvgoal = self.qs_vel[
                self.counter if self.counter < len(self.qs_vel) else -1
            ]
            self.counter += 1
            self.last_hl_time = sim_time

        self.tau = self.low_level_force_control.get_u(q, qv)

        self.robot_head.set_control("ctrl_joint_torques", self.tau)


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


class Perception_head_sim:
    """ """

    def __init__(self, mujoco_env, objects, delay_dt=0, noise_data_std={}):
        """
        Args:
            objects: List of objects to track.
            noise_model: Std of the noise to apply to the measurements.
        """
        self.objects = objects
        self.mujoco_env = mujoco_env

        self.update_delay(delay_dt)
        self.update_noise_data(noise_data_std)

        self.use_delay = True
        self.use_noise_model = True

        num_objects = len(self.objects)
        # self._sensor_object_position = np.zeros((num_objects,3))
        # self._sensor_object_rotation = np.zeros((num_objects,9))
        # self._sensor_object_velocity = np.zeros((num_objects,6))
        self._sensor_object_position = {}
        self._sensor_object_rotation = {}
        self._sensor_object_velocity = {}

    def get_sensor(self, sensor_name):
        return self.__dict__["_sensor_" + sensor_name]

    def read(self):
        for _object in self.objects:
            self._sensor_object_position[_object] = np.copy(
                self.mujoco_env.robot_simulator.mjdata.body(_object).xpos
            )
            self._sensor_object_rotation[_object] = np.copy(
                self.mujoco_env.robot_simulator.mjdata.body(_object).xmat
            ).flatten()
            self._sensor_object_velocity[_object] = np.copy(
                self.mujoco_env.robot_simulator.mjdata.body(_object).cvel
            )

    def write(self):
        """
        We do not write data into the Perception head
        """
        pass

    def sim_step(self):
        """
        Simulation is managed by the SIM head or the real environment
        """
        pass

    def update_delay(self, delay_dt):
        # self.delay_dt = delay_dt
        # length = delay_dt + 1
        #
        # self.fill_history = True
        #
        # # For each object, setup a hisotry for position and body velocity.
        # self.history = {}
        # for i, obj in enumerate(self.objects):
        #     self.history[obj] = {
        #         'position': np.zeros((length, 7)),
        #         'body_velocity': np.zeros((length, 6))
        #     }
        if delay_dt > 1e-6:
            raise ValueError("Not supporting delay right now.")

    def update_noise_data(self, noise_data_std={}):
        if not (noise_data_std is None or noise_data_std == {}):
            raise ValueError("Not supporting noise right now.")
        # raise ValueError("Not supporting noise right now.")
        # self.noise_data_std = noise_data_std
        # if not 'position_xyzrpy' in noise_data_std:
        #     self.noise_data_std['position_xyzrpy'] = np.zeros(6)
        # if not 'body_velocity' in noise_data_std:
        #     self.noise_data_std['body_velocity'] = np.zeros(6)

    def apply_noise_model(self, pos, vel):
        raise ValueError("Not supporting noise right now.")
        # def sample_noise(entry):
        #     noise_var = self.noise_data_std[entry]**2
        #     return np.random.multivariate_normal(np.zeros_like(noise_var), np.diag(noise_var))
        #
        # noise_pos = sample_noise('position_xyzrpy')
        #
        # se3 = pin.XYZQUATToSE3(pos)
        # se3.translation += noise_pos[:3]
        # se3.rotation = se3.rotation @ pin.rpy.rpyToMatrix(*noise_pos[3:])
        #
        # pos = pin.SE3ToXYZQUAT(se3)
        # vel += sample_noise('body_velocity')
        # return pos, vel

    # def update(self, thread_head):
    #     # Write the position and velocity to the history buffer for each
    #     # tracked object.
    #     history = self.history
    #     self.write_idx = thread_head.ti % (self.delay_dt + 1)
    #     self.read_idx = (thread_head.ti + 1) % (self.delay_dt + 1)
    #
    #     for i, obj in enumerate(self.objects):
    #         robot, frame = obj.split('/')
    #         assert robot == frame, "Not supporting other frames right now."
    #
    #         # Seek the head with the vicon object.
    #         for name, head in thread_head.heads.items():
    #             if head._vicon_name == robot:
    #                 pos, vel = self.apply_noise_model(
    #                     head._sensor__vicon_base_position.copy(),
    #                     head._sensor__vicon_base_velocity.copy())
    #
    #                 # At the first timestep, filll the full history.
    #                 if self.fill_history:
    #                     self.fill_history = False
    #                     history[obj]['position'][:] = pos
    #                     history[obj]['body_velocity'][:] = vel
    #                 else:
    #                     history[obj]['position'][self.write_idx] = pos
    #                     history[obj]['body_velocity'][self.write_idx] = vel
    #
    #                 self.vicon_frames[obj] = {
    #                     'idx': i,
    #                     'head': head,
    #                 }

    # def get_state(self, vicon_object):
    #     pos = self.history[vicon_object]['position'][self.read_idx]
    #     vel = self.history[vicon_object]['body_velocity'][self.read_idx]
    #     # pos[:3] -= self.bias_xyz[self.vicon_frames[vicon_object]['idx']]
    #     return (pos, vel)

    # def reset_bias(self, vicon_object):
    #     self.bias_xyz[self.vicon_frames[vicon_object]['idx']] = 0

    # def bias_position(self, vicon_object):
    #     pos = self.history[vicon_object]['position'][self.read_idx]
    #     self.bias_xyz[self.vicon_frames[vicon_object]['idx'], :2] = pos[:2].copy()


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
    perception_head = Perception_head_sim(env, ["cube_small"])


# !!!!!!!!!!!!!!!!
# !! REAL ROBOT !!
# !!!!!!!!!!!!!!!!
else:
    config["T_tot"] = 400
    path = MiM_Robots["iiwa"].dgm_path
    head = dynamic_graph_manager_cpp_bindings.DGMHead(path)
    target = None
    env = None

# TODO: I don't know the robots falls down?
# TODO: just send the gravity compentation torque to the sim robot!!


Planner = PushRandomPlanner

ctrl = Planner(
    SIM,
    head,
    pin_robot,
    IDX_PUSHER_TIP,
    q0=np.array(config["q0"]),
    mujoco_viewer=env.viewer if SIM else None,
    perception_head=perception_head,
    add_capsules=True if SIM else None,
)


thread_head = ThreadHead(
    1.0 / config["ctrl_freq"],  # dt.
    HoldPDControllerGravComp(head, robot_simulator, SIM, 5.0, 0.1, with_sliders=False),
    {"robot_head": head, "perception_head": perception_head},
    # head,  # Heads to read / write from.
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

thread_head.plot_timing() # <<<<<<<<<<<<< Comment out to skip timings plot


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
