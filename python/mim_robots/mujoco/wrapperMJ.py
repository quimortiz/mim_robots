"""wrapper

Pybullet interface using pinocchio's convention.

License: BSD 3-Clause License
Copyright (C) 2018-2019, New York University , Max Planck Gesellschaft
Copyright note valid unless otherwise stated in individual files.
All rights reserved.
"""

import pybullet
import pinocchio
import numpy as np
from numpy.random import default_rng
from time import sleep
from pinocchio.utils import zero
import pinocchio as pin
import mujoco

class PinMujocoWrapper(object):
    """[summary]

    Attributes:
        nq (int): Dimension of the generalized coordiantes.
        nv (int): Dimension of the generalized velocities.
        nj (int): Number of joints.
        nf (int): Number of end-effectors.
        robot_id (int): PyBullet id of the robot.
        pinocchio_robot (:obj:'Pinocchio.RobotWrapper'): Pinocchio RobotWrapper for the robot.
        useFixedBase (bool): Determines if the robot base if fixed.
        nb_dof (int): The degrees of freedom excluding the base.
        joint_names (:obj:`list` of :obj:`str`): Names of the joints.
        endeff_names (:obj:`list` of :obj:`str`): Names of the end-effectors.
    """

    def __init__(
        self, mjmodel, mjdata, pinocchio_robot, joint_names, endeff_names, useFixedBase=False
    ):
        """Initializes the wrapper.

        Args:
            robot_id (int): PyBullet id of the robot.
            pinocchio_robot (:obj:'Pinocchio.RobotWrapper'): Pinocchio RobotWrapper for the robot.
            joint_names (:obj:`list` of :obj:`str`): Names of the joints.
            endeff_names (:obj:`list` of :obj:`str`): Names of the end-effectors.
            useFixedBase (bool, optional): Determines if the robot base if fixed.. Defaults to False.
        """
        self.nq = pinocchio_robot.nq
        self.nv = pinocchio_robot.nv
        self.nj = len(joint_names)
        self.nf = len(endeff_names)
        self.mjmodel = mjmodel
        self.mjdata = mjdata
        self.pin_robot = pinocchio_robot
        self.useFixedBase = useFixedBase
        self.nb_dof = self.nv - 6

        self.joint_names = joint_names
        self.endeff_names = endeff_names

        self.base_linvel_prev = None
        self.base_angvel_prev = None
        self.base_linacc = np.zeros(3)
        self.base_angacc = np.zeros(3)

        # IMU pose offset in base frame
        self.rot_base_to_imu = np.identity(3)
        self.r_base_to_imu = np.array([0.10407, -0.00635, 0.01540])

        self.rng = default_rng()

        self.base_imu_accel_bias = np.zeros(3)
        self.base_imu_gyro_bias = np.zeros(3)
        self.base_imu_accel_thermal = np.zeros(3)
        self.base_imu_gyro_thermal = np.zeros(3)
        self.base_imu_accel_thermal_noise = 0.0001962  # m/(sec^2*sqrt(Hz))
        self.base_imu_gyro_thermal_noise = 0.0000873  # rad/(sec*sqrt(Hz))
        self.base_imu_accel_bias_noise = 0.0001  # m/(sec^3*sqrt(Hz))
        self.base_imu_gyro_bias_noise = 0.000309  # rad/(sec^2*sqrt(Hz))

        # bullet_joint_map = {}
        # for ji in range(pybullet.getNumJoints(robot_id)):
        #     bullet_joint_map[
        #         pybullet.getJointInfo(robot_id, ji)[1].decode("UTF-8")
        #     ] = ji
        #
        # self.bullet_joint_ids = np.array(
        #     [bullet_joint_map[name] for name in joint_names]
        # )
        # self.pinocchio_joint_ids = np.array(
        #     [pinocchio_robot.model.getJointId(name) for name in joint_names]
        # )

        # self.pin2bullet_joint_only_array = []
        #
        # if not self.useFixedBase:
        #     for i in range(2, self.nj + 2):
        #         self.pin2bullet_joint_only_array.append(
        #             np.where(self.pinocchio_joint_ids == i)[0][0]
        #         )
        # else:
        #     for i in range(1, self.nj + 1):
        #         self.pin2bullet_joint_only_array.append(
        #             np.where(self.pinocchio_joint_ids == i)[0][0]
        #         )

        # Disable the velocity control on the joints as we use torque control.
        
        # pybullet.setJointMotorControlArray(
        #     robot_id,
        #     self.bullet_joint_ids,
        #     pybullet.VELOCITY_CONTROL,
        #     forces=np.zeros(self.nj),
        # )

        # In pybullet, the contact wrench is measured at a joint. In our case
        # the joint is fixed joint. Pinocchio doesn't add fixed joints into the joint
        # list. Therefore, the computation is done wrt to the frame of the fixed joint.
        
        # self.bullet_endeff_ids = [bullet_joint_map[name] for name in endeff_names]
        # self.pinocchio_endeff_ids = [
        #     pinocchio_robot.model.getFrameId(name) for name in endeff_names
        # ]
        # #
        # self.nb_contacts = len(self.pinocchio_endeff_ids)
        # self.contact_status = np.zeros(self.nb_contacts)
        # self.contact_forces = np.zeros([self.nb_contacts, 6])

    def freeze_joints(self, locked_joints_names, robot_full, qref, actuated_joints_names = None):
        raise NotImplementedError("not implemented in Mujoco Wrapper")

    def get_force(self):
        """Returns the force readings as well as the set of active contacts
        Returns:
            (:obj:`list` of :obj:`int`): List of active contact frame ids.
            (:obj:`list` of np.array((6,1))) List of active contact forces.
        """
        raise NotImplementedError("not implemented in Mujoco Wrapper")


    def end_effector_forces(self):
        """Returns the forces and status for all end effectors

        Returns:
            (:obj:`list` of :obj:`int`): list of contact status for each end effector.
            (:obj:`list` of np.array(6)): List of force wrench at each end effector
        """
        raise NotImplementedError("not implemented in Mujoco Wrapper")

    def get_base_velocity_world(self):
        """Returns the velocity of the base in the world frame.

        Returns:
            np.array((6,1)) with the translation and angular velocity
        """
        raise NotImplementedError("not implemented in Mujoco Wrapper")

    def get_base_acceleration_world(self):
        """Returns the numerically-computed acceleration of the base in the world frame.

        Returns:
            np.array((6,1)) vector of linear and angular acceleration
        """
        return np.concatenate((self.base_linacc, self.base_angacc))

    def get_base_imu_angvel(self):
        """Returns simulated base IMU gyroscope angular velocity.

        Returns:
            np.array((3,1)) IMU gyroscope angular velocity (base frame)
        """
        raise NotImplementedError("not implemented in Mujoco Wrapper")

    def get_base_imu_linacc(self):
        """Returns simulated base IMU accelerometer acceleration.

        Returns:
            np.array((3,1)) IMU accelerometer acceleration (base frame, gravity offset)
        """
        raise NotImplementedError("not implemented in Mujoco Wrapper")

    def get_state(self):
        """Returns a pinocchio-like representation of the q, dq matrices. Note that the base velocities are expressed in the base frame.

        Returns:
            ndarray: Generalized positions.
            ndarray: Generalized velocities.
        """

        # q = zero(self.nq)
        # dq = zero(self.nv)
        # joints = ["A1", "A2", "A3", "A4", "A5", "A6", "A7"]
        q = np.array([self.mjdata.joint(j).qpos[0] for j in self.joint_names])
        dq = np.array([self.mjdata.joint(j).qvel[0] for j in self.joint_names])
        return q, dq

    def get_imu_frame_position_velocity(self):
        """Returns the position and velocity of IMU frame. Note that the velocity is expressed in the IMU frame.

        Returns:
            np.array((3,1)): IMU frame position expressed in world.
            np.array((3,1)): IMU frame velocity expressed in IMU frame.
        """
        raise NotImplementedError("not implemented in Mujoco Wrapper")

    def update_pinocchio(self, q, dq):
        """Updates the pinocchio robot.

        This includes updating:
        - kinematics
        - joint and frame jacobian
        - centroidal momentum

        Args:
          q: Pinocchio generalized position vector.
          dq: Pinocchio generalize velocity vector.
        """
        self.pin_robot.computeJointJacobians(q)
        self.pin_robot.framesForwardKinematics(q)
        self.pin_robot.centroidalMomentum(q, dq)

    def get_state_update_pinocchio(self):
        """Get state from pybullet and update pinocchio robot internals.

        This gets the state from the pybullet simulator and forwards
        the kinematics, jacobians, centroidal moments on the pinocchio robot
        (see forward_pinocchio for details on computed quantities)."""
        q, dq = self.get_state()
        self.update_pinocchio(q, dq)
        return q, dq

    def reset_state(self, q, dq):
        """Reset the robot to the desired states.

        Args:
            q (ndarray): Desired generalized positions.
            dq (ndarray): Desired generalized velocities.
        """

        for name,x,vx in zip(self.joint_names, q, dq):
            self.mjdata.joint(name).qpos[0] = x
            self.mjdata.joint(name).qvel[0] = vx

    def send_joint_command(self, tau):
        """Apply the desired torques to the joints.

        Args:
            tau (ndarray): Torque to be applied.
        """
        self.mjdata.ctrl = tau

    def step_simulation(self):
        """Step the simulation forward."""
        # pybullet.stepSimulation()
        mujoco.mj_step(self.mjmodel, self.mjdata)


    def compute_numerical_quantities(self, dt):
        """Compute numerical robot quantities from simulation results.

        Args:
            dt (float): Length of the time step.
        """
        raise NotImplementedError("not implemented in Mujoco Wrapper")

    def print_physics_params(self):
        """Print physics engine parameters."""
        raise NotImplementedError("not implemented in Mujoco Wrapper")

    def _action(self, pos, rot):
        """Generate the adjoint from translation and rotation.

        Args:
            pos (np.array(3, )): Translation vector.
            rot (np.array(3, 3)): Rotation matrix.

        Returns:
            np.array(3, 3): The adjoint of the given translation and rotation.
        """
        res = np.zeros((6, 6))
        res[:3, :3] = rot
        res[3:, 3:] = rot
        res[3:, :3] = pinocchio.utils.skew(np.array(pos)).dot(rot)
        return res
