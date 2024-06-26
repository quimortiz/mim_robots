"""solo12wrapper

Solo12 pybullet interface using pinocchio's convention.

License: BSD 3-Clause License
Copyright (C) 2018-2019, New York University , Max Planck Gesellschaft
Copyright note valid unless otherwise stated in individual files.
All rights reserved.
"""
import numpy as np
import pybullet

from mim_robots.pybullet.wrapper import PinBulletWrapper
import pinocchio as pin
from mim_robots.robot_loader import load_pinocchio_wrapper

dt = 1e-3


class Solo12Robot(PinBulletWrapper):
    def __init__(
        self,
        robotinfo, 
        locked_joints_names=None, 
        qref=np.zeros(19),
        pos=None,
        orn=None,
        init_sliders_pose=4
        * [
            0,
        ],
    ):

        self.initial_configuration = (
                    [0.2, 0.0, 0.25, 0.0, 0.0, 0.0, 1.0]
                    + 2 * [0.0, 0.8, -1.6]
                    + 2 * [0.0, -0.8, 1.6]
                    )
        
        self.initial_velocity = (8 + 4 + 6) * [
            0,
        ]

        # Load the robot
        if pos is None:
            pos = [0.0, 0, 1.80]
        if orn is None:
            orn = pybullet.getQuaternionFromEuler([0, 0, 0])

        self.robotId = pybullet.loadURDF(
            robotinfo.urdf_path,
            pos,
            orn,
            flags=pybullet.URDF_USE_INERTIA_FROM_FILE,
            useFixedBase= robotinfo.fixed_base,
        )
        pybullet.getBasePositionAndOrientation(self.robotId)

        # Create the robot wrapper in pinocchio.
        # self.pin_robot = load_pinocchio_wrapper("solo12")  
        robot_full = load_pinocchio_wrapper("solo12")  

        # Query all the joints.
        num_joints = pybullet.getNumJoints(self.robotId)

        for ji in range(num_joints):
            pybullet.changeDynamics(
                self.robotId,
                ji,
                linearDamping=0.04,
                angularDamping=0.04,
                restitution=0.0,
                lateralFriction=0.5,
            )


        self.slider_a = pybullet.addUserDebugParameter(
            "a", 0, 1, init_sliders_pose[0]
        )
        self.slider_b = pybullet.addUserDebugParameter(
            "b", 0, 1, init_sliders_pose[1]
        )
        self.slider_c = pybullet.addUserDebugParameter(
            "c", 0, 1, init_sliders_pose[2]
        )
        self.slider_d = pybullet.addUserDebugParameter(
            "d", 0, 1, init_sliders_pose[3]
        )

        # List actuated joints
        actuated_joints_names = []
        for leg in ["FL", "FR", "HL", "HR"]:
            actuated_joints_names += [leg + "_HAA", leg + "_HFE", leg + "_KFE"]
        # Optionally reduce the model
        if(locked_joints_names is not None):
            self.pin_robot, controlled_joints_names = self.freeze_joints(locked_joints_names, robot_full, qref, actuated_joints_names)
        else:
            controlled_joints_names = actuated_joints_names
            self.pin_robot = robot_full

        self.base_link_name = "base_link"
        self.end_eff_ids = []
        self.end_effector_names = []

        # List end eff names and ids 
        for leg in ["FL", "FR", "HL", "HR"]:
            self.end_eff_ids.append(self.pin_robot.model.getFrameId(leg + "_FOOT"))
            self.end_effector_names.append(leg + "_FOOT")

        self.joint_names = controlled_joints_names
        self.nb_ee = len(self.end_effector_names)

        self.hl_index = self.pin_robot.model.getFrameId("HL_ANKLE")
        self.hr_index = self.pin_robot.model.getFrameId("HR_ANKLE")
        self.fl_index = self.pin_robot.model.getFrameId("FL_ANKLE")
        self.fr_index = self.pin_robot.model.getFrameId("FR_ANKLE")

        # Creates the wrapper by calling the super.__init__.
        super(Solo12Robot, self).__init__(
            self.robotId,
            self.pin_robot,
            controlled_joints_names,
            ["FL_ANKLE", "FR_ANKLE", "HL_ANKLE", "HR_ANKLE"],
            useFixedBase=robotinfo.fixed_base
        )

    def forward_robot(self, q=None, dq=None):
        if not q:
            q, dq = self.get_state()
        elif not dq:
            raise ValueError("Need to provide q and dq or non of them.")

        self.pin_robot.forwardKinematics(q, dq)
        self.pin_robot.computeJointJacobians(q)
        self.pin_robot.framesForwardKinematics(q)
        self.pin_robot.centroidalMomentum(q, dq)

    def get_slider_position(self, letter):
        try:
            if letter == "a":
                return pybullet.readUserDebugParameter(self.slider_a)
            if letter == "b":
                return pybullet.readUserDebugParameter(self.slider_b)
            if letter == "c":
                return pybullet.readUserDebugParameter(self.slider_c)
            if letter == "d":
                return pybullet.readUserDebugParameter(self.slider_d)
        except Exception:
            # In case of not using a GUI.
            return 0.

    def reset_to_initial_state(self) -> None:
        """Reset robot state to the initial configuration (based on Solo12Config)."""
        q0 = np.matrix(self.initial_configuration).T
        dq0 = np.matrix(self.initial_velocity).T
        self.reset_state(q0, dq0)

class Solo12RobotWithoutPybullet():
    """
    Similar to the class above, but without PyBullet. Used for ROS + Gazebo projects
    """
    def __init__(self):


        self.initial_configuration = (
        [0.2, 0.0, 0.25, 0.0, 0.0, 0.0, 1.0]
        + 2 * [0.0, 0.8, -1.6]
        + 2 * [0.0, -0.8, 1.6]
        )
        
        self.initial_velocity = (8 + 4 + 6) * [
            0,
        ]

        # Create the robot wrapper in pinocchio.
        self.pin_robot = load_pinocchio_wrapper("solo12")  

        self.base_link_name = "base_link"
        self.end_eff_ids = []
        self.end_effector_names = []
        controlled_joints = []

        for leg in ["FL", "FR", "HL", "HR"]:
            controlled_joints += [leg + "_HAA", leg + "_HFE", leg + "_KFE"]
            self.end_eff_ids.append(
                self.pin_robot.model.getFrameId(leg + "_FOOT")
            )
            self.end_effector_names.append(leg + "_FOOT")

        self.joint_names = controlled_joints
        self.nb_ee = len(self.end_effector_names)

        self.hl_index = self.pin_robot.model.getFrameId("HL_ANKLE")
        self.hr_index = self.pin_robot.model.getFrameId("HR_ANKLE")
        self.fl_index = self.pin_robot.model.getFrameId("FL_ANKLE")
        self.fr_index = self.pin_robot.model.getFrameId("FR_ANKLE")

    def forward_robot(self, q=None, dq=None):
        if q is None:
            q, dq = self.get_state()
        elif dq is None:
            raise ValueError("Need to provide q and dq or non of them.")

        self.pin_robot.forwardKinematics(q, dq)
        self.pin_robot.computeJointJacobians(q)
        self.pin_robot.framesForwardKinematics(q)
        self.pin_robot.centroidalMomentum(q, dq)

    def reset_to_initial_state(self) -> None:
        """Reset robot state to the initial configuration (based on Solo12Config)."""
        q0 = np.array(self.initial_configuration)
        dq0 = np.array(self.initial_velocity)
        self.reset_state(q0, dq0)
    
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
        self.pin_robot.forwardKinematics(q, dq)
        self.pin_robot.computeJointJacobians(q)
        self.pin_robot.framesForwardKinematics(q)
        self.pin_robot.centroidalMomentum(q, dq)
