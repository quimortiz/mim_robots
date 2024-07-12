import numpy as np
import math
import pinocchio as pin
import cv2
import mujoco


def ray_cube_intersection( p: np.ndarray, v: np.ndarray, lb: np.ndarray, ub: np.ndarray):
    """
    Intersection test

    Notes:
    If the point is inside the box, the intersection is in tmax
    """
    tmin = 0.0 
    tmax = np.inf

    for d in range(2):
        t1 = (lb[d] - p[d]) / v[d]
        t2 = (ub[d] - p[d]) / v[d]
        tmin = min(max(t1, tmin), max(t2, tmin))
        tmax = max(min(t1, tmax), min(t2, tmax))

    return tmin <= tmax, tmin, tmax



def solve_ik(robot, oMgoal: pin.SE3, idx_ee: int, q0: np.ndarray):
    """ """
    num_iterations = 10
    alpha_0 = 1.0
    rate_alpha = 0.5
    min_alpha = 1e-4
    decrease_coeff = 1e-4

    q = np.copy(q0)
    f = math.inf

    for it in range(num_iterations):

        # Evaluate error at current configuration
        pin.framesForwardKinematics(robot.model, robot.data, q)
        oMtool = robot.data.oMf[idx_ee]
        tool_nu = pin.log(oMtool.inverse() * oMgoal).vector
        error = tool_nu @ tool_nu

        # Get jacobian and descent direction
        tool_Jtool = pin.computeFrameJacobian(
            robot.model, robot.data, q, idx_ee
        )
        vq = np.linalg.pinv(tool_Jtool) @ tool_nu

        # Start line search
        alpha = alpha_0
        f = error

        q_ls = pin.integrate(robot.model, q, vq * alpha)
        pin.framesForwardKinematics(robot.model, robot.data, q_ls)
        tool_nu_ls = pin.log(oMtool.inverse() * oMgoal).vector
        f_ls = tool_nu_ls @ tool_nu_ls

        while f_ls > f - decrease_coeff * alpha * vq @ vq:
            alpha *= rate_alpha
            if alpha < min_alpha:
                break
            q_ls = pin.integrate(robot.model, q, vq * alpha)
            pin.framesForwardKinematics(robot.model, robot.data, q_ls)
            tool_nu_ls = pin.log(oMtool.inverse() * oMgoal).vector
            f_ls = tool_nu_ls @ tool_nu_ls
        print("accepted alpha is", alpha)

        q = q_ls
        f = f_ls

        if np.linalg.norm(vq) < 1e-4:
            print(f"converged at it {it}")
            break

    return q, f


def add_visual_capsule(scene, point1, point2, radius, rgba):
    """Adds one capsule to an mjvScene."""
    if scene.ngeom >= scene.maxgeom:
        return
    scene.ngeom += 1  # increment ngeom\n",
    # initialise a new capsule, add it to the scene using mjv_makeConnector\n",
    # mujoco.mjv_initGeom(scene.geoms[scene.ngeom - 1])
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom - 1],
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        np.zeros(3),
        np.zeros(3),
        np.zeros(9),
        rgba.astype(np.float32),
    )
    mujoco.mjv_makeConnector(
        scene.geoms[scene.ngeom - 1],
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        radius,
        point1[0],
        point1[1],
        point1[2],
        point2[0],
        point2[1],
        point2[2],
    )


def view_image(image):
    cv2.imshow(f"tmp", image)
    while True:
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyWindow("tmp")



class Goal_controller_q:

    def __init__(self, SIM: bool, pin_model):
        """
        """
        self.SIM = SIM
        self.qvl = 1.0
        self.kvv = 1.0
        self.qgoal = None
        self.qvgoal = None

        self.ff_kp = 0.0
        self.ff_kv = 0.0
        self.pin_model = pin_model

    def get_u(self,q,qv):
        """ 
        """

        assert self.qgoal is not None

        if self.qvgoal is not None:
            aq = self.kvv * (self.qgoal - q) - self.qvl * (qv - self.qvgoal)
        else:
            aq = self.kvv * (self.qgoal - q) - self.qvl * qv

        print("aq", aq)
        print("q", q)
        print("qv", qv)

        ff = pin.rnea(self.pin_model.model, self.pin_model.data, q, qv, aq)
        out =  ff + self.ff_kp * (self.qgoal - q) + self.ff_kv * (self.qvgoal - qv)
        print("self.qgoal - q", self.qgoal - q) 
        print("self.qvgoal - qv", self.qvgoal - qv)



        if not self.SIM:
            out -=  pin.computeGeneralizedGravity(self.pin_model.model, self.pin_model.data, q)
        return out



            # remove the gravity!



