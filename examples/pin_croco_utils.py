import numpy as np
import math
import pinocchio as pin
import cv2
import mujoco


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

