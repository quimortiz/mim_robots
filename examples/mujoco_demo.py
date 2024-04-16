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

q = data.joint("cube_j").qpos
q[0] = 2
mujoco.mj_forward(model, data)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

camera_names = ["camera-right", "camera-front", "camera-left"]

renderer = mujoco.Renderer(model, 720, 1080)
renderer.update_scene(data)

for camera in camera_names:
    print(f"camera {camera}")
    renderer.update_scene(data, camera=camera)
    pixels = renderer.render()
    image = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
    view_image(image)


pattern_size = (8, 5)
square_size = 0.03
pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
pattern_points *= square_size

object_points_3D = pattern_points

camera_ts = []
camera_Rs = []

# From the mujoco tutorial:
# pos = np.mean([camera.pos for camera in renderer.scene.camera], axis=0)
# z = -np.mean([camera.forward for camera in renderer.scene.camera], axis=0)
# y = np.mean([camera.up for camera in renderer.scene.camera], axis=0)
# rot = np.vstack((np.cross(y, z), y, z))

dcamera_gt ={}

camera_num_2_name = [ "camera-front", "camera-right", "camera-left" ]

for i in range(model.ncam):
    camera_pos = data.cam_xpos[i]
    camera_xmat = data.cam_xmat[i]
    R = np.array(data.cam_xmat[i]).reshape(3, 3)
    camera_ts.append(camera_pos)
    camera_Rs.append(R)
    print(f"Camera {i}:")
    print(f"  Position: {camera_pos}")
    print(f"  Orientation (quaternion): {camera_xmat}")
    print("R", R)
    dcamera_gt[camera_num_2_name[i]] = {"t": camera_pos, "R": R}
    

dintrinsic = {}

# geo-chess" pos='0 0 0.0' size=".105 0.1485 0.0005" type="box" material="chess-mat"/>

chessboard_joint = "chessboard_joint" 
big_marker_joint = "big_marker_j"

dcalib = {}

img_shape = (1080, 720)
if args.callibrate_intrinsics:
    

    data.joint(big_marker_joint).qpos[0] = 2

    num_chess_images = 20


    data.joint(chessboard_joint).qpos[0] = 0.6
    data.joint(chessboard_joint).qpos[1] = 0.0
    data.joint(chessboard_joint).qpos[2] = 0.2

    ref_pose = np.copy(data.joint(chessboard_joint).qpos[:3])

    for i, camera_name in enumerate(camera_names):

        good_images = []
        image_points = []
        object_points = []

        for i in range(num_chess_images):
            if i > 0:
                rand_noise = np.random.normal(0, 0.1, 3)
                rand_noise_p = np.random.normal(0, 0.1, 3)
                quat = RR.from_euler("xyz", rand_noise).as_quat()
                data.joint(chessboard_joint).qpos[3:7] = quat
                data.joint(chessboard_joint).qpos[0:3] = ref_pose + rand_noise_p

            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera_name)
            pixels = renderer.render()

            image = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
            image_shape = image.shape[::-1]
            # prina(image.shape)
            # sys.exit()

            view_image(image)

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            success, corners = cv2.findChessboardCorners(gray, pattern_size, None)
            print("success", success)
            if success:
                good_images.append(image)
                cv2.drawChessboardCorners(image, pattern_size, corners, success)

                corners_2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                image_points.append(corners_2)
                object_points.append(object_points_3D)

                view_image(image)

            # Display the window for a short period. Used for testing.

        ret, mtx, dist, rvecs_0, tvecs_0 = cv2.calibrateCamera(
            object_points, image_points, 
            img_shape, None, None)

        dcalib[camera_name] = {"mtx": mtx, "dist": dist}
        print("calibration done!")
        print(dcalib[camera_name])

    fileout = "calib_three_cameras.pkl"
    with open(fileout, "wb") as f:
        pickle.dump(dcalib, f)

    fileout = "calib_three_cameras.yaml"
    Dyaml = {} 
    for dk, dv in dcalib.items():
        d = {}
        for k,v in dv.items():
            d[k] = v.tolist()
        Dyaml[dk] = d
    with open(fileout, "w") as f:
        yaml.dump(Dyaml, f)


else: 
    filein = "calib_three_cameras.pkl"
    with open(filein, "rb") as f:
        dcalib = pickle.load(f)

use_params_of_front_camera = True
camera_names = ["camera-right", "camera-front", "camera-left"]

dcalib["camera-right"] = copy.deepcopy(dcalib["camera-front"])
dcalib["camera-left"] = copy.deepcopy(dcalib["camera-front"])

dR_camera = {}
dR_camera_mujoco = {}
dp_camera = {}





if args.callibrate_extrinsics:


    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    # move the chessboard out of the way
    data.joint(chessboard_joint).qpos[0] = 3
    data.joint(big_marker_joint).qpos[0] = 0.5


    for ii, camera in enumerate(camera_names):

        renderer.update_scene(data, camera=camera)
        pixels = renderer.render()
        image = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
        view_image(image)

        markerCorners, markerIds, rejectedCandidates = detector.detectMarkers(image)

        markerCorner = markerCorners[0] # only one marker in scene
        corners = markerCorner.reshape((4, 2))
        (topLeft, topRight, bottomRight, bottomLeft) = corners
        detected_markers = aruco_display(markerCorners, markerIds, rejectedCandidates, image)

        corners_for_pnp = np.array([topRight, bottomRight, bottomLeft, topLeft])

        # corners = 
        # (topLeft, topRight, bottomRight, bottomLeft) = corners
        # # convert each of the (x, y)-coordinate pairs to integers
        topRight = (int(topRight[0]), int(topRight[1]))
        bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
        bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
        topLeft = (int(topLeft[0]), int(topLeft[1]))


        world_points = np.stack([
        data.geom("big_marker_p1").xpos,
        data.geom("big_marker_p2").xpos,
        data.geom("big_marker_p3").xpos,
        data.geom("big_marker_p4").xpos
        ])

        print("world_points", world_points)

        for c in corners:
            _img = cv2.circle(
                image,
                (int(c[0]), int(c[1])),
                radius=5,
                color=(0, 0, 255),
                thickness=-1,
            )

            view_image(_img)


        markerLength = .2

        # extract the marker corners (which are always returned in
        # top-left, top-right, bottom-right, and bottom-left order)
        p1 =  np.array([-markerLength/2, markerLength/2, 0])
        p2 = np.array([markerLength/2, markerLength/2, 0])
        p3 = np.array([markerLength/2, -markerLength/2, 0])
        p4 = np.array([-markerLength/2, -markerLength/2, 0])

        objPoints = np.array([p1, p2, p3, p4])

        # D = dcalib[camera]
        # fileout = "calib.pkl"
        # with open(fileout, "rb") as f:
        #     D = pickle.load(f)
            # pickle.dump({"mtx": mtx, "dist": dist}, f)

        dist = dcalib[camera]["dist"]
        mtx = dcalib[camera]["mtx"]
        print("mtx")
        print(mtx)
        print("dist")
        print(dist)

        



        rvec = np.zeros(3)
        tvec = np.zeros(3)
        ret, rvecs, tvecs = cv2.solvePnP(world_points, corners_for_pnp, mtx , dist)

        print("ret", ret)
        print("rvecs", rvecs)
        print("tvecs", tvecs)

        R, _ = cv2.Rodrigues(rvecs.flatten())
        print("R")
        tt = tvecs.flatten()
        print("tvecs", tt)
        p = -R.T @ tt
        print("pose of camera is ", p)
        print("Rotation of camera is ", R.T)
        print("mujoco frame of camera is ", R.T @ np.array([[1,0,0], [0,-1,0],[0,0,-1]]))
        p_camera = p


        R_camera = R.T
        R_camera_mujoco = R.T @ np.array([[1,0,0], [0,-1,0],[0,0,-1]])

        dp_camera[camera] = p
        dR_camera[camera] = R_camera
        dR_camera_mujoco[camera] = R_camera_mujoco

        #dcamera_gt ={} check error against mujoco!!
        compare_against_gt = True
        if compare_against_gt:
            p_gt = dcamera_gt[camera]["t"]
            errro_p = np.linalg.norm(p - p_gt)
            error_r = np.linalg.norm(R_camera_mujoco - dcamera_gt[camera]["R"])
            print("error in p", errro_p)
            print("error in R", error_r)

        view_image(detected_markers)
    # write down the results
    filout = "extrinsics.pkl"

    data_out = {"p": dp_camera, "R": dR_camera, "R_mujoco": dR_camera_mujoco}
    with open(filout, "wb") as f:
        pickle.dump(data_out, f)

    # write down as yaml

    dp_camera_list = copy.deepcopy(dp_camera)
    dR_camera_list = copy.deepcopy(dR_camera)
    dR_camera_mujoco_list = copy.deepcopy(dR_camera_mujoco)

    for k,v in dp_camera_list.items():
        dp_camera_list[k] = v.tolist()
    for k,v in dR_camera_list.items():
        dR_camera_list[k] = v.tolist()
    for k,v in dR_camera_mujoco_list.items():
        dR_camera_mujoco_list[k] = v.tolist()

    with open("extrinsics.yaml", "w") as f:
        yaml.dump({"p": dp_camera_list, "R": dR_camera_list, "R_mujoco": dR_camera_mujoco_list}, f)

else:
    file_in = "extrinsics.pkl"
    with open(file_in, "rb") as f:
        extrinsics = pickle.load(f)
    dp_camera = extrinsics["p"]
    dR_camera = extrinsics["R"]
    dR_camera_mujoco = extrinsics["R_mujoco"]

if args.record_movement:

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    markerLength = .2

    data.joint(chessboard_joint).qpos[0] = 3
    data.joint(big_marker_joint).qpos[0] = 3


    mujoco.mj_forward(model, data)
    q = data.joint("cube_j").qpos
    q[0] = .8
    q[1] = 0 

    mujoco.mj_forward(model, data)

    radius = .1
    c = (.7, .0)

    num_steps = 10
    x_block =  c[0] + radius * np.cos(np.linspace(0, 2*np.pi, num_steps))
    y_block =  c[1] + radius * np.sin(np.linspace(0, 2*np.pi, num_steps))
    angle_block = np.linspace(0, 2*np.pi, num_steps)


    for j,camera_name in enumerate(camera_names):

        real_pose = []
        real_orientation = []
        estimated_pose = []
        estimated_orientation = []

        mtx = dcalib[camera_name]["mtx"]
        dist = dcalib[camera_name]["dist"]

        r_camera = dR_camera[camera_name]
        p_camera = dp_camera[camera_name]

        for ii in range(num_steps):

            q = data.joint("cube_j").qpos
            q[0] = x_block[ii]
            q[1] = y_block[ii]
            angle = angle_block[ii]

            print("angle is", angle)
            # convert euler to quaternion using rowan
            q_ = RR.from_euler( "xyz", [0., 0., angle]).as_quat()

            print("q_", q_)
            q[3] = q_[3]
            q[4] = q_[0]
            q[5] = q_[1]
            q[6] = q_[2]

            print("angle", angle)
            mujoco.mj_forward(model, data)
            real_pose.append(np.copy(data.geom("geo-marker").xpos))
            print( data.geom("geo-marker").xpos)
            real_orientation.append(np.copy(data.geom("geo-marker").xmat))

            renderer.update_scene(data, camera=camera_name)
            pixels = renderer.render()
            image = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)

            view_image(image)



            # lets take a picture with the camera!

            markerCorners, markerIds, rejectedCandidates = detector.detectMarkers(image)
            print("markerCorners", markerCorners)
            print("markerIds", markerIds)
            print("rejectedCandidates", rejectedCandidates)
            markerCorner = markerCorners[0] # only one marker in scene
            corners = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners
            detected_markers = aruco_display(markerCorners, markerIds, rejectedCandidates, image)

            corners_for_pnp = np.array([topRight, bottomRight, bottomLeft, topLeft])

            # corners = 
            # (topLeft, topRight, bottomRight, bottomLeft) = corners
            # # convert each of the (x, y)-coordinate pairs to integers
            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))


            for c in corners:
                _img = cv2.circle(
                    image,
                    (int(c[0]), int(c[1])),
                    radius=5,
                    color=(0, 0, 255),
                    thickness=-1,
                )
                view_image(_img)



            object_points = np.array([[markerLength/2, markerLength/2, 0],  
                                      [markerLength/2, -markerLength/2, 0],
                                      [-markerLength/2, -markerLength/2, 0],
                                      [-markerLength/2, markerLength/2, 0]])

            print("object_points")
            print(object_points)


            ret, rvecs, tvecs = cv2.solvePnP(object_points, corners_for_pnp, mtx , dist)

            r, _ = cv2.Rodrigues(rvecs.flatten())
            tvecs = tvecs.flatten()


            print("ret", ret)
            print("rvecs", rvecs)
            print("tvecs", tvecs)

            print("R_camera @ tvecs + p_camera", r_camera @ tvecs + p_camera)
            estimated_pose.append(r_camera @ tvecs + p_camera)
            print("R_camera.T @ tvecs + p_camera", r_camera.T @ tvecs + p_camera)
            print("R_camera @ R", r_camera @ r)
            print("geo marker", data.geom("geo-marker").xpos)
            estimated_orientation.append( r_camera @ r)


        fig, ax =  plt.subplots(1,1)
        ax.plot(np.array(real_pose)[:,0], np.array(real_pose)[:,1], label="real")
        ax.plot(np.array(estimated_pose)[:,0], np.array(estimated_pose)[:,1], label="estimated")
        ax.legend()
        ax.set_aspect('equal', 'box')
        plt.show()

        fig, ax =  plt.subplots(1,1)
        ax.plot(np.array(real_pose)[:,2],label="real")
        ax.plot(np.array(estimated_pose)[:,2],label="estimated")
        ax.legend()
        plt.show()

    # lets plot the axes

        fig, ax =  plt.subplots(1,1)
        for ii in range(num_steps):
            x = real_pose[ii]

            R = real_orientation[ii].reshape(3,3)

            draw_2d_axis_from_R(ax, x, R, length=.05)

            x = estimated_pose[ii]
            R = estimated_orientation[ii].reshape(3,3)
            draw_2d_axis_from_R(ax, x, R, length=.05)

        ax.plot(np.array(real_pose)[:,0], np.array(real_pose)[:,1], label="real")
        ax.plot(np.array(estimated_pose)[:,0], np.array(estimated_pose)[:,1], label="estimated")
        ax.legend()
        ax.set_aspect('equal', 'box')
        plt.show()

        fig = plt.figure()
        ax = plt.axes(projection="3d")
        for ii in range(num_steps):
            x = real_pose[ii]
            R = real_orientation[ii].reshape(3,3)
            draw_3d_axis_from_R(ax, x, R, length=.05)

            x = estimated_pose[ii]
            R = estimated_orientation[ii].reshape(3,3)
            draw_3d_axis_from_R(ax, x, R, length=.05)

        ax.plot(np.array(real_pose)[:,0], np.array(real_pose)[:,1],np.array(real_pose)[:,2], label="real")
        ax.plot(np.array(estimated_pose)[:,0], np.array(estimated_pose)[:,1], np.array(estimated_pose)[:,2], label="estimated")
        ax.legend()
        # add labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        # ax.set_aspect('equal', 'box')
        plt.show()

sys.exit()

      # <body name='cube' pos='-0.4 0.2 0.2'>



for i in range(15):



    if i > 0:
        # # q  = q_orig[:3] + np.random.normal(0, 0.05, 3)
        # q[2] = 1
        rand_noise = np.random.normal(0, 0.1, 3)
        quat = RR.from_euler("xyz", rand_noise).as_quat()
        data.joint(myjoint).qpos[3:7] = quat
        data.joint(myjoint).qpos[2] = 0.1

    mujoco.mj_forward(model, data)

    renderer.update_scene(data, camera="camera-front")
    pixels = renderer.render()

    image = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
    # save image as png
    cv2.imshow(f"image", image)

    while True:
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    success, corners = cv2.findChessboardCorners(gray, pattern_size, None)
    print("success")
    print(success)

    if success:
        good_images.append(image)
        cv2.drawChessboardCorners(image, pattern_size, corners, success)

        corners_2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        image_points.append(corners_2)
        object_points.append(object_points_3D)

        # Display the image. Used for testing.
        cv2.imshow("Image", image)
        # Display the window for a short period. Used for testing.
        cv2.waitKey(200)
    cv2.destroyAllWindows()


ret, mtx, dist, rvecs_0, tvecs_0 = cv2.calibrateCamera(
    object_points, image_points, gray.shape[::-1], None, None
)

# lets save mtx and dist to a yaml file and to a pickle file


print("calibration done!")
print("mtx")
print(mtx)

print("dist")
print(dist)

fileout = "calib.yaml"
with open(fileout, "w") as f:
    yaml.dump({"mtx": mtx, "dist": dist}, f)

fileout = "calib.pkl"
with open(fileout, "wb") as f:
    pickle.dump({"mtx": mtx, "dist": dist}, f)


# import pdb
# pdb.set_trace()


# Find the rotation and translation vectors.
# ret,rvecs, tvecs = cv.solvePnP(objp, corners2, mtx, dist)
# project 3D points to image plane

axis = np.float32([[5, 0, 0], [0, 5, 0], [0, 0, 5]]).reshape(-1, 3) * square_size
for img in good_images:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow(f"gray", gray)
    while True:
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

    cv2.drawChessboardCorners(img, pattern_size, corners, True)

    cv2.imshow(f"corners", img)
    while True:
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    if ret == True:
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        # Find the rotation and translation vectors.

        # import pdb
        # pdb.set_trace()
        for i, c in enumerate(corners2):
            print(object_points_3D[i])
            print(f" i:{i} corner:{corners2[i]}")
            _img = cv2.circle(
                img,
                (int(c[0][0]), int(c[0][1])),
                radius=5,
                color=(0, 0, 255),
                thickness=-1,
            )
            cv2.imshow(f"corner", _img)
            while True:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        ret, rvecs, tvecs = cv2.solvePnP(object_points_3D, corners2, mtx, dist)
        print("ret of solve pnp", ret)
        print("rvecs", rvecs)
        print("tvecs", tvecs)
        print("mtx", mtx)
        # project 3D points to image plane
        origin = np.zeros(3)
        R, _ = cv2.Rodrigues(rvecs.flatten())

        # rvec and tvec are the R and t from https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
        # corner = fx * X / Z + cx
        # where X and Z are the tvec
        # 871 / .507 * -.186 + 525
        #

        imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)

        # R.T @ tvecs.flatten() array([-0.10378983, -0.77556981,  0.68128425])
        # the origin of camera with respect to object.
        # i get the same with output of calib
        # what about the rotation?

        # <camera name="camera-front" pos="1.2 0.0 .7" euler="0 .5 0 "/>

        # <body name='chessboard-0' pos='.44 -.105 0.01'>
        # Center of the camera is (.44, -.105, 0.01) + Rflip -1 R.T @ tvecs
        # where Rflip change the x,y,z to y,x,-z

        # it seems that the object frame the z is pointing downards and the x and y are switched

        Rextra = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])

        # Rextra @ R.T @ Rextra2 = camera_Rs[0]
        # (Pdb) Rextra
        # array([[ 0,  1,  0],
        #        [ 1,  0,  0],
        #        [ 0,  0, -1]])

        # And:
        # (Pdb) Rextra2
        # array([[ 1,  0,  0],
        #        [ 0, -1,  0],
        #        [ 0,  0, -1]])

        # TODO: in PNP, I can just set well the world points to match my world coordinates!
        # use it now?
        # actually, I could just use three points! -- I can try!

        # Notes: using opencv, I get the matrix of the camera (e.g., not exactly what i write in mujoco). Then I
        # use this to track the pose of the object! Lets continue here tomorrow!

        import pdb

        pdb.set_trace()

        img = draw(img, corners2, imgpts)
        cv2.imshow("img", img)
        while True:
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break


#
# imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
# img = draw(img,corners2,imgpts)
# cv.imshow('img',img)


print("calibration done")


print("ret", ret)
print("mtx", mtx)
print("dist", dist)
print("rvecs", rvecs)
print("tvecs", tvecs)

tt = tvecs[0].flatten()
rt = rvecs[0].flatten()
R, _ = cv2.Rodrigues(rt)

# import pdb
# pdb.set_trace()


# sys.exit()


# renderer.update_scene(data, camera=camera)
# pixels = renderer.render()
#
# image = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
# # save image as png
# cv2.imshow("image", image)


# sys.exit()
for ii, camera in enumerate(camera_names):

    image_points = []

    print(f"camera {camera}")
    renderer.update_scene(data, camera=camera)
    pixels = renderer.render()

    image = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
    # save image as png
    cv2.imshow("image", image)
    out = f"tmp_pics/camera_{camera}.png"
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(f"tmp_pics/camera_{camera}.png", image)

    while True:
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # image_path = "/home/quim/Pictures/Screenshots/Screenshot from 2024-04-14 17-17-31.png"
    # load image

    # image_path = "/home/quim/code/camera_calib/play_aruco/pictures_chess/2024-03-14-151319.jpg"

    # gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    cv2.imshow("Image", gray)
    # Display the window for a short period. Used for testing.
    cv2.waitKey(200)

    success, corners = cv2.findChessboardCorners(gray, pattern_size, None)
    print("success")
    print(success)

    if success:
        cv2.drawChessboardCorners(image, pattern_size, corners, success)

        corners_2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        image_points.append(corners_2)

        # Display the image. Used for testing.
        cv2.imshow("Image", image)
        # Display the window for a short period. Used for testing.
        cv2.waitKey(200)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            object_points, image_points, gray.shape[::-1], None, None
        )

        # tation and translation vector performs a change of basis from object coordinate space to camera coordinate space.

        # Due to its duality, this tuple is equivalent to the position of the calibration pattern with respect to the camera coordinate space.

        print("ret", ret)
        print("mtx", mtx)
        print("dist", dist)
        print("rvecs", rvecs)
        print("tvecs", tvecs)

        # lets project
        rotation_matrix, _ = cv2.Rodrigues(rvecs[0].flatten())
        p = tvecs[0].flatten()
        print("rotation matrix", rotation_matrix)
        print("R\n", camera_Rs[ii])
        print("ts\n", camera_ts[ii])

        # the

        # z is -z
        # x is y
        # y is x

        #

        Rextra = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]]).transpose()
        print("Rextra @ rotation_matrix", Rextra @ rotation_matrix)

        pw = (Rextra @ camera_Rs[ii]) @ p + camera_ts[ii]

        print("pw is ", pw)

        # import pdb
        # pdb.set_trace()

        sys.exit()


image_freq = 10  # per second

with mujoco.viewer.launch_passive(model, data) as viewer:
    # Close the viewer automatically after 30 wall-seconds.
    start = time.time()
    sim_time = data.time
    last_image_time = sim_time
    while viewer.is_running() and time.time() - start < 30:
        step_start = time.time()

        # mj_step can be replaced with code that also evaluates
        # a policy and applies a control signal before stepping the physics.
        mujoco.mj_step(model, data)

        # Example modification of a viewer option: toggle contact points every two seconds.
        with viewer.lock():
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(data.time % 2)

        # Pick up changes to the physics state, apply perturbations, update options from GUI.
        viewer.sync()

        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

        # Continue here!! Render three images of the robot from the different cameras!

        sim_time = data.time
        if sim_time - last_image_time > 1 / image_freq:
            print("rendering images")
            print("time is ", sim_time)
            for camera in camera_names:
                print(f"camera {camera}")
                renderer.update_scene(data, camera=camera)
                pixels = renderer.render()
                cv2.imshow("image", cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR))
                while True:
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
            last_image_time = sim_time

        # model = mujoco.MjModel.from_xml_string(tippe_top)

        #

if args.vis > 1:
    while True:
        # cv2 uses BGR, mujoco uses RGB
        cv2.imshow("image", cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))
        # close the window when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()

    # media.show_image(renderer.render())

use_pinocchio = False

if use_pinocchio:

    model_pin = load_pinocchio_wrapper("iiwa_pusher")
    print(model_pin)
    print(model_pin.model)

    # get the pose of model_pin!!

    # double chec

    # model, collision_model, visual_model = pin.buildModelsFromUrdf(
    #     urdf_model_path, mesh_dir, pin.JointModelFreeFlyer()
    # )

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
        viz.initViewer(open=True)
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
    q0 = pin.neutral(model_pin.model)
    viz.display(q0)
    viz.displayVisuals(True)

    # IDX_TOOL = model_pin.model.getFrameId('frametool')
    # IDX_BASIS = model_pin.model.getFrameId('framebasis')

    print("name")
    for i, n in enumerate(model_pin.model.names):
        print(i, n)

    print("frames")
    for f in model_pin.model.frames:
        print(f.name, "attached to joint #", f.parent)

    IDX_PUSHER = model_pin.model.getFrameId("pusher")

    q = q0
    robot = model_pin
    pin.framesForwardKinematics(model_pin.model, model_pin.data, q)
    oMtool = robot.data.oMf[IDX_PUSHER]
    print("Tool placement:", oMtool)
