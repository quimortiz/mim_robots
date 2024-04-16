import cv2
import numpy as np

ARUCO_DICT = {
	"DICT_4X4_50": cv2.aruco.DICT_4X4_50,
	"DICT_4X4_100": cv2.aruco.DICT_4X4_100,
	"DICT_4X4_250": cv2.aruco.DICT_4X4_250,
	"DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
	"DICT_5X5_50": cv2.aruco.DICT_5X5_50,
	"DICT_5X5_100": cv2.aruco.DICT_5X5_100,
	"DICT_5X5_250": cv2.aruco.DICT_5X5_250,
	"DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
	"DICT_6X6_50": cv2.aruco.DICT_6X6_50,
	"DICT_6X6_100": cv2.aruco.DICT_6X6_100,
	"DICT_6X6_250": cv2.aruco.DICT_6X6_250,
	"DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
	"DICT_7X7_50": cv2.aruco.DICT_7X7_50,
	"DICT_7X7_100": cv2.aruco.DICT_7X7_100,
	"DICT_7X7_250": cv2.aruco.DICT_7X7_250,
	"DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
	"DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
	"DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
	"DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
	"DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
	"DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}

def aruco_display(corners, ids, rejected, image):
	if len(corners) > 0:
		# flatten the ArUco IDs list
		ids = ids.flatten()
		# loop over the detected ArUCo corners
		for (markerCorner, markerID) in zip(corners, ids):
			# extract the marker corners (which are always returned in
			# top-left, top-right, bottom-right, and bottom-left order)
			corners = markerCorner.reshape((4, 2))
			(topLeft, topRight, bottomRight, bottomLeft) = corners
			# convert each of the (x, y)-coordinate pairs to integers
			topRight = (int(topRight[0]), int(topRight[1]))
			bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
			bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
			topLeft = (int(topLeft[0]), int(topLeft[1]))

			cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
			cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
			cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
			cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)
			# compute and draw the center (x, y)-coordinates of the ArUco
			# marker
			cX = int((topLeft[0] + bottomRight[0]) / 2.0)
			cY = int((topLeft[1] + bottomRight[1]) / 2.0)
			cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)
			# draw the ArUco marker ID on the image
			cv2.putText(image, str(markerID),(topLeft[0], topLeft[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
				0.5, (0, 255, 0), 2)
			print("[Inference] ArUco marker ID: {}".format(markerID))
			# show the output image
	return image


def draw_3d_axis_from_R(ax, x, R, length=1.0):
    """
    Draw a 3D axis on a matplotlib 3D subplot.
    
    Parameters:
    - ax: matplotlib 3D subplot axis to draw on.
    - x: Origin of the axis in 3D space as [x, y, z].
    - R: 3x3 rotation matrix defining the orientation of the axis.
    - length: Length of each axis.
    """
    # Create the points for the ends of each axis (in local coordinates)
    end_points = np.array([[length, 0, 0],
                           [0, length, 0],
                           [0, 0, length]])
    
    # Transform end points with the rotation matrix and add the origin offset
    end_points_transformed = np.dot(R, end_points.T).T + x
    
    # Draw the x-axis in red
    ax.plot([x[0], end_points_transformed[0, 0]], [x[1], end_points_transformed[0, 1]], [x[2], end_points_transformed[0, 2]], 'r-')
    # Draw the y-axis in green
    ax.plot([x[0], end_points_transformed[1, 0]], [x[1], end_points_transformed[1, 1]], [x[2], end_points_transformed[1, 2]], 'g-')
    # Draw the z-axis in blue
    ax.plot([x[0], end_points_transformed[2, 0]], [x[1], end_points_transformed[2, 1]], [x[2], end_points_transformed[2, 2]], 'b-')


def draw_2d_axis_from_R(ax, x, R, length=1.0):
    """
    Draw a 2D axis on a matplotlib 2D subplot.
    
    Parameters:
    - ax: matplotlib 3D subplot axis to draw on.
    - x: Origin of the axis in 3D space as [x, y, z].
    - R: 3x3 rotation matrix defining the orientation of the axis.
    - length: Length of each axis.
    """
    # Create the points for the ends of each axis (in local coordinates)
    end_points = np.array([[length, 0, 0],
                           [0, length, 0],
                           [0, 0, length]])
    
    # Transform end points with the rotation matrix and add the origin offset
    end_points_transformed = np.dot(R, end_points.T).T + x
    
    # Draw the x-axis in red
    ax.plot([x[0], end_points_transformed[0, 0]], [x[1], end_points_transformed[0, 1]],  'r-')
    # Draw the y-axis in green
    ax.plot([x[0], end_points_transformed[1, 0]], [x[1], end_points_transformed[1, 1]], 'g-')
    # Draw the z-axis in blue


def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel().astype(int))
    print("corner", corner)
    print("img", tuple(imgpts[0].ravel()))
    img = cv2.line(img, corner, tuple(imgpts[0].ravel().astype(int)), (255, 0, 0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel().astype(int)), (0, 255, 0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel().astype(int)), (0, 0, 255), 5)
    return img

