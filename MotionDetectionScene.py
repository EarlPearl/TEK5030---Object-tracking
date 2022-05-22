import cv2
import numpy as np
import utils
import Pose

class Scene:
    def __init__(self, marker_size, marker_seperation, markersX, markersY, sceneWidth, sceneHeight):
        self.dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_ARUCO_ORIGINAL)
        self.params = cv2.aruco.DetectorParameters_create()
        self.matrix_coefficients =  np.array([
        [6.6051081297156020e+02, 0., 3.1810845757653777e+02],
        [0., 6.6051081297156020e+02, 2.3995332228230293e+02],
        [0., 0., 1.]
        ])
        self.distortion_coefficients = np.array([0., 2.2202255011309072e-01, 0., 0., -5.0348071005413975e-01])
        self.marker_size = marker_size  # marker size in meters
        self.board = cv2.aruco.GridBoard_create(markersX, markersY, marker_size, marker_seperation, self.dict)
        self.num_markers = markersX * markersY
        self.sceneHeight = sceneHeight
        self.sceneWidth = sceneWidth

    def detect_markers(self, frame):
        (corners, ids, rejected) = cv2.aruco.detectMarkers(frame, self.dict,
                                                         parameters=self.params)
        pose = None
        if len(corners) > 0:
            for i in range(0, len(ids)):
                # Estimate pose of each marker and return the values rvec and tvec---(different from those of camera coefficients)
                rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], self.marker_size,
                                                                               self.matrix_coefficients,
                                                                               self.distortion_coefficients)
                # Draw a square around the markers
                cv2.aruco.drawDetectedMarkers(frame, corners)

                # Draw Axis
                cv2.aruco.drawAxis(frame, self.matrix_coefficients, self.distortion_coefficients, rvec, tvec, 0.01)

            rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[0], self.marker_size,
                                                                           self.matrix_coefficients,
                                                                           self.distortion_coefficients)
            if (len(ids) > 1):

                retval, rvec, tvec = cv2.aruco.estimatePoseBoard(corners, ids, self.board, self.matrix_coefficients,
                                                                 self.distortion_coefficients, rvec, tvec)
                cv2.aruco.drawAxis(frame, self.matrix_coefficients, self.distortion_coefficients, rvec, tvec, 0.1)
                # cv2.aruco.drawPlanarBoard(board, frame.shape[0:2], frame, 0, 5)
                if retval > 0:
                    pose = rvec, tvec
        return frame, pose

    def run_camera_calibration(self, num_calibration_markers, num_detection):
        video_source = 0
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print(f"Could not open video source {video_source}")
            return
        else:
            print(f"Successfully opened video source {video_source}")
        with utils.ViewGui('Camera calibration') as gui:
            self.scene_pose = None
            detect = False
            all_corners = np.array([[]])
            all_ids = np.array([[]])
            counter_array = np.array([[]])
            detect_count = 0
            while True:

                success, frame = cap.read()
                if not success:
                    break
                if detect:
                    (corners, ids, rejected) = cv2.aruco.detectMarkers(frame, self.dict,
                                                                   parameters=self.params)
                    if len(corners) > 0:
                        print(len(ids))
                        if len(ids) == num_calibration_markers:
                            print("Detection ok")
                            all_corners = np.concatenate((all_corners, np.array(corners)), axis=0) if np.array(all_corners).size else np.array(corners)
                            all_ids = np.concatenate((all_ids, ids), axis=0) if np.array(all_ids).size else np.array(ids)
                            number_of_markers = np.array([[len(ids)]])
                            counter_array = np.concatenate((counter_array, number_of_markers), axis=0) if np.array(counter_array).size else number_of_markers
                            detect_count += 1
                            print(f"{detect_count} out of {num_detection} completed")
                            detect = False
                if detect_count > (num_detection - 1):
                    print("All detection completed. Calibrating...")
                    retval, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraAruco(all_corners,
                                                                                                      all_ids, counter_array, self.board, np.shape(frame)[0:2],cameraMatrix=None, distCoeffs=None)
                    if retval > 0:
                        print("Calibrated")
                        print(camera_matrix)
                        print(dist_coeffs)
                        self.matrix_coefficients = camera_matrix
                        self.distortion_coefficients = dist_coeffs
                        break

                gui.show_frame(frame)
                key = gui.wait_key(1)
                if key == ord("q"):
                    break
                if key == ord("d"):
                    detect = True


    def run_scene_analyze(self, video_source=0):
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print(f"Could not open video source {video_source}")
            return
        else:
            print(f"Successfully opened video source {video_source}")
        with utils.ViewGui() as gui:
            self.scene_pose = None

            while True:

                success, frame = cap.read()
                frame = cv2.undistort(frame, self.matrix_coefficients, self.distortion_coefficients)
                if not success:
                    break

                if self.scene_pose is None:
                    frame, pose = self.detect_markers(frame)
                    if pose:
                        rvec, tvec = pose
                        self.scene_pose = Pose.ArucoPose(rvec, tvec, self.matrix_coefficients)
                else:
                    cv2.aruco.drawAxis(frame, self.matrix_coefficients, self.distortion_coefficients, self.scene_pose.rvec, self.scene_pose.tvec, 0.1)

                gui.show_frame(frame)
                key = gui.wait_key(1)
                if key == ord("q"):
                    if pose is None:
                        print("No scene detected")
                    break
                if key == ord("r"):
                    self.scene_pose = None
               


