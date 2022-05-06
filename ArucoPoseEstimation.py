import random

import cv2
import numpy as np


class ArucoPoseEstimator:
    def __init__(self, matrix_coefficients, distortion_coefficients, marker_size):
        self.dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_50)
        self.params = cv2.aruco.DetectorParameters_create()
        self.matrix_coefficients = matrix_coefficients
        self.distortion_coefficients = distortion_coefficients
        self.marker_size = marker_size #marker size in meters
    def draw_pose(self, frame):
        (corners, ids, rejected) = cv2.aruco.detectMarkers(frame, self.dict,
                                                           parameters=self.params)
        if len(corners) > 0:
            for i in range(0, len(ids)):
                # Estimate pose of each marker and return the values rvec and tvec---(different from those of camera coefficients)
                rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], self.marker_size, self.matrix_coefficients,
                                                                               self.distortion_coefficients)
                # Draw a square around the markers
                cv2.aruco.drawDetectedMarkers(frame, corners)

                # Draw Axis
                cv2.aruco.drawAxis(frame, self.matrix_coefficients, self.distortion_coefficients, rvec, tvec, 0.01)

        return frame

    def generate_markers(self, num_markers, tag_size, path):
        dict_size = len(self.dict.bytesList)
        print(dict_size)
        ids = random.sample(range(0, dict_size), num_markers)
        for id in ids:
            print("Generating ArUCo tag of type '{}' with ID '{}'".format(self.dict, id))
            tag = np.zeros((tag_size, tag_size, 1), dtype="uint8")
            cv2.aruco.drawMarker(self.dict, id, tag_size, tag, 1)

             # Save the tag generated
            tag_name = f'{path}/{self.dict}_id_{id}.png'
            cv2.imwrite(tag_name, tag)
            cv2.imshow("ArUCo Tag", tag)