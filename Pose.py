import numpy as np
import cv2

class ArucoPose:
    # notes:
    #For axis print - X: red, Y: green, Z: blue

    def __init__(self, rv, tv):
        self.rvec = rv
        self.tvec = tv
        self.rotationMatrix, _ = cv2.Rodrigues(rv)
        self.poseMatrix = np.concatenate((self.rotationMatrix, self.tvec), axis=1)
        self.poseMatrix = np.concatenate((self.poseMatrix, np.array([[0, 0, 0, 1]])), axis=0)
        print(f"Pose matrix:{self.poseMatrix}")

    def boardToCameraCoordinate(self, x, y, cameraMatrix):
        pos_boardPlane_h = np.concatenate(([[x, y]], [[0, 1]]), axis=1).T
        R_t = np.concatenate((self.rotationMatrix, self.tvec), axis=1)
        u_h = cameraMatrix @ R_t @ pos_boardPlane_h
        return u_h[0:2, :]