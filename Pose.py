import numpy as np
import cv2

class ArucoPose:
    # notes:
    #For axis print - X: red, Y: green, Z: blue

    def __init__(self, rv, tv, cameraMatrix):
        self.rvec = rv
        self.tvec = tv
        self.rotationMatrix, _ = cv2.Rodrigues(rv)
        self.poseMatrix = np.concatenate((self.rotationMatrix, self.tvec), axis=1)
        self.poseMatrix = np.concatenate((self.poseMatrix, np.array([[0, 0, 0, 1]])), axis=0)
        self.homoegraphy = cameraMatrix @ np.concatenate((self.rotationMatrix[:, 0:2], tv), axis=1)
        self.cameraMatrix = cameraMatrix

    def boardToCameraCoordinate(self, x, y):
        xw = np.array((x, y))[:, np.newaxis]
        x_w_h = homogeneous(xw)
        u_h = self.homoegraphy @ x_w_h
        return hnormalized(u_h)

    def cameraToboardCoordinate(self, u, v):
        xp = np.array((u, v))[:, np.newaxis]
        x_p_h = homogeneous(xp)
        pos_board_h = np.linalg.inv(self.homoegraphy) @ x_p_h
        return hnormalized(pos_board_h)

    def transformImage(self, frame):
        return frame

def homogeneous(x):
    """Transforms Cartesian column vectors to homogeneous column vectors"""
    return np.r_[x, [np.ones(x.shape[1])]]


def hnormalized(x):
    """Transforms homogeneous column vector to Cartesian column vectors"""
    return x[:-1] / x[-1]
