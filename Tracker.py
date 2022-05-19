import numpy as np
import cv2
import utils
from Entities import Entities


class Tracker:
    def __init__(self):
        self.mog = cv2.createBackgroundSubtractorMOG2()
        self.entities = Entities()

    def detect(self, frame):
        fgmask = self.mog.apply(frame, 123, 0.01)

        blured = cv2.GaussianBlur(src=fgmask, ksize=(9, 9), sigmaX=0)
        _, thresh = cv2.threshold(blured, 220, 255, cv2.THRESH_BINARY)
        kernel = np.ones((9, 9))
        dilated = cv2.dilate(thresh, kernel, 1)

        contours, _ = cv2.findContours(image=dilated, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) < 400:
                #too small!
                continue

        (x, y, w, h) = cv2.boundingRect(contour)
        center = (x+w//2, y+h//2)
        self.entities.queue_point(center, (w//2, h//2))

        self.entities.update()

    def draw(self, frame):
        self.entities.draw(frame)

    def get_objects(self):
        return self.entities.entities

    def get_enitites(self):
        return self.entities.entities

    def flush(self):
        self.entities.flush()