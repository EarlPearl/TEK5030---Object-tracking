import numpy as np
import cv2
import utils
from Entities import Entities


class Tracker:
    def __init__(self, MAX_ENTS, threshold, decay, MAX_POINTS):
        self.mog = cv2.createBackgroundSubtractorMOG2()
        self._entities = Entities(MAX_ENTS, threshold, decay, MAX_POINTS)
        self.hackyhackvariable = 0

    def detect(self, frame):
        if self.hackyhackvariable < 10:
            fgmask = self.mog.apply(frame, 123, 0.01)
            self.hackyhackvariable += 1
        else:
            fgmask = self.mog.apply(frame, 123, 0)

        blured = cv2.GaussianBlur(src=fgmask, ksize=(9, 9), sigmaX=0)
        _, thresh = cv2.threshold(blured, 220, 255, cv2.THRESH_BINARY)
        kernel = np.ones((9, 9))
        self.dilated = cv2.dilate(thresh, kernel, 1)

        contours, _ = cv2.findContours(image=self.dilated, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) < 400:
                #too small!
                continue

            (x, y, w, h) = cv2.boundingRect(contour)
            center = (x+w//2, y+h//2)
            self._entities.queue_point(center, (w//2, h//2))

        self._entities.update()

    def draw(self, frame):
        self._entities.draw(frame)

    def get_enitites(self):
        return self._entities.entities

    def flush(self):
        self._entities.flush()