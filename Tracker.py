import numpy as np
import cv2
import utils
from Entities import Entities


class Tracker:
    """
    Detects motion and manages entites
    """
    def __init__(self, MAX_ENTS, threshold, decay, MAX_POINTS):
        """
        Args
            MAX_ENTS: number of entites to trak
            threshold: distance before a point is concidered a new object
            decay: how long to remember objects no longer tracked
            MAX_POINTS: how many coordinates can each object remember
        """
        self.mog = cv2.createBackgroundSubtractorMOG2()
        self._entities = Entities(MAX_ENTS, threshold, decay, MAX_POINTS)
        self.hackyhackvariable = 0 #counts frames, used to stop the MOG adapting

    def detect(self, frame):
        """
        detects motion with MOG2, finds contours.
        Queues the points and updates entites.

        Args:
            frame, an image

        """
        if self.hackyhackvariable < 10:
            fgmask = self.mog.apply(frame, 123, 0.01)
            self.hackyhackvariable += 1
        else:
            fgmask = self.mog.apply(frame, 123, 0)

        blured = cv2.GaussianBlur(src=fgmask, ksize=(9, 9), sigmaX=0)
        _, thresh = cv2.threshold(blured, 220, 255, cv2.THRESH_BINARY)
        kernel = np.ones((9, 9))
        self.dilated = cv2.dilate(thresh, kernel, 1)

        self.contours, _ = cv2.findContours(image=self.dilated, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

        for contour in self.contours:
            if cv2.contourArea(contour) < 400:
                #too small!
                continue

            (x, y, w, h) = cv2.boundingRect(contour)
            center = (x+w//2, y+h//2)
            self._entities.queue_point(center, (w//2, h//2))

        self._entities.update()

    def draw(self, frame):
        """
        draws all entites
        """
        self._entities.draw(frame)

    def get_enitites(self):
        """
        returns
            a list of all tracked entites
        """
        return self._entities.entities

    def flush(self):
        """
        cleares all tracked entites and queued points
        """
        self._entities.flush()