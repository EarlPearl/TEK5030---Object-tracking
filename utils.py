import cv2
from time import time
import numpy as np

class ViewGui:
    """
    A simple GUI for visualising results.
    It is based on the code form the segmentation lab:
    https://github.com/tek5030/lab-segmentation-py/blob/master/common_lab_utils.py
    """

    def __init__(self):
        """
        Constructs the GUI
        """
        # Create windows.

        cv2.namedWindow("main", cv2.WINDOW_NORMAL)
        cv2.namedWindow("map", cv2.WINDOW_NORMAL)

        #init timer
        self.last_t = time()

    def __enter__(self):
        """Initialises the GUI"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Destroys the GUI"""
        cv2.destroyWindow("main")
        cv2.destroyWindow("map")

    def show_frame(self, frame_img):
        curr_t = time()
        """Show an image in the "Segmented frame" window"""
        text = f"FPS: {1/(curr_t - self.last_t):.0f}"
        font = cv2.FONT_HERSHEY_PLAIN
        font_scale = 1.5
        thickness = 2
        x, y = (0, 30)
        
        (w, h), _ = cv2.getTextSize(text, font, font_scale, thickness)
        img = cv2.rectangle(frame_img, (x, y - 20), (x + w, y), (125, 125, 125), -1)
        cv2.putText(frame_img, text, (x, y), font, font_scale, (0, 255, 0),thickness)
        cv2.imshow("main", frame_img)


        self.last_t = curr_t

    def show_map(self, frame_img):
        cv2.imshow("map", frame_img)

    def wait_key(self, time_ms):
        """Runs the highgui event loop and receives keypress events"""
        return cv2.waitKey(time_ms)

def drawGridImage(image, grid_interval,scene_X, scene_Y):
    h = image.shape[0]
    w = image.shape[1]

    pixel_per_meter_x = w/scene_X
    pixel_per_meter_y = h/scene_Y

    pixel_interval_x = int(grid_interval * pixel_per_meter_x)
    pixel_interval_y = int(grid_interval * pixel_per_meter_y)

    for v in range(0, h, pixel_interval_y):
        image = cv2.line(image, (v, 0), (v, w), (0, 255, 0), 1)
    for u in range(0, w, pixel_interval_x):
        image = cv2.line(image, (0, u), (h, u), (0, 255, 0), 1)

