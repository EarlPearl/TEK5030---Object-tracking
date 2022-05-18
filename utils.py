import cv2
from time import time
import numpy as np

class ViewGui:
    """
    A simple GUI for visualising results.
    It is based on the code form the segmentation lab:
    https://github.com/tek5030/lab-segmentation-py/blob/master/common_lab_utils.py
    """

    def __init__(self, name):
        """
        Constructs the GUI
        """
        # Create windows.
        self.win_name = 'Segmented frame'
        cv2.namedWindow(self.win_name, cv2.WINDOW_NORMAL)

        #init timer
        self.last_t = time()

    def __enter__(self):
        """Initialises the GUI"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Destroys the GUI"""
        cv2.destroyWindow(self.win_name)

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
        cv2.imshow(self.win_name, frame_img)


        self.last_t = curr_t

    def wait_key(self, time_ms):
        """Runs the highgui event loop and receives keypress events"""
        return cv2.waitKey(time_ms)

class Projection2DPlot:
    """Plots projected point on to a background image"""
    def __init__(self, background, actual_width, actual_height):
        self.background = background
        self.image_height, self.image_width = self.background.shape[0:2]
        """Actual size of projection surface given in meters"""
        self.width = actual_width
        self.height = actual_height

    def plot_point(self, p):
        p = np.squeeze(p)
        x_pixel = round(p[0]*(self.image_height/self.height))
        y_pixel = round(p[1]*(self.image_width/self.width))
        return cv2.circle(self.background, (x_pixel, y_pixel), 20, (0, 0, 255), -1)

    @staticmethod
    def add_to_frame(frame, plot, relative_size):
        frame_width = frame.shape[1]
        frame_height = frame.shape[0]
        plot_width = int(frame_width * relative_size)
        plot_height = int(frame_height * relative_size)
        dim = (plot_width, plot_height)

        # resize image
        resized = cv2.resize(plot, dim, interpolation=cv2.INTER_AREA)

        new_frame = frame.copy()
        print(frame.shape)
        print(resized.shape)
        new_frame[:plot_height, (frame_width-plot_width):frame_width] = resized

        return new_frame
