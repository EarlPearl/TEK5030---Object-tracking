import cv2

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
        self.segm_win = 'Segmented frame'
        cv2.namedWindow(self.segm_win, cv2.WINDOW_NORMAL)

    def __enter__(self):
        """Initialises the GUI"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Destroys the GUI"""
        cv2.destroyWindow(self.segm_win)

    def show_frame(self, frame_img):
        """Show an image in the "Segmented frame" window"""
        cv2.imshow(self.segm_win, frame_img)

    def wait_key(self, time_ms):
        """Runs the highgui event loop and receives keypress events"""
        return cv2.waitKey(time_ms)