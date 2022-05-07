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

class Projection2DPlot:
    """Plots projected point on to a background image"""
    def __init__(self, background, actual_width, actual_height):
        self.background = background
        self.image_height, self.image_width = self.background.shape[0:2]
        """Actual size of projection surface given in meters"""
        self.width = actual_width
        self.height = actual_height

    def plot_point(self, x, y):
        x_pixel = round(x*(self.image_height/self.height))
        y_pixel = round(y*(self.image_width/self.width))
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
