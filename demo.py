import os

import numpy as np
import random
import cv2

import Pose
import utils
import ArucoPoseEstimation
import MotionDetectionScene
import Tracker



def run_motion_detction(video_source=0):
    # Connect to the camera.
    # Change to video file if you want to use that instead.
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Could not open video source {video_source}")
        return
    else:
        print(f"Successfully opened video source {video_source}")

    # Read the first frame.
    success, frame = cap.read()
    if not success:
        return
    with utils.ViewGui() as gui:
        frame_count = 0
        previous_frame = None
        while True:
            success, frame = cap.read()
            if not success:
                break
            frame_count += 1

            # 1. Load image; convert to RGB
            img_brg = np.array(frame)
            img_rgb = cv2.cvtColor(src=img_brg, code=cv2.COLOR_BGR2RGB)

            # 2. Prepare image; grayscale and blur
            prepared_frame = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
            prepared_frame = cv2.GaussianBlur(src=prepared_frame, ksize=(5, 5), sigmaX=0)

            # 3. Set previous frame and continue if there is None
            if previous_frame is None:
                # First frame; there is no previous one yet
                previous_frame = prepared_frame
                continue

            # calculate difference and update previous frame
            diff_frame = cv2.absdiff(src1=previous_frame, src2=prepared_frame)
            previous_frame = prepared_frame

            # 4. Dilute the image a bit to make differences more seeable; more suitable for contour detection
            kernel = np.ones((5, 5))
            diff_frame = cv2.dilate(diff_frame, kernel, 1)

            # 5. Only take different areas that are different enough (>20 / 255)
            thresh_frame = cv2.threshold(src=diff_frame, thresh=20, maxval=255, type=cv2.THRESH_BINARY)[1]

            contours, _ = cv2.findContours(image=thresh_frame, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
            # cv2.drawContours(image=frame, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2,
             #                lineType=cv2.LINE_AA)

            for contour in contours:
                if cv2.contourArea(contour) < 50:
                   # too small: skip!
                    continue
                (x, y, w, h) = cv2.boundingRect(contour)
                cv2.rectangle(img=frame, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 0), thickness=2)

            gui.show_frame(frame)
            key = gui.wait_key(1)


def main(video_source=0):
    # Connect to the camera.
    # Change to video file if you want to use that instead.
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Could not open video source {video_source}")
        return
    else:
        print(f"Successfully opened video source {video_source}")

    # Read the first frame.
    success, frame = cap.read()
    if not success:
        return

    # Stop video source.
    # cap.release()

def Generate_Aruco_markers():
    K = np.array([
        [6.6051081297156020e+02, 0., 3.1810845757653777e+02],
        [0., 6.6051081297156020e+02, 2.3995332228230293e+02],
        [0., 0., 1.]
    ])
    dist_coeffs = np.array([0., 2.2202255011309072e-01, 0., 0., -5.0348071005413975e-01])
    aruco = ArucoPoseEstimation.ArucoPoseEstimator(K, dist_coeffs, 0.1)
    path = os.path.dirname(os.path.abspath(__file__))
    aruco.generate_markers(4, 200, path)

def Test_Detect_Aruco_markers(video_source=0):
    K = np.array([
        [6.6051081297156020e+02, 0., 3.1810845757653777e+02],
        [0., 6.6051081297156020e+02, 2.3995332228230293e+02],
        [0., 0., 1.]
    ])
    dist_coeffs = np.array([0., 2.2202255011309072e-01, 0., 0., -5.0348071005413975e-01])
    aruco = ArucoPoseEstimation.ArucoPoseEstimator(K, dist_coeffs, 0.075)
    # Connect to the camera.
    # Change to video file if you want to use that instead.
    
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Could not open video source {video_source}")
        return
    else:
        print(f"Successfully opened video source {video_source}")

    # Read the first frame.
    success, frame = cap.read()
    if not success:
        return
    with utils.ViewGui() as gui:
        while True:
            success, frame = cap.read()
            if not success:
                break
            frame = aruco.detect_planar_board(frame)
            frame = gui.show_frame(frame)
            key = gui.wait_key(1)


def Test_Scene_detection():
    K = np.array([
        [6.6051081297156020e+02, 0., 3.1810845757653777e+02],
        [0., 6.6051081297156020e+02, 2.3995332228230293e+02],
        [0., 0., 1.]
    ])

    dist_coeffs = np.array([0., 2.2202255011309072e-01, 0., 0., -5.0348071005413975e-01])
    scene_detection = MotionDetectionScene.Scene(0.068, 0.0015, 4, 2)
    #scene_detection.run_camera_calibration(8, 20)
    scene_detection.run_scene_analyze()

    #print(scene_detection.scene_pose)

def Test2DPlot(video_source=0):
    # Connect to the camera.
    # Change to video file if you want to use that instead.
    img_1 = np.zeros([512, 512, 3], dtype=np.uint8)
    img_1.fill(255)

    scene_heigt = 5
    scene_width = 5
    x = 2
    y = 4
    plot = utils.Projection2DPlot(img_1, scene_heigt, scene_width)

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Could not open video source {video_source}")
        return
    else:
        print(f"Successfully opened video source {video_source}")

    # Read the first frame.
    success, frame = cap.read()
    if not success:
        return
    with utils.ViewGui() as gui:
        while True:
            success, frame = cap.read()
            if not success:
                break

            plot_image = plot.plot_point(x, y)
            frame = utils.Projection2DPlot.add_to_frame(frame, plot_image, 0.25)
            frame = gui.show_frame(frame)
            key = gui.wait_key(1)

def run_Object_tracking(video_source=0):
    scene_detection = MotionDetectionScene.Scene(0.135, 0.004, 1, 2, 1.5, 1.5)
    scene_detection.run_scene_analyze(video_source)
    background_scene = np.zeros([512, 512, 3], dtype=np.uint8)
    background_scene.fill(255)


    scenex = scene_detection.sceneWidth
    sceney = scene_detection.sceneHeight

    utils.drawGridImage(background_scene, 0.1, scenex, sceney)

    tracker = Tracker.Tracker(5, 50, 4, 500)

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Could not open video source {video_source}")
        return
    else:
        print(f"Successfully opened video source {video_source}")

    # Read the first frame.
    success, frame = cap.read()
    if not success:
        return

    h = background_scene.shape[0]
    w = background_scene.shape[1]
    pixel_per_meter_x = w/scenex
    pixel_per_meter_y = h/sceney

    mode = 1

    with utils.ViewGui() as gui:
        while True:
            success, frame = cap.read()
            if not success:
                break

            #frame = cv2.undistort(frame, scene_detection.matrix_coefficients, scene_detection.distortion_coefficients)
            c0x, c0y = scene_detection.scene_pose.boardToCameraCoordinate(0, 0)
            c1x, c1y = scene_detection.scene_pose.boardToCameraCoordinate(scenex, 0)
            c2x, c2y = scene_detection.scene_pose.boardToCameraCoordinate(scenex, sceney)
            c3x, c3y = scene_detection.scene_pose.boardToCameraCoordinate(0, sceney)
           
            corner0 = (int(c0x+0.5), int(c0y+0.5))
            corner1 = (int(c1x+0.5), int(c1y+0.5))
            corner2 = (int(c2x+0.5), int(c2y+0.5))
            corner3 = (int(c3x+0.5), int(c3y+0.5))
            cv2.line(frame, corner0, corner1, (0,255,0), 3)
            cv2.line(frame, corner1, corner2, (0,0,255), 3)
            cv2.line(frame, corner2, corner3, (255,0,0), 3)
            cv2.line(frame, corner3, corner0, (255,255,255), 3)

            cv2.circle(frame, corner0, 10, (0, 255, 0), 10)

            tracker.detect(frame)
            if mode == 1:
                tracker.draw(frame)
            elif mode == 2:
                frame = tracker.dilated
            elif mode == 3:
                cv2.drawContours(frame, tracker.contours, -1, (0,255,0), 2)

            gui.show_frame(frame)

            if scene_detection.scene_pose:
                for ent in tracker.get_enitites():
                    x,y = scene_detection.scene_pose.cameraToboardCoordinate(ent.x, ent.y)
                    if x < 0 or y < 0:
                        continue
                    x *= pixel_per_meter_x
                    y *= pixel_per_meter_y
                    x = int(x)
                    y = int(y)
                    cv2.circle(background_scene, (x, h - y), 4, ent.color, 4)
                    

            gui.show_map(background_scene)

            key = gui.wait_key(1)

            if key == ord("q"):
                break

            if key == ord("1"):
                mode = 1
            if key == ord("2"):
                mode = 2
            if key == ord("3"):
                mode = 3

            if key == ord(" "):
                tracker.flush()
                background_scene.fill(255)
                utils.drawGridImage(background_scene, 0.1, scenex, sceney)



if __name__== '__main__':
    run_Object_tracking(2)
