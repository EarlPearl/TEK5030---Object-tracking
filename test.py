import cv2
import utils



def run_motion_detction():
    # Connect to the camera.
    # Change to video file if you want to use that instead.
    video_source = 0
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
    with utils.ViewGui("asd") as gui:
        
        while True:
            success, frame = cap.read()
            if not success:
                break

            gui.show_frame(frame)
            key = gui.wait_key(1)
            if key == ord("q"): break

run_motion_detction()