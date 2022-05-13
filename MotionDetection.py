import numpy as np
import cv2
import utils
import Entities


def run_motion_detction():
    video_source = 2
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Could not open video source {video_source}")
        return
    else:
        print(f"Successfully opened video source {video_source}")

    mog = cv2.createBackgroundSubtractorMOG2()
    entities = Entities()
    with utils.ViewGui() as gui:
        while True:
            
            success, frame = cap.read()

            if not success:
                break

            fgmask = mog.apply(frame, 123, 0.01)
            
            blured = cv2.GaussianBlur(src=fgmask, ksize=(9, 9), sigmaX=0)
            _, thresh = cv2.threshold(blured, 220, 255, cv2.THRESH_BINARY)

            # 4. Dilute the image a bit to make differences more seeable; more suitable for contour detection
            kernel = np.ones((5, 5))
            dilated = cv2.dilate(thresh, kernel, 1)
            
            contours, _ = cv2.findContours(image=dilated, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
            #cv2.drawContours(image=frame, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2,
            #                 lineType=cv2.LINE_AA)


            for contour in contours:
                if cv2.contourArea(contour) < 400:
                #too small!
                    continue

                (x, y, w, h) = cv2.boundingRect(contour)
                cv2.rectangle(img=frame, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 0), thickness=2)
                center = (x+w//2, y+h//2)
                entities.queue_point(center)

            col = cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR)
            col[:,:,2] = 0
            col[:,:,1] = 0

            ttt = cv2.addWeighted(frame, 1, col, 0.40, 0.0)

            entities.update()
            entities.draw(ttt)

            gui.show_frame(ttt)
            key = gui.wait_key(1)
            if key == ord("q"):
                break            
            if key == ord(" "):
                pass


def main():
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

    # Stop video source.
    # cap.release()


if __name__== '__main__':
    run_motion_detction()
