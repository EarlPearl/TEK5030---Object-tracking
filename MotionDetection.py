import numpy as np
import cv2
import utils
from Entities import Entities


def run_motion_detction():
    video_source = 0
    mode = 1
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
            kernel = np.ones((9, 9))
            dilated = cv2.dilate(thresh, kernel, 1)

            contours, _ = cv2.findContours(image=dilated, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                if cv2.contourArea(contour) < 400:
                #too small!
                    continue

                (x, y, w, h) = cv2.boundingRect(contour)
                center = (x+w//2, y+h//2)
                entities.queue_point(center, (w//2, h//2))

            entities.update()
            if mode == 1:
                entities.draw(frame)
                out = frame
                
            if mode == 2:
                out = fgmask
                
            if mode == 3:
                out = thresh
                
            if mode == 4:
                out = frame
                cv2.drawContours(frame, contours, -1, (255,0,0), 1)
                            
            gui.show_frame(out)
            #cv2.drawContours(image=frame, contours=contours, contourIdx=-1, color=(0, 0, 255), thickness=1, lineType=cv2.LINE_AA)

            
            key = gui.wait_key(1)
            if key == ord("1"):
                mode = 1
            if key == ord("2"):
                mode = 2
            if key == ord("3"):
                mode = 3
            if key == ord("4"):
                mode = 4    

            if key == ord("q"):
                break            
            if key == ord(" "):
                entities.flush()

if __name__== '__main__':
    run_motion_detction()
