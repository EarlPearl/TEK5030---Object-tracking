import cv2
import numpy as np



def getQRPose(image):
    qrDecoder = cv2.QRCodeDetector()
    data, bbox, rectifiedImage = qrDecoder.detectAndDecode(image)
    if len(data) > 0:
        print("Decoded Data : {}".format(data))
        return data
    else:
        return -1


