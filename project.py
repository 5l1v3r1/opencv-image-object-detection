import numpy as np
import cv2
from imageai.Detection import ObjectDetection
import os
import threading


def cap_frame():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    while True:

        ret, frame = cap.read()
        cv2.imshow('FaceDetection', frame)
        k = cv2.waitKey(1)
        if k % 256 == 27: 
            break

        elif k % 256 == 32:
            img_name = "captured-image.jpg"
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            break

    cap.release()
    cv2.destroyAllWindows()


def process_frame():
    execution_path = os.getcwd()
    detector = ObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath(os.path.join(execution_path, "yolo.h5"))
    detector.loadModel()
    detections = detector.detectObjectsFromImage(input_image=os.path.join(
        execution_path, "captured-image.jpg"), output_image_path=os.path.join(execution_path, "processed-image.jpg"), minimum_percentage_probability=50)


def show_frame():
    image=cv2.imread("processed-image.jpg")
    cv2.imshow('PROCESSED-IMAGE',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":

    t1 = threading.Thread(target=cap_frame)
    t2 = threading.Thread(target=process_frame)
    cap_frame()
    print(">>PROCESS FOR IMAGE DETECTION ?  [Y/N]")
    choice = input(">> ")
    if choice == "Y":
        process_frame()
    else:
        sys.exit()

    show_frame()
