from ultralytics import RTDETR,YOLO
import torch
import cv2
import time
torch.cuda.empty_cache()
def main():
    conf = 0.3
    min_distance = 40 # min distance
    #speed: {'preprocess': 3.0002593994140625, 'inference': 46.000003814697266, 'postprocess': 0.99945068359375}]
    #model = RTDETR('rtdetr-x.pt')
    #speed: {'preprocess': 2.9997825622558594, 'inference': 29.50763702392578, 'postprocess': 0.0}]
    model = RTDETR('best7.pt')
    #speed: {'preprocess': 1.0006427764892578, 'inference': 35.5069637298584, 'postprocess': 1.0006427764892578}]
    #model = YOLO('yolov8x.pt')
    #speed: {'preprocess': 1.0073184967041016, 'inference': 6.002187728881836, 'postprocess': 0.99945068359375}]
    #model = YOLO('yolov8n.pt')

    # Open the video file or camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error opening video stream or file")


    while True:
        # Read a new frame
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        print("Frame read successfully")

        result = model.predict(source='sample3.mp4', show=True, conf=0.6)
        print(result)

      
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()