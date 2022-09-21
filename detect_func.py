
import os
from datetime import datetime

import cv2
import easyocr
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

#load YOLOv3 and YOLOv3 custom
net = cv2.dnn.readNet('/Users/nighttwinkle/Documents/ANPR-V0.2/Weights/yolov3_plate.weights', 
                      '/Users/nighttwinkle/Documents/ANPR-V0.2/YoloV3 cfg/yolov3_plate.cfg')
net_car = cv2.dnn.readNet('yolov3_land_vehicle.weights', 'yolov3_land_vehicle.cfg')
#easyOCR pipeline
reader = easyocr.Reader(['en'], gpu=True)

#classes 
vehicle_class = []
with open('coco.names', 'r') as vc:
    vehicle_class = vc.read().splitlines()

plate_class = []
with open('obj.names', 'r') as pc:
    plate_class = pc.read().splitlines()


font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(100, 3))
text = "Press 'q' to quit"

boxes = []
confidences = []
class_ids = []

#fucntion to detect plate
def detectPlate(file):
    img = cv2.imread(file)
    height, width, _ = img.shape

    blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
    net.setInput(blob)

    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)
    if len(indexes)>0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            confidence = str(round(confidences[i],2))
            color = colors[i]
            cv2.rectangle(img, (x,y), (x+w, y+h), color, 1)
            crop_img = img[y:y+h, x:x+w]
            saved = savefImg(crop_img)
            return saved

#function to save files
def savefImg(file):
    save_date = datetime.now()
    path = f'/Users/nighttwinkle/Documents/ANPR-V0.2/Saved detection/{save_date}.jpeg'
    try:
        img = cv2.imwrite(path, file)
    except(cv2.error):
        return None
    return path

#function to read text from plate
def readPlate(file):
    result = reader.readtext(file)
    result 
    img = cv2.imread(file)
    for detection in result:
        top_left = tuple([int(val) for val in detection[0][0]])
        bottom_right = tuple([int(val) for val  in detection[0][2]])
        text = detection[1]
        #saving the result into csv file
        with open('/Users/nighttwinkle/Documents/ANPR-V0.2/Saved detection/Car plate and color/numbers.csv', 'w') as n:
            n.write(text)

#function to detect car
def detectCar(file):
    img = cv2.imread(file)
    height, width, _ = img.shape

    blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
    net_car.setInput(blob)

    output_layers_names = net_car.getUnconnectedOutLayersNames()
    layerOutputs = net_car.forward(output_layers_names)

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)
    
    if len(indexes)>0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            confidence = str(round(confidences[i],2))
            color = colors[i]
            cv2.rectangle(img, (x,y), (x+w, y+h), color, 4)
            crop_img = img[y:y+h, x:x+w]
            saved = savefImg(crop_img)
            return saved

def detectFromVideo(file):

    cap = cv2.VideoCapture(file)

    while True: 
        _, img = cap.read()
        height, width, _ = img.shape

        blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
        net_car.setInput(blob)
        output_layers_names = net.getUnconnectedOutLayersNames()
        layerOutputs = net_car.forward(output_layers_names)

        boxes = []
        confidences = []
        class_ids = []

        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.8:
                    center_x = int(detection[0]*width)
                    center_y = int(detection[1]*height)
                    w = int(detection[2]*width)
                    h = int(detection[3]*height)

                    x = int(center_x - w/2)
                    y = int(center_y - h/2)

                    boxes.append([x, y, w, h])
                    confidences.append((float(confidence)))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)
        if class_id == 'car':
            if len(indexes)>0:
                for i in indexes.flatten():
                    x, y, w, h = boxes[i]
                    label = str(vehicle_class[class_ids[i]])
                    confidence = str(round(confidences[i],2))
                    color = colors[i]
                    cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
                    crop_img = img[y:y+h, x:x+w]
                    saved = savefImg(crop_img)
                    plate_img = detectPlate(saved)
                    plate_text = readPlate(plate_img)
        cv2.putText(img,  text, (100,100), font, 2, (255,255,255), 2)
        cv2.imshow('Image', img)
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
        
def detectFromCam():

    cap = cv2.VideoCapture(0)

    while True: 
        _, img = cap.read()
        height, width, _ = img.shape

        blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
        net_car.setInput(blob)
        output_layers_names = net.getUnconnectedOutLayersNames()
        layerOutputs = net_car.forward(output_layers_names)

        boxes = []
        confidences = []
        class_ids = []

        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.8:
                    center_x = int(detection[0]*width)
                    center_y = int(detection[1]*height)
                    w = int(detection[2]*width)
                    h = int(detection[3]*height)

                    x = int(center_x - w/2)
                    y = int(center_y - h/2)

                    boxes.append([x, y, w, h])
                    confidences.append((float(confidence)))
                    class_ids.append(class_id)
        if class_ids == 'car' or 'motobike' or 'bus' or 'truck':
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)
            if len(indexes)>0:
                for i in indexes.flatten():
                    x, y, w, h = boxes[i]
                    label = str(vehicle_class[class_ids[i]])
                    confidence = str(round(confidences[i],2))
                    color = colors[i]
                    cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
                    crop_img = img[y:y+h, x:x+w]
                    saved = savefImg(crop_img)
                    plate_img = detectPlate(saved)
            cv2.putText(img,  text, (100,100), font, 2, (255,255,255), 2)
            cv2.imshow('Image', img)
            if cv2.waitKey(1) & 0xFF==ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()











        






                



    

    
