
import cv2 
import easyocr
import numpy as np 
import matplotlib.pyplot as plt 
import os
from datetime import datetime
from PIL import Image

#load YOLOv3 and YOLOv3 custom
net = cv2.dnn.readNet('/Users/nighttwinkle/Documents/ANPR-V0.2/Weights/yolov3_custom.weights', 
                      '/Users/nighttwinkle/Documents/ANPR-V0.2/YoloV3 cfg/yolov3_custom.cfg')
net_car = cv2.dnn.readNet('/Users/nighttwinkle/Documents/ANPR-V0.2/Weights/yolov3.weights',
                          '/Users/nighttwinkle/Documents/ANPR-V0.2/YoloV3 cfg/yolov3.cfg')
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

boxes = []
confidences = []
class_ids = []

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
            
#later add show function
        

#function to save files
def savefImg(file):
    save_date = datetime.now()
    path = f'/Users/nighttwinkle/Documents/ANPR-V0.2/Saved detection/{save_date}.jpeg'
    try:
        img = cv2.imwrite(path, file)
    except(cv2.error):
        return None
    return path

#create log file then after reading it, delete using os.remove()
#needs to be saved so can pass into readPlate function
#function to read from car plate
def readPlate(file):
    result = reader.readtext(file)
    result 
    img = cv2.imread(file)
    for detection in result:
        top_left = tuple([int(val) for val in detection[0][0]])
        bottom_right = tuple([int(val) for val  in detection[0][2]])
        text = detection[1]
        font = cv2.FONT_HERSHEY_SIMPLEX
        img = cv2.rectangle(img, top_left, bottom_right, (0,255,0), 5)
        img = cv2.putText(img, text, top_left, font, 2, (255,255,255), 2, cv2.LINE_AA)
    with open('number.txt', 'w') as n:
        n.write(text)
        #don't need to show the text, or make another plt to show text 

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






    






                



    

    
