import os
import cv2
import tkinter as tk 
from tkinter import filedialog
import detect_func as df
import matplotlib.pyplot as plt
from detect_func import detectCar, detectFromVideo, detectPlate 


#function for main window
def mainwin():
    global root 
    root = tk.Tk()
    root.geometry('200x200')
    root.title("ANPR")


#function to call filedialog
def callback():
    root.filename = filedialog.askopenfilename(parent=root, 
                                               initialdir= "/path/to/start",
                                               title = "Choose a file")
    filetype(root.filename)

#function to check filetype
def filetype(file):
    if file.endswith('.jpeg'):
        detectFromImg(file)
    else: 
        detectFromVideo(file)

#function to detect from image
def detectFromImg(file):
    car_img = df.detectCar(file)
    plate_img = df.detectPlate(file)
    text_img = df.readPlate(file) 


    #showing preview of detection
    car_imgPLT = plt.imread(car_img)
    plate_imgPLT = plt.imread(plate_img)
    row = 1 
    columns = 3
    fig = plt.figure(figsize=(10,10))
    fig.add_subplot(row, columns, 1)
    plt.imshow(car_imgPLT)
    fig.add_subplot(row, columns, 2)
    plt.imshow(plate_imgPLT) #need to g
    plt.show()

























