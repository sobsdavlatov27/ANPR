import os
from subprocess import call #needed to manipulate files 
import cv2 #maybe deleted 
import tkinter as tk 
from datetime import datetime #only needed to save files wihtout overwriting them
from tkinter import filedialog
import detect_func as df
import matplotlib.pyplot as plt

from detect_func import detectCar, detectPlate


#fucntion to initialize main window
def mainwin():
    global root 
    root = tk.Tk()
    root.geometry('200x200')
    root.title("ANPR")
#filedialog function 
def callback():
    root.filename = filedialog.askopenfilename(parent=root, 
                                               initialdir= "/path/to/start",
                                               title = "Choose a file")
    filetype(root.filename)
#filetype check function, cheks then uses two functions to detect from image and detect from video
def filetype(file):
    if file.endswith('.jpeg'):
        detectFromImg(file)
    else: 
        pass #video detection function


#funtion to get colro of a vehicle
def detectFromImg(file):
    #saves images of detection and text
    car_img = df.detectCar(file)
    plate_img = df.detectPlate(file)
    text_img = df.readPlate(file)
 #add 3 subplots into 1 plot



def detectFromVideo(file):
    pass 

detectFromImg('/Users/nighttwinkle/Downloads/Testing images/ford_vtti_research_04_hr_1280x720.jpeg')













