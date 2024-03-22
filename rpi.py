import os
import math
import sys
import random
import neat
import cv2
import pickle
import RPi.GPIO as GPIO        
import numpy as np

from time import sleep

import time
from gpiozero import Buzzer
buzzer = Buzzer(22)

from neat import nn, population

import os
from math import floor
from adafruit_rplidar import RPLidar

from gpiozero import DistanceSensor
ultrasonic = DistanceSensor(echo=17, trigger=4, threshold_distance=0.5)


# Setup the RPLidar
PORT_NAME = '/dev/ttyUSB0'
lidar = RPLidar(None, PORT_NAME, timeout=3)

#setting up config for Object Detection 
classNames = []
classFile = "/home/pi/Desktop/Object_Detection_Files/Objects.names"
with open(classFile, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

configPath = "/home/pi/Desktop/Object_Detection_Files"
weightsPath = "/home/pi/Desktop/Object_Detection_Files"

#using OpenCV library
net = cv2.dnn_DetectionModel(weightsPath, configPath)

net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 12))
net.setInputSwapRB(True)

# used to scale data to fit on the screen
max_distance = 0 
in1 = 17
in2 = 22
en = 27

temp1=1

#GPIO pins for motor, button, and ultrasonic sensor
GPIO.setmode(GPIO.BCM)
GPIO.setup(in1,GPIO.OUT)
GPIO.setup(in2,GPIO.OUT)
GPIO.setup(en,GPIO.OUT)
GPIO.output(in1,GPIO.LOW)
GPIO.output(in2,GPIO.LOW)
p=GPIO.PWM(en,1000)
p.start(25)
GPIO.output(in1,GPIO.LOW)
GPIO.output(in2,GPIO.LOW)
GPIO.setmode(GPIO.BCM)
button_pin = 2
GPIO.setup(button_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
run = True

#function for identifying objects and turning on buzzer 
def getObjects(img, thres, nms, objects=[]):
  classIds, confs, bbox = net.detect(img, confThreshold=thres,  nmsThreshold=nms)
  objectInfo = []
  if len(classIds) != 0:
      for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
          className = classNames[classId - 1]
          if className in objects:
              objectInfo.append([box, className])
              buzzer.on()
              time.sleep(1)
              buzzer.off()

#continous loop for main function
def main(run):
  
    #checking if start button has been pressed 
    if run == True:
        #getting config file
        config_path = "/home/nirmitshah/projects/pickle/config-feedforward.txt"
        config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                    neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

        #getting pickle file
        with open('/home/nirmitshah/Downloads/Py_pickle_1', 'rb') as file:
            loaded_model = pickle.load(file)

        #creating array for array of inputs 
        scan_data = [0, 0 , 0, 0, 0, 0]

        #creating neural network with pickle file 
        neat_net = neat.nn.FeedForwardNetwork.create(loaded_model, config)

        while(1):

            #again checking if button is pressed 
            button_state = GPIO.input(button_pin)
            if button_state == False:
                break
            j = 0

            #getting inputs from lidar
            for scan in lidar.iter_scans():
                button_state = GPIO.input(button_pin)
                if button_state == False:
                    break
             
                #creating range of values to gaurantee that the values are within the range of the lidar and inputting into array 
                for (_, angle, distance) in scan:
                    if 0 < floor(angle) < 10:
                        scan_data[0] = distance
                    if 40 < floor(angle) < 50:
                        scan_data[1] = distance
                    if 85 < floor(angle) < 95:
                        scan_data[2] = distance
                    if 265 < floor(angle) < 275:
                        scan_data[3] = distance
                    if 310 < floor(angle) < 320:
                        scan_data[4] = distance

                #sixth input comes from ultrasonic sensor
                scan_data[5] = ultrasonic.distance

                #reading input from camera 
                image = cap.read()

                #filtering input for edge detection 
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 150)

                #utilizng library to identify lines               
                lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
                filtered_lines = []

                sideWalkLen = 5.45
                pos = sideWalkLen/2;

                #filtering lines for only lines that are edges of the sidewalk (vertical and significant lenght)
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                    if 80 < angle < 100 and length > 50:
                        filtered_lines.append(line)

                #finding x intercepts of lines and truncating the scan_data accordingly
                if len(filtered_lines) > 1:
                    x1, y1, x2, y2 = filtered_lines[0]
                    x3, y3, x4, y4 = filtered_lines[1]
                    m1 = (y2 - y1) / (x2 - x1)
                    m2 = (y4 - y3) / (x4 - x3)
                    b1, b2 = y1 - m1 * x1, y3 - m2 * x3
                    xInt1 = -b1/m1
                    xInt2 = -b2/m2
                    if xInt1 > xInt2:
                      d1 = xInt1 - pos
                      if scan_data[4] > d1:
                        scan_data[4] = d1
                      d2 = pos - xInt2
                      if scan_data[0] > d2:
                        scan_data[0] = d2
                elif len(filtered_lines) > 0:
                  x1, y1, x2, y2 = filtered_lines[0]
                  m1 = (y2 - y1) / (x2 - x1)
                  b1 = y1 - m1 * x1
                  xInt1 = -b1/m1
                  if xInt1 > pos:
                    d1 = xInt1 - pos
                    scan_data[4] = d1
                  else:
                    d1 = pos - xInt1
                    scan_data[0] = d1

                #utilizing nueral network with inputs from lidar and ultrasonic sensor 
                output = neat_net.activate(scan_data)
                i = output.index(max(output))
                print(output)
              
                #code for motor ouput depending on models ouput 
                if i == 0:
                    p.ChangeDutyCycle(75)
                    GPIO.output(in1,GPIO.HIGH)
                    GPIO.output(in2,GPIO.LOW)
                    print("forward")
                if i == 1:
                    p.ChangeDutyCycle(75)
                    GPIO.output(in1,GPIO.LOW)
                    GPIO.output(in2,GPIO.HIGH)
                    print("backward")
                if i == 2:
                    p.ChangeDutyCycle(25)
                    GPIO.output(in1,GPIO.LOW)
                    GPIO.output(in2,GPIO.LOW)
                    print("stop")

if __name__ == "__main__":

    #video capture for object detection 
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
  
    while True:
        
        button_state = GPIO.input(button_pin)
        
        success, img = cap.read()
        objectInfo = getObjects(img, 0.45, 0.2)
        cv2.waitKey(1)
      
        main(run)