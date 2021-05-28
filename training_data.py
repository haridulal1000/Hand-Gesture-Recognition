# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 00:29:14 2019

@author: Admin
"""

import cv2
import numpy as np
import os
x=4000
s=False
from pynput.mouse import Button,Controller
def No(x):
    pass
mouse=Controller()
cv2.namedWindow("Tracks")
cv2.createTrackbar('L-H','Tracks',0,255,No)
cv2.createTrackbar('L-S','Tracks',0,255,No)
cv2.createTrackbar('L-V','Tracks',0,255,No)
cv2.createTrackbar('H-H','Tracks',0,255,No)
cv2.createTrackbar('H-S','Tracks',0,255,No)
cv2.createTrackbar('H-V','Tracks',0,255,No)
cap=cv2.VideoCapture(0)
ret,frame=cap.read()
capt=False

while ret:
    ret,frame=cap.read()
    frame=cv2.GaussianBlur(frame,(15,15),0)
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    l_h=cv2.getTrackbarPos("L-H","Tracks")
    l_s=cv2.getTrackbarPos("L-S","Tracks")
    l_v=cv2.getTrackbarPos("L-V","Tracks")
    h_h=cv2.getTrackbarPos("H-H","Tracks")
    h_v=cv2.getTrackbarPos("H-S","Tracks")
    h_s=cv2.getTrackbarPos("H-V","Tracks")
    lower=np.array([l_h,l_s,l_v])
    upper=np.array([h_h,h_s,h_v])
    mask=cv2.inRange(hsv,lower,upper)
    cv2.imshow("mask",mask)
    
    #final=cv2.bitwise_and(frame,frame,mask=mask)
    #cv2.imshow("Final",mask)
    
    #cv2.imshow("img",frame)
    a=cv2.waitKey(1)
    #print(a)
    if a==115:
        capt=True
    if(capt==True):
        cv2.imwrite("img{:d}.jpg".format(x),mask)
        x+=1
        print(x)
    if a==27:
        break
cap.release()
cv2.destroyAllWindows()

"""


import cv2
import os
cap=cv2.VideoCapture(0)
ret,frame=cap.read()
x=0
path="D:/Py/One"
while ret:
    ret,frame=cap.read()
    cv2.imshow("Frame",frame)
    frame=cv2.resize(frame,(200,200))
    
    cv2.imwrite(os.path.join(path,'img{:d}.jpg').format(x),frame)
    x=x+1
    if cv2.waitKey(1)==27:
        break
cap.release()
cv2.destroyAllWindows()

"""
