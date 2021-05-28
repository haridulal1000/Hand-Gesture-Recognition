# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 23:07:36 2019

@author: Admin
"""

import cv2
import numpy as np
import tensorflow as tf
from pynput.mouse import Controller,Button
clk="Yes"
mouse=Controller()
def No(x):
    pass
model=tf.keras.models.load_model("Model.model")
kernelOpen=np.ones((5,5))
kernelClose=np.ones((20,20))
cv2.namedWindow("Window")
cv2.createTrackbar("H-H","Window",0,255,No)
cv2.createTrackbar("H-S","Window",0,255,No)
cv2.createTrackbar("H-V","Window",0,255,No)
cv2.createTrackbar("L-H","Window",0,255,No)
cv2.createTrackbar("L-S","Window",0,255,No)
cv2.createTrackbar("L-V","Window",0,255,No)
liste=[]
cap=cv2.VideoCapture(0)
ret,frame=cap.read()
start=False
goal="None"
cate=["One","Two","Three","None"]
while True:
    if cv2.waitKey(10)==115:
        if start==False:
            start=True
        elif start==True:
            start=False
    ret,frame=cap.read()
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    l_h=cv2.getTrackbarPos("L-H","Window")
    l_s=cv2.getTrackbarPos("L-S","Window")
    l_v=cv2.getTrackbarPos("L-V","Window")
    h_h=cv2.getTrackbarPos("H-H","Window")
    h_v=cv2.getTrackbarPos("H-S","Window")
    h_s=cv2.getTrackbarPos("H-V","Window")
    lower=np.array([l_h,l_s,l_v])
    upper=np.array([h_h,h_s,h_v])
    frame=cv2.GaussianBlur(frame,(15,15),0)
    mask=cv2.inRange(hsv,lower,upper)
    cv2.imshow("Window",mask)
    mask2=mask
    maskOpen=cv2.morphologyEx(mask2,cv2.MORPH_OPEN,kernelOpen)
    maskClose=cv2.morphologyEx(mask2,cv2.MORPH_CLOSE,kernelClose)
    maskFinal=maskClose
    conts,h=cv2.findContours(maskFinal.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    if len(conts)<5000 and len(conts)>=1:
        areas=[cv2.contourArea(c) for c in conts]
        max_ind=np.argmax(areas)
        cnt=conts[max_ind]
        if cv2.contourArea(cnt)>1000:

            x1,y1,h1,w1=cv2.boundingRect(cnt)
            if start==True:
                x2,y2,h2,w2=cv2.boundingRect(cnt)
                centerX,centerY=(x2+h2/2,y2+w2/2)
                distX,distY=(320-centerX,240-centerY)
                dx=distX
                dy=distY
                dx1=dx
                dy1=dy
                if abs(distX)>20 and abs(distY)>20:
                    mouse.move(dx/10,-dy/10)
                    clk="None"
                else:
                    clk="Yes"


                mask=cv2.resize(mask,(50,50))
        
                mask=mask/255
        
        
        
            
            
                mask=mask.reshape(-1,50,50,1)
                ans=model.predict(mask)
                state=cate[np.argmax(ans)]
                if clk!="None" or goal=="None":
                    if state=="One":
                        mouse.click(Button.left,2)
                        goal="Free"
                    else:
                        goal="None"


                    

                   
    print(start)
    if cv2.waitKey(10)==27:
        break
cv2.destroyAllWindows()
cap.release()
        