# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 12:57:05 2019

@author: Admin
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 22:58:17 2019

@author: Admin
"""

import numpy as np
import cv2 
import os
training_data=[]
x=0
DATADIR="D:/Py/tensorflow/Data"
CATEGORIES=["Fist","None","One","Three"]
for category in CATEGORIES:
    path=os.path.join(DATADIR,category)
    class_num=CATEGORIES.index(category)
    for img in os.listdir(path):
        try:
            img_array=cv2.imread(os.path.join(path,img),0)
            IMG_SIZE=50
            new_array=cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
            training_data.append([new_array,class_num])
            print("{}".format(x))
            x+=1
        except Exception as e:
            pass
        
import random
random.shuffle(training_data)
random.shuffle(training_data)
for i in training_data[:10]:
    print(i[1])
X=[]
y=[]
for x_,y_ in training_data:
    X.append(x_)
    y.append(y_)
X=np.array(X).reshape(-1,50,50,1)
import pickle
pickle_out=open("train_X.pickle","wb")
pickle.dump(X,pickle_out)
pickle_out.close()
pickle_out=open("train_y.pickle","wb")
pickle.dump(y,pickle_out)
pickle_out.close()
print("Done")
    
        
        