# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 13:13:12 2019

@author: Admin
"""

import tensorflow as tf
import cv2
import numpy as np
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, Dropout
pickle_out=open("train_X.pickle","rb")
X=pickle.load(pickle_out)
pickle_out.close()
pickle_in2=open("train_y.pickle","rb")
y=pickle.load(pickle_in2)
pickle_in2.close()

X=X/255
train_dataset = tf.data.Dataset.from_tensor_slices((X, y))
test_dataset = tf.data.Dataset.from_tensor_slices((X, y))
BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 100

train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)
model=tf.keras.models.Sequential()
model.add(Conv2D(64,(3,3),input_shape=X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64,(3,3),input_shape=X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(32,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(100))
model.add(Activation("relu"))
model.add(Dropout(0.2))
model.add(Dense(100))
model.add(Activation("relu"))
model.add(Dense(4))
model.add((Activation("softmax")))
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="adam",
              metrics=['accuracy'])

model.fit(train_dataset, epochs=10)
model.evaluate(test_dataset)

model.save("Model1.model")

frame=cv2.imread("3.jpg",0)
frame=cv2.resize(frame,(50,50))
frame=frame/255
new_frame=frame.reshape(-1,50,50,1)
ans=model.predict(new_frame)
print(ans)
