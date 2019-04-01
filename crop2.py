import tkinter as tk
from tkinter.filedialog import askopenfilename
import shutil
import os
import sys
from PIL import Image, ImageTk
from orca.debug import println
import io
import base64
import numpy as np
import pickle
import cv2
from os import listdir
from PIL import Image
import matplotlib.pyplot as plt
from keras.preprocessing.image import img_to_array

default_image_size = tuple((64,64))
directory_root = 'PlantVillage/PlantVillage'


def convert_image(image_dir):
 try:
    image=cv2.imread(image_dir)
    #print("image--",image)
    if image is not None:
        image=cv2.resize(image,default_image_size)
        #cv2.imshow('image',image)
        #cv2.waitKey(0)
        #print("image_to_array-->",img_to_array(image))
        return img_to_array(image)
    else:
        return np.array([])
 except Exception as e:
     print(f"Error : {e}")
     return None



model_file = "cnn_model.pkl"
saved_classifier_model = pickle.load(open(model_file, 'rb'))
#print("prabhat")
image="/home/prabhat/PycharmProjects/mlpackage01/PlantVillage/PlantVillage/Pepper__bell___Bacterial_spot/0a0dbf1f-1131-496f-b337-169ec6693e6f___NREC_B.Spot 9241.JPG"

image=convert_image(image)
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)
prediction = saved_classifier_model.predict(image)
idx = np.argmax(prediction)

lb = pickle.load(open("label_transform.pkl", 'rb'))
label = lb.classes_[idx]
print("classes_==",lb.classes_)

print("label==",label)
print("idx==",idx)
#print("np.argmax==",np.argmax(prediction))
#print("prediction==",prediction)
