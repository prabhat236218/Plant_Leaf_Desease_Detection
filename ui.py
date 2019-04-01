import tkinter as tk
from tkinter.filedialog import askopenfilename
import shutil
import pickle
from matplotlib import pyplot as plt
import os
import sys
import cv2  # working with, mainly resizing, images
import numpy as np  # dealing with arrays
import os  # dealing with directories
from random import shuffle  # mixing up or currently ordered data that might lead our network astray in training.
from tqdm import \
    tqdm  # a nice pretty percentage bar for tasks. Thanks to viewer Daniel BA1/4hler for this suggestion

import cv2
from PIL import Image, ImageTk
from keras_preprocessing.image import img_to_array
from orca.debug import println

default_image_size = tuple((50,50))

window = tk.Tk()

window.title("Plant leaf Deseas Detector")

window.geometry("500x500")

window.configure(background ="green")

title = tk.Label(text="Click below to choose picture for testing disease....", background = "red", fg="White", font=("", 15))
title.grid()


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

def analysis():

    verify_dir = 'testpicture'

    img1 = os.listdir(verify_dir)
    print("img1==", img1[0])
    path1 = os.path.join(verify_dir, img1[0])

    model_file = "cnn_model.pkl"
    model = pickle.load(open(model_file, 'rb'))
    z=pickle.load(open('zipData.pkl','rb'))

    #for i in z:
    #    img=i[0].astype("float")
    #    print("image ki shape==",img.shape)
    #    plt.imshow(img)
    #    #print(img)
    #    break;
    #image = "/home/prabhat/PycharmProjects/mlpackage01/PlantVillage/PlantVillage/Pepper__bell___Bacterial_spot/0abffc81-6be8-4b17-a83c-4d2830e30382___JR_B.Spot 9076.JPG"
    image=path1
    image = convert_image(image)
    #print("imag-==",image)
    image = image.astype("float") / 255.0
    #print("image array==",image)
    #image = img_to_array(image)
    #print("image dim1==", image.shape)
    #plt.imshow(image)

    image = np.expand_dims(image, axis=0)
    #print("image dim2==",image.shape)

    prediction = model.predict(image)
    idx = np.argmax(prediction)
    lb = pickle.load(open("label_transform.pkl", 'rb'))
    label = lb.classes_[idx]
    #print("classes_==", lb.classes_)
    diseasename =lb.classes_[idx]
    disease = tk.Label(text='Disease Name: ' + diseasename, background="lightgreen",
                               fg="Black", font=("", 15))
    disease.grid(column=0, row=4, padx=10, pady=10)
    plt.show()


def openphoto():
    dirPath = "testpicture"
    fileList = os.listdir(dirPath)
    for fileName in fileList:
        os.remove(dirPath + "/" + fileName)
        
    fileName = askopenfilename(initialdir='image', title='Select image for analysis ',
                           filetypes=[('image files', '.jpg')])
    dst = "testpicture"
    shutil.copy(fileName, dst)
    load = Image.open(fileName)
    render = ImageTk.PhotoImage(load)
    img = tk.Label(image=render, height="250", width="500")
    img.image = render
    img.place(x=0, y=0)
    img.grid(column=0, row=1, padx=10, pady = 10)
    title.destroy()
    button1.destroy()
    button2 = tk.Button(text="Analyse Image", command=analysis)
    button2.grid(column=0, row=2, padx=10, pady = 10)
button1 = tk.Button(text="Get Photo", command = openphoto)
button1.grid(column=0, row=1, padx=10, pady = 10)

window.mainloop()
