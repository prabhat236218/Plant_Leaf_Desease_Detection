import numpy as np
import pickle
import cv2
from os import listdir
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

EPOCHS = 50
INIT_LR = 1e-3
BS =64
default_image_size = tuple((50,50))

directory_root = '/home/prabhat/PycharmProjects/mlpackage01/PlantVillage/PlantVillage'

width=50
height=50
depth=3


def convert_image_to_array(image_dir):
 try:
    image=cv2.imread(image_dir)
    #print("image--",image)
    if image is not None:
        image=cv2.resize(image,default_image_size)
        #print("image_to_array-->",img_to_array(image))
        return img_to_array(image)
    else:
        return np.array([])
 except Exception as e:
     print(f"Error : {e}")
     return None



image_list,label_list=[],[]

try:
    print("info ...loading images.....")
    root_dir=listdir(directory_root)
    for directory in root_dir:
        #print("directory-->",directory)
        if directory == ".DS_Store":
            root_dir.remove(directory)

    imageData=0
    #print("root_dir-->",root_dir)
    for plant_folder in root_dir:
        plant_disease_image_list = listdir(f"{directory_root}/{plant_folder}")
        #print("plant_disease_image_list===",plant_disease_image_list)
        #print("plant_folder-->",plant_folder)

        for single_plant_disease_image in plant_disease_image_list:
            if single_plant_disease_image == ".DS_Store":
                plant_disease_image_list.remove(single_plant_disease_image)
        #print("directory_root/plant_folder/plant_disease_image_list",directory_root,plant_folder,plant_disease_image_list)
        for image in plant_disease_image_list[:300]:
            image_directory=f"{directory_root}/{plant_folder}/{image}"
            imageData=imageData+1
            if image_directory.endswith(".jpg")==True or image_directory.endswith(".JPG")==True:
                image_list.append(convert_image_to_array(image_directory))
                label_list.append(plant_folder)
    print("imagaeData==",imageData)
    #print("image_list.size()-->",len(image_list))
    #print("image_list-->",image_list[0])

    np_image_list = np.array(image_list, dtype=np.float16) / 255.0
    #np_image_list=np_image_list.reshape(len(np_image_list),3,128,128)
    #print("np_image_list shape",np_image_list.shape)
    #print("np_image-->",np_image_list[0].shape)
    label_binarizer = LabelBinarizer()
    image_labels = label_binarizer.fit_transform(label_list)

    pickle.dump(label_binarizer,open('label_transform.pkl','wb'))
    n_classes=len(label_binarizer.classes_)
    x_train, x_test, y_train, y_test = train_test_split(np_image_list, image_labels, test_size=0.25, random_state=42)
    z=zip(x_test,y_test);
    pickle.dump(z,open('zipData2.pkl','wb'));

    #print("number of classes-->",n_class)
    #print("label-->",image_labels)
    #print("x train--",x_train)
    aug = ImageDataGenerator(
        rotation_range=25, width_shift_range=0.1,
        height_shift_range=0.1, shear_range=0.2,
        zoom_range=0.2, horizontal_flip=True,
        fill_mode="nearest")

    model = Sequential()
    inputShape = (height, width, depth)
    chanDim = -1
    if K.image_data_format() == "channels_first":
        inputShape = (depth, height, width)
        chanDim = 1
    model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))

    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))


    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(1024))
    model.add(Activation("relu"))


    model.add(Dense(512))
    model.add(Activation("relu"))

    model.add(Dense(256))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.75))

    model.add(Dense(n_classes))
    model.add(Activation("softmax"))
    #model.summary()
    opt = Adam(lr=INIT_LR, decay=INIT_LR /BS)

    # distribution
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
    # train the network
    print("[INFO] training network...")
    history = model.fit_generator(#x_train,y_train,epochs=2,batch_size=2
        aug.flow(x_train, y_train, batch_size=BS),
        validation_data=(x_test, y_test),
        steps_per_epoch=len(x_train) // BS,
        epochs=EPOCHS, verbose=1
    )

    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'r-', label='Training acc')
    plt.plot(epochs, val_acc, 'b-', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'r-', label='Training loss')
    plt.plot(epochs, val_loss, 'b-', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    
    plt.show();
    
    pickle.dump(model, open('cnn_model.pkl', 'wb'))
    print("saved successfully")
    print("[INFO] Calculating model accuracy")
    scores = model.evaluate(x_test, y_test)
    print(f"Test Accuracy: {scores[1] * 100}")

except Exception as e:
    print(f"Error : {e}")
