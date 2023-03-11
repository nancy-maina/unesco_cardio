# importing libraries
from asyncore import write
from fileinput import filename
from locale import normalize
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import streamlit as st
import joblib
import random
import glob # to find files
from IPython.display import Image
import h5
# Seaborn library for bar chart
import seaborn as sns
import tensorflow
import tensorflow_hub as hub
import keras
# Libraries for TensorFlow
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import image
from tensorflow.keras import models, layers
from PIL import Image,ImageOps
# Library for Transfer Learning
# from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
import matplotlib.pyplot as plt 
img_width=128
img_height=128

# variable for model
batch_size=64
epochs=10

st.title("CARDIOMEGALY DETECTION ")

st.write("This application is being developed to examine the accurate diagnosis of cardiomegaly on frontal chest radiographs ")

from joblib import dump, load

# path = "D:\\Unesco\\chest-xray-pneumonia1\\chest_xray"

# # train directory
# train_folder=path+"\\train"
# train_normal_dir=train_folder+"\\NORMAL"
# train_pneu_dir=train_folder+"\\PNEUMONIA"
# # test directory
# test_folder=path+"\\test"
# test_normal_dir=test_folder+"\\NORMAL"
# test_pneu_dir=test_folder+"\\PNEUMONIA"
# # validation directory
# val_folder=path+"\\val"
# val_normal_dir=val_folder+"\\NORMAL"
# val_pneu_dir=val_folder+"\\PNEUMONIA"

# # variables for image size
# img_width=128
# img_height=128

# # variable for model
# batch_size=64
# epochs=10

# train_class_names=os.listdir(train_folder)
# test_class_names=os.listdir(test_folder)
# val_class_names=os.listdir(val_folder)

# def Get_Xray_Type(argument):
#     switcher = {
#         "NORMAL": "Normal",
#         "PNEUMONIA": "Pneumonia",
#     }
#     return switcher.get(argument, "Invalid X-ray")

# train_normal_cases = glob.glob(train_normal_dir + '*png')
# train_pneu_cases = glob.glob(train_pneu_dir + '*png')

# test_normal_cases = glob.glob(test_normal_dir + '*png')
# test_pneu_cases = glob.glob(test_pneu_dir + '*png')

# val_normal_cases = glob.glob(val_normal_dir + '*png')
# val_pneu_cases = glob.glob(val_pneu_dir + '*png')

# train_list = []
# test_list = []
# val_list = []

# for x in train_normal_cases:
#     train_list.append([x, "Normal"])
    
# for x in train_pneu_cases:
#     train_list.append([x, "Pneumonia"])
    
# for x in test_normal_cases:
#     test_list.append([x, "Normal"])
    
# for x in test_pneu_cases:
#     test_list.append([x, "Pneumonia"])
    
# for x in val_normal_cases:
#     val_list.append([x, "Normal"])
    
# for x in val_pneu_cases:
#     val_list.append([x, "Pneumonia"])

# # create dataframes
# train_df = pd.DataFrame(train_list, columns=['image', 'Diagnos'])
# test_df = pd.DataFrame(test_list, columns=['image', 'Diagnos'])
# val_df = pd.DataFrame(val_list, columns=['image', 'Diagnos'])

# # result= st.button("Plot of Distribution of Dataset")
# st.dataframe(train_df)
# try:
#     result = st.button("Plot of Distribution of Dataset")
# except ValueError:
#     result = 0

# st.write(result)
# if result:
#     fig = plt.figure(figsize = (20,5))
    

#     plt.subplot(1,3,1)
#     # plt.countplot(train_df['Diagnos'])
#     plt.title('Train data')
    
#     # plt.subplot(1,3,2)
#     # fig2 = plt.figure(figsize = (20,5))
#     # sns.countplot(test_df['Diagnos'])
#     # st.pyplot(fig2)
#     plt.title('Test data')

#     plt.subplot(1,3,3)
#     # sns.countplot(val_df['Diagnos'])
#     plt.title('Validation data')
#     plt.legend()
#     st.pyplot(fig)

#     plt.subplot(1,3,2)

#     fig2 = plt.figure(figsize=(20,5))
#     sns.countplot(data=test_df['Diagnos'])
    
#     st.pyplot(fig2)
#     # plt.legend()

import io

def main():
    file_uploaded= st.file_uploader('Choose the file',type=['jpg','png','jpeg'])
    if file_uploaded is not None:
        image2=tensorflow.keras.utils.load_img(file_uploaded,target_size=(img_width,img_height))
        # image2=Image.open(file_uploaded)
        figure=plt.figure(figsize=(20,5))
        plt.imshow(image2 ,cmap='gray')
        plt.axis('off')
        result=predict_class(image2)
        st.write(result)
        st.pyplot(figure)



def predict_class(image1):
    classifier_model = tensorflow.keras.models.load_model("C:\\Users\\tanma\\Downloads\\model_name.h5")
    # img=image.load_img(image1,target_size=(img_width,img_height))
    img=image.img_to_array(image1)
    
    img=img.astype('int32')
    # fig2 = plt.figure(figsize = (20,5))
    # plt.imshow(img.astype('int32'))
    # # plt.show()
    # st.pyplot(fig2)

    img=preprocess_input(img)
#     plt.imshow(img.astype('int32'))
#     plt.show()
    class_names=['CARDIOMEGALY','NORMAL']
    prediction=classifier_model.predict(img.reshape(1,img_width,img_height,3))
    output=np.argmax(prediction)
    # print(train_class_names[output] + ": " + Get_Xray_Type(train_class_names[output]))
    st.write(class_names[output])

    # shape=((128,128,3))
    # model=tensorflow.keras.Sequential(hub.KerasLayer(classifier_model,input_shape=shape))
    # # test_image=image.reshape((1,128,128,3))
    # test_image=image
    # test_image=tensorflow.keras.preprocessing.image.img_to_array(test_image)
    # test_image=np.expand_dims(test_image,axis=0)
    # class_names=['PNEUMONIA','NORMAL']
    # predictions=classifier_model.predict(test_image)
    # # scores=tensorflow.nn.softmax(predictions[0])
    # # scores=scores.numpy()
    # image_class=class_names[np.argmax(predictions)]
    # result="The image uploaded is : {}".format(image_class)
    # return result

if __name__ =="__main__" :
    main()






