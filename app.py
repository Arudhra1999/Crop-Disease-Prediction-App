#!/usr/bin/env python
# coding: utf-8

# In[15]:


import streamlit as st
from keras.models import load_model
import numpy as np
import cv2
from PIL import Image
import lime
from lime import lime_image
import time
from skimage.segmentation import mark_boundaries


# Loading the model (Created and trained by Ayoub.Berdeddouch)

# In[3]:


model = load_model("EfficientNetB3-landscape-99.65.h5")


# In[4]:


# for dirpath,dirname,filename in os.walk("Crop_disease_prediction"):
#      print(f"There are {len(dirname)} directories and {len(filename)} images in {dirpath}.")


# class names are hardcoded

# In[5]:


class_names = ['Apple___Apple_scab','Apple___Black_rot','Apple___Cedar_apple_rust',
 'Apple___healthy' ,'Blueberry___healthy',
 'Cherry_(including_sour)___Powdery_mildew',
 'Cherry_(including_sour)___healthy',
 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight',
 'Corn_(maize)___healthy', 'Grape___Black_rot',
 'Grape___Esca_(Black_Measles)',
 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
 'Peach___healthy', 'Pepper,_bell___Bacterial_spot',
 'Pepper,_bell___healthy' ,'Potato___Early_blight' ,'Potato___Late_blight',
 'Potato___healthy' ,'Raspberry___healthy' ,'Soybean___healthy',
 'Squash___Powdery_mildew' ,'Strawberry___Leaf_scorch',
 'Strawberry___healthy' ,'Tomato___Bacterial_spot', 'Tomato___Early_blight',
 'Tomato___Late_blight' ,'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
 'Tomato___healthy']


# In[6]:


len(class_names)


# In[7]:


healthy_cls = [3,4,10,14,17,19,22,23,24,27,37]


# In[8]:


# model.summary()


# Testing the image prediction before deploying into production

# In[9]:


# test_img = Image.open("E://Omdena//Crop_disease_prediction//train//Apple___Black_rot//8a1bb676-5d3e-423b-a667-c72d099215ab___JR_FrgE.S 2842.JPG")
# test_img = np.array(test_img.convert("RGB"))
# test_img = cv2.resize(test_img,(224,224),interpolation = cv2.INTER_NEAREST)
# plt.imshow(test_img)
# plt.axis("off")
# test_img = np.expand_dims(test_img,axis=0)
# prediction = model.predict(test_img)
# prediction_cls = np.argmax(prediction)
# print(class_names[prediction_cls])


# In[28]:


st.title("Omdena Crop Disease Prediction using Machine and Deep Learning")


# In[11]:


st.subheader("Model used : Efficient-Net")
st.subheader("Model accuracy : 99.65%")


# In[12]:


uploaded_file = st.file_uploader("Upload your image for prediction",type = ['jpg','jpeg','png','webp'])


# In[26]:


if st.button("Process"):
        test_image = Image.open(uploaded_file)
        test_image = np.array(test_image.convert('RGB'))
        test_image = cv2.resize(test_image, (224,224),interpolation=cv2.INTER_NEAREST)
        st.image(test_image,caption="Your input image")
        test_image = np.expand_dims(test_image,axis=0)
        st.write("Processing the image for prediction...")
        
        progress = st.progress(0)
        progress_text = st.empty()
        
        for i in range(101):
            time.sleep(0.2)
            progress.progress(i)
            progress_text.text(f"Progress:{i}%")
            
        probs = model.predict(test_image)
        pred_class = np.argmax(probs)

        pred_class_name = class_names[pred_class]
        
        if pred_class in healthy_cls:
            msg = f'Your crop-leaf is healthy and predicted class is {pred_class_name} with probability of {int(probs[0][pred_class]*100)}%'
            st.success(msg)
        else:
            msg = f'Your crop-leaf is not healthy and predicted class is {pred_class_name} with probability of {int(probs[0][pred_class]*100)}%'
            st.error(msg)
        
        st.subheader("Plotting your input image with boundries for enhanced visualization")
        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(test_image[0], model.predict, top_labels=5, hide_color=0, num_samples=100)
        st.write("Processing the image for boundries...")
        
        progress = st.progress(0)
        progress_text = st.empty()
        
        for i in range(101):
            time.sleep(0.1)
            progress.progress(i)
            progress_text.text(f"Progress:{i}%")
            
        temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
        st.image(mark_boundaries(temp, mask),caption="Your crop with marked boundries")
        
        st.write("Thank you for using our app")


# In[ ]:




