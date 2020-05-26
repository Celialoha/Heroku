import streamlit as st
import pandas as pd
import numpy as np
import os, urllib
import time
from PIL import Image
#import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model

from keras.applications import VGG16
vgg = VGG16(weights='imagenet',
                 include_top=False)



cni = load_model('vgg16_92.h5')
 
def preprocess_image(img):
    img = Image.open(img)
    img = img.resize((224, 224), Image.ANTIALIAS)
    img_array = tf.keras.preprocessing.image.img_to_array(img, data_format=None, dtype=None)
    x = np.expand_dims(img_array, axis=0)
    x = x/255.
    return x

def cni_or_not(image,cni):        
    img_norm = preprocess_image(image)
    features = extract_features(img_norm)
    preds = cni.predict(features)
    return preds    

def extract_features(img_norm):
    features = np.zeros(shape=(7,7,512))
    features = vgg.predict(img_norm)
    return features

st.title('Authentification de CNI')


uploaded_file = st.file_uploader("Téléchargez votre carte d'identité", type=['png','jpg'])
'Veuillez patienter, nous étudions votre photo...'

if uploaded_file is not None:
    st.image(uploaded_file)
    resultat = cni_or_not(uploaded_file, cni)
    if resultat > 0.5:
        st.write("Ceci est une CNI à {:.2f} %".format(resultat [0,0]*100))
    else :
        st.write("Ceci n'est pas une CNI à {:.2f} %. Veuillez télécharger la bonne image".format(resultat [0,0]*100))

## Add a placeholder
latest_iteration = st.empty()
bar = st.progress(0)

for i in range(10):
  # Update the progress bar with each iteration.
  latest_iteration.text(f'Chargement {i+1}')
  bar.progress(i + 1)
  time.sleep(0.1)

'... Voici ce que nous en pensons !'


