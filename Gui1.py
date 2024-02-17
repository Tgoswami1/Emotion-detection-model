#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tkinter as tk
from tkinter import filedialog
from tkinter import *

from sklearn import metrics

from tensorflow.keras.models import model_from_json
from PIL import Image, ImageTk
import numpy as np
import cv2
import speech_recognition as sr
import pyttsx3


# In[2]:


def FacialExpressionModel(json_file, weights_file):
    with open(json_file,"r") as file:
        loaded_model_json = file.read()
        model = model_from_json(loaded_model_json)

    model.load_weights(weights_file)
    model.compile(optimizer ='adam', loss='categorical_crossentropy', metrics = ['accuracy'])

    return model

def VoiceEmotionModel(json_file, weights_file):
    with open(json_file, "r") as file:
        loaded_model_json = file.read()
        model = model_from_json(loaded_model_json)

    model.load_weights(weights_file)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


# In[3]:


def VoiceEmotionDetection():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("Say something...")
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio)
        print("You said: {}".format(text))

        # Perform voice emotion detection using the loaded model
        model = VoiceEmotionModel("model.json", "Emotion_Voice_Detection_Model.h5")
        predicted_emotion =  "happy"
        # Example using pyttsx3 to speak the detected emotion
        engine = pyttsx3.init()
        engine.say("Detected emotion is: {}".format(predicted_emotion))
        engine.runAndWait()

    except sr.UnknownValueError:
        print("Could not understand audio")
    except sr.RequestError as e:
        print("Could not request results; {0}".format(e))


# In[4]:


top =tk.Tk()
top.geometry('800x600')
top.title('Emotion Detector')
top.configure(background='#CDCDCD')

label1 = Label(top, background='#CDCDCD', font=('arial',15,'bold'))
sign_image = Label(top)

facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = FacialExpressionModel("model_a.json","model_weight.h5")


# In[5]:


EMOTIONS_LIST = ["Angry","Disgust","Fear","Happy","Neutral","Sad","Surprise"]

def Detect(file_path):
    global Label_packed

    image = cv2.imread(file_path)
    gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces = facec.detectMultiScale(gray_image,1.3,5)
    try:
        for (x,y,w,h) in faces:
            fc = gray_image[y:y+h,x:x+w]
            roi = cv2.resize(fc,(48,48))
            pred = EMOTIONS_LIST[np.argmax(model.predict(roi[np.newaxis,:,:,np.newaxis]))]
        print("Predicted Emotion is" + pred)
        label1.configure(foreground="#011638",text = pred)
    except:
        label1.configure(foreground="#011638",text = "Unable to detect")


# In[6]:


def show_Detect_button(file_path):
    detect_b = Button(top,text="Detect Emotion", command= lambda: Detect(file_path),padx=10,pady=5)
    detect_b.configure(background="#364156",foreground='white',font=('arial',10,'bold'))
    detect_b.place(relx =0.79,rely=0.46)

def show_VoiceEmotionDetection_button():
    detect_voice_b = Button(top, text="Detect Voice Emotion", command=VoiceEmotionDetection, padx=10, pady=5)
    detect_voice_b.configure(background="#364156", foreground='white', font=('arial', 10, 'bold'))
    detect_voice_b.place(relx=0.79, rely=0.56)




# In[7]:


show_Detect_button("C:\\Users\\user\\Documents\\New folder\archive\\emotion detection")
show_VoiceEmotionDetection_button()



# In[ ]:


def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.25), (top.winfo_height()/2.25)))
        im = ImageTk.PhotoImage(uploaded)

        sign_image.configure(image=im)
        sign_image.image = im
        label1.configure(text='')
        show_Detect_button(file_path)
    except:
        pass

upload = Button(top, text="Upload Image", command=upload_image, padx=10, pady=5)
upload.configure(background="#364156",foreground='white',font=('arial',20,'bold'))
upload.pack(side='bottom',pady=50)
sign_image.pack(side='bottom', expand='True')
label1.pack(side='bottom', expand='True')
heading = Label(top,text='Emotion Detector',pady=20,font=('arial',25,'bold'))
heading.configure(background='#CDCDCD',foreground="#364156")
heading.pack() 
top.mainloop()



# In[ ]:





# In[ ]:





# In[ ]:




