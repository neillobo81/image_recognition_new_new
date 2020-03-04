#!/usr/bin/python3
from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from json import dumps
import requests
import os
import os.path
from os import path
import matplotlib
import tensorflow as tf
import keras
from keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pylab as plt
import matplotlib.pyplot as plt
import h5py
import importlib



app = Flask(__name__)
api = Api(app)
batch_size = 128
epochs = 15
IMG_HEIGHT = 150
IMG_WIDTH = 150


class Picture(Resource):
    
    def post(self, url):
        r = requests.get(url, allow_redirects=True)
        
        if path.exists("/Users/neilmarlonlobo/Desktop/hackathon/python_rest_flask/image_to_recognize/image_file.jpg"):
            os.remove("/Users/neilmarlonlobo/Desktop/hackathon/python_rest_flask/image_to_recognize/image_file.jpg")
        
        open('image_to_recognize/image_file.jpg', 'wb').write(r.content)

        return ulcerRecognitionFunction


api.add_resource(Picture, '/recognize_pic/<url>')

def ulcerRecognitionFunction():
    model = tf.keras.models.load_model("my_model.h5")

    train_image_generator = ImageDataGenerator(rescale=1./255)
    prediction_data = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                            directory=predict_dir,
                                                            shuffle=True,
                                                            target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                            class_mode='binary')


    prediction_data.reset()
    prediction = model.predict(prediction_data)

    for a in prediction:
        if a[0] > 2:
            return "cat"
        else:
            return "foot ulcer"

if __name__ == '__main__':
    app.run(port='5555')
