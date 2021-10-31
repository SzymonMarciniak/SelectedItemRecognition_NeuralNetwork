from operator import pos
import cv2
import os 
import random 
import numpy as np 
from matplotlib import pyplot as plt

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
import tensorflow as tf

Positive = os.path.join("data", "positive")
Negative = os.path.join("data", "negative")
Anchor = os.path.join("data", "anchor")

anchor = tf.data.Dataset.list_files(Anchor+'/*.jpg').take(300)
positive = tf.data.Dataset.list_files(Positive+'/*.jpg').take(300)
negative = tf.data.Dataset.list_files(Negative+'/*.jpg').take(300)


def preprocessing(file):
    byte_img = tf.io.read_file(file)
    img = tf.io.decode_jpeg(byte_img)
    img = tf.image.resize(img, (105,105))
    img = img /255.0
    return img

positives = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
negatives = tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
data = positives.concatenate(negatives)


def second_preprocesig(input_image, validation_image, label):
    return(preprocessing(input_image), preprocessing(validation_image), label)

#pipeline
data = data.map(second_preprocesig)
data = data.cache()
data = data.shuffle(buffer_size=1024)

#train data
train_data = data.take(round(len(data)*.7))
train_data = train_data.batch(16)
train_data = train_data.prefetch(8)

#test data
test_data = data.skip(round(len(data)*.7))
test_data = test_data.take(round(len(data)*.3))
test_data = test_data.batch(16)
test_data = test_data.prefetch(8)

