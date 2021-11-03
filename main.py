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

def embedding_function():
    
    input_ = Input(shape=(105,105,3), name="Input")

    c1 = Conv2D(64, (10,10), activation="relu")(input_)
    m1 = MaxPooling2D(64, (2,2), padding="same")(c1)

    c2 = Conv2D(128, (7,7), activation="relu")(m1)
    m2 = MaxPooling2D(64, (2,2), padding="same")(c2)

    c3 = Conv2D(128, (4,4), activation="relu")(m2)
    m3 = MaxPooling2D(64, (2,2), padding="same")(c3)

    c4 = Conv2D(256, (4,4), activation="relu")(m3)
    f1 = Flatten()(c4)
    d1 = Dense(4096, activation="sigmoid")(f1)

    return Model(inputs=[input_], outputs=[d1], name="embedding")

embedding = embedding_function()

class L1Distance(Layer):
    def __init__(self, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable=trainable, name=name, dtype=dtype, dynamic=dynamic, **kwargs)
    
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)


def siamese_model():

    input_embedding = Input(shape=(105,105,3), name="Input embadding")
    validation_embedding = Input(shape=(105,105,3), name="Validation embadding")

    siamese_layer = L1Distance()
    distance = siamese_layer(embedding(input_embedding), embedding(validation_embedding))

    classifer = Dense(1, activation="sigmoid")(distance)

    return Model(inputs=[input_embedding, validation_embedding], outputs=classifer, name="SiameseNetwork")

s_siamese_model = siamese_model()
x = s_siamese_model.summary()
print(x)