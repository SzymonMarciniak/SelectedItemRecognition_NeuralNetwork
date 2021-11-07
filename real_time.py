import os 

import cv2
import numpy as np
from tensorflow.keras.layers import Layer
import tensorflow as tf

class L1Distance(Layer):
    def __init__(self, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable=trainable, name=name, dtype=dtype, dynamic=dynamic, **kwargs)
    
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)


def preprocessing(file):
    byte_img = tf.io.read_file(file)
    img = tf.io.decode_jpeg(byte_img)
    img = tf.image.resize(img, (105,105))
    img = img /255.0
    return img


def verify(model, detection_threshold, verification_threshold):

    results = []

    for img in os.listdir(os.path.join("app_data", "verify_img")):

        input_img = preprocessing(os.path.join("app_data", "input_img", "input_image.jpg"))
        validation_img = preprocessing(os.path.join("app_data", "verify_img", img))

        result = model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
        results.append(result)
    
    detection = np.sum(np.array(results) > detection_threshold)

    verification = detection / len(os.listdir(os.path.join("app_data", "verify_img")))
    print(verification)
    verified = verification > verification_threshold

    return results, verified


siamese_model = tf.keras.models.load_model("SiameseModel.h5", custom_objects={"L1Distance":L1Distance, "BinaryCrossentropy": tf.losses.BinaryCrossentropy})

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    frame = frame[100:100+250,250:250+250, :]
    
    cv2.imshow('Verify', frame)
 
    if cv2.waitKey(10) & 0xFF == ord('e'):

        cv2.imwrite(os.path.join('app_data', 'input_img', 'input_image.jpg'), frame)
       
        results, verified = verify(siamese_model, 0.5, 0.5)
        print(verified)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()