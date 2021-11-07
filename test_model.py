

import os 

from tensorflow.keras.layers import Layer
import tensorflow as tf
from tensorflow.keras.metrics import Precision, Recall



class L1Distance(Layer):
    def __init__(self, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable=trainable, name=name, dtype=dtype, dynamic=dynamic, **kwargs)
    
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)

checkpoint_dir = "./training_checkpoints"

Positive = os.path.join("data", "positive")
Negative = os.path.join("data", "negative")
Anchor = os.path.join("data", "anchor")

anchor = tf.data.Dataset.list_files(Anchor+'/*.jpg').take(300)
positive = tf.data.Dataset.list_files(Positive+'/*.jpg').take(300)
negative = tf.data.Dataset.list_files(Negative+'/*.jpg').take(300)


model = tf.keras.models.load_model("SiameseModel.h5", custom_objects={"L1Distance":L1Distance, "BinaryCrossentropy": tf.losses.BinaryCrossentropy})

positives = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
negatives = tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
data = positives.concatenate(negatives)


def preprocessing(file):
    byte_img = tf.io.read_file(file)
    img = tf.io.decode_jpeg(byte_img)
    img = tf.image.resize(img, (105,105))
    img = img /255.0
    return img


def second_preprocesig(input_image, validation_image, label):
    return(preprocessing(input_image), preprocessing(validation_image), label)


data = data.map(second_preprocesig)
data = data.cache()
data = data.shuffle(buffer_size=10000)

test_data = data.skip(round(len(data)*.7))
test_data = test_data.take(round(len(data)*.3))
test_data = test_data.batch(16)
test_data = test_data.prefetch(8)

test_input, test_val, y_true = test_data.as_numpy_iterator().next()

y_pred = model.predict([test_input,  test_val])


r = Recall()
r.update_state(y_true, y_pred)
print(r.result().numpy())

p = Precision()
p.update_state(y_true, y_pred)
print(p.result().numpy())

print(y_true)
print(y_pred)

