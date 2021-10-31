import os 
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

Positive = os.path.join("data", "posotive")
Negative = os.path.join("data", "negative")
Anchor = os.path.join("data", "anchor")

os.makedirs(Positive)
os.makedirs(Negative)
os.makedirs(Anchor)
