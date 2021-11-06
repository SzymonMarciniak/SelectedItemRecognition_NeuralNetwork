import os 

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
import tensorflow as tf
from tensorflow.keras.metrics import Precision, Recall

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


checkpoint_dir = "./training_checkpoints"

Positive = os.path.join("data", "positive")
Negative = os.path.join("data", "negative")
Anchor = os.path.join("data", "anchor")

anchor = tf.data.Dataset.list_files(Anchor+'/*.jpg').take(300)
positive = tf.data.Dataset.list_files(Positive+'/*.jpg').take(300)
negative = tf.data.Dataset.list_files(Negative+'/*.jpg').take(300)


def preprocessing(file):
    byte_img = tf.io.read_file(file)
    img = tf.io.decode_jpeg(byte_img)
    img = tf.image.resize(img, (100,100))
    img = img /255.0
    return img

positives = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
negatives = tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
data = positives.concatenate(negatives)


def second_preprocesig(input_image, validation_image, label):
    return(preprocessing(input_image), preprocessing(validation_image), label)

#pipeline
data = data.map(second_preprocesig)
data = data.cache()
data = data.shuffle(buffer_size=10000)

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
    
    input_ = Input(shape=(100,100,3), name="Input")

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


def siamese_model_function():

    input_embedding = Input(shape=(100,100,3), name="Input embadding")
    validation_embedding = Input(shape=(100,100,3), name="Validation embadding")

    siamese_layer = L1Distance()
    distance = siamese_layer(embedding(input_embedding), embedding(validation_embedding))

    classifer = Dense(1, activation="sigmoid")(distance)

    return Model(inputs=[input_embedding, validation_embedding], outputs=classifer, name="SiameseNetwork")


siamese_model = siamese_model_function()

binary_cross_loss_function = tf.losses.BinaryCrossentropy()
optimizer = tf.optimizers.Adam(1e-4) 

checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer, siamese_model=siamese_model)

@tf.function
def train_step(batch):

    with tf.GradientTape() as tape:

        X = batch[:2]
        Y_true = batch[2]

        Y_pred = siamese_model(X, training=True)
        loss = binary_cross_loss_function(Y_true, Y_pred)
    print(loss)
    
    gradient = tape.gradient(loss, siamese_model.trainable_variables)

    optimizer.apply_gradients(zip(gradient, siamese_model.trainable_variables))

    return loss 

def training(data, epochs):

    for epoch in range(1,epochs+1):
        print(f"\nEpoch {epoch}/{epochs}")
        progbar = tf.keras.utils.Progbar(len(data))

        r = Recall()
        p = Precision()

        for idx, batch in enumerate(data):
            loss = train_step(batch)
            y_pred = siamese_model.predict(batch[:2])
            r.update_state(batch[2], y_pred)
            p.update_state(batch[2], y_pred) 
            progbar.update(idx+1)
        print(loss.numpy(), r.result().numpy(), p.result().numpy())
        

        if epoch % 10 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

epochs = 50
#training(train_data, epochs)

test_input, test_val, y_true = test_data.as_numpy_iterator().next()
#y_pred = siamese_model.predict([test_input, test_val])

siamese_model.save("SiameseModel.h5")
model = tf.keras.models.load_model("SiameseModel.h5", custom_objects={"L1Distance":L1Distance, "BinaryCrossentropy": tf.losses.BinaryCrossentropy})
#print(model.summary())

y_pred = model.predict([test_input,  test_val])

r = Recall()
r.update_state(y_true, y_pred)
print(r.result().numpy())

p = Precision()
p.update_state(y_true, y_pred)
print(p.result().numpy())

print(y_true)
print(y_pred)




        