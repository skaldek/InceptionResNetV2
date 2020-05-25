import math
import os

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from Inception import inception
import hyperparams as hp

BUFFER_SIZE = 100

x_train = []
y_train = []
categories = os.listdir(path='Images')[:10]
print(len(categories))
num_categories = len(categories)
for idx, category in enumerate(categories):
    files = os.listdir(path='Images/' + category)
    for file in files:
        img = tf.keras.preprocessing.image.load_img('Images/%s/%s' % (category, file), target_size=(hp.ss, hp.ss))
        img = tf.keras.preprocessing.image.img_to_array(img)
        img /= 255
        x_train.append(img)
        y_train.append(idx)

x_train = np.array(x_train, dtype=np.float32)
print(y_train)
y_train = tf.keras.utils.to_categorical(np.array(y_train))

print(x_train.shape)
print(y_train)

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(BUFFER_SIZE).batch(hp.batch_size)

print(train_dataset)

model = inception(num_categories)
tf.keras.utils.plot_model(model, show_shapes=True)

optimizer = tf.keras.optimizers.Adam(1e-4)


checkpoint_prefix = os.path.join(hp.checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(model=model,
                                 optimizer=optimizer)

if os.path.exists(hp.checkpoint_dir):
    checkpoint.restore(tf.train.latest_checkpoint(hp.checkpoint_dir))

# print(model.summary())

loss_object = tf.keras.losses.CategoricalCrossentropy()

print('Prediction: ', np.argmax(model.predict(x_train[0].reshape(1, 299, 299, 3))))

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')


@tf.function
def train_step(imgs, lbls):
    with tf.GradientTape() as tape:
        predictions = model(imgs, training=True)
        loss = loss_object(lbls, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(lbls, predictions)


def train():
    def ceil(val):
        return math.ceil(val * 100) / 100

    for epoch in range(hp.epochs):
        train_loss.reset_states()
        train_accuracy.reset_states()

        for images, labels in tqdm(train_dataset, total=len(list(train_dataset))):
            train_step(images, labels)

        if (epoch + 1) % 1 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        template = 'Epoch {}, Loss: {}, Accuracy: {}'
        print(template.format(epoch + 1,
                              ceil(train_loss.result()),
                              ceil(train_accuracy.result() * 100)))


train()
