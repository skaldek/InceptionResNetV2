import os

import tensorflow as tf
from modules import *
import numpy as np
import matplotlib.pyplot as plt


ss = 299
BUFFER_SIZE = 100
BATCH_SIZE = 4

x_train = []
y_train = []
categories = os.listdir(path='Images')
print(len(categories))
num_categories = len(categories)
for category in categories:
    files = os.listdir(path='Images/' + category)
    for file in files:
        img = tf.keras.preprocessing.image.load_img('Images/%s/%s' % (category, file), target_size=(ss, ss))
        img = tf.keras.preprocessing.image.img_to_array(img)
        img /= 255
        x_train.append(img)
        y_train.append(str(category))

x_train = np.array(x_train, dtype=np.float32)
print(y_train)
y_train = tf.keras.utils.to_categorical(np.array(y_train))

print(x_train.shape)
print(y_train)

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

print(train_dataset)


def inception():
    inp = layers.Input(shape=(299, 299, 3))
    x = stem_block(inp)

    for _ in range(5):
        x = block_a(x)

    x = reduction_a(x)

    for _ in range(10):
        x = block_b(x)

    x = reduction_b(x)

    for _ in range(5):
        x = block_c(x)

    x = layers.GlobalAvgPool2D('channels_last')(x)
    x = layers.Dropout(0.8)(x)
    x = layers.Dense(num_categories, activation='softmax')(x)

    return tf.keras.Model(inp, x)


model = inception()
# tf.keras.utils.plot_model(model, show_shapes=True)

optimizer = tf.keras.optimizers.Adam(1e-4)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(model=model,
                                 optimizer=optimizer)

if os.path.exists('training_checkpoints'):
    checkpoint.restore(tf.train.latest_checkpoint('training_checkpoints'))

# print(model.summary())

loss_object = tf.keras.losses.BinaryCrossentropy()


print(np.argmax(model.predict(x_train[0].reshape(1, 299, 299, 3))))

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')


@tf.function
def train_step(imgs, lbls):
    with tf.GradientTape() as tape:
        predictions = model(imgs, training=True)
        loss = loss_object(lbls, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(lbls, predictions)


EPOCHS = 5


def train():
    for epoch in range(EPOCHS):
        train_loss.reset_states()
        train_accuracy.reset_states()

        for images, labels in train_dataset:
            train_step(images, labels)

        template = 'Epoch {}, Loss: {}, Accuracy: {}'
        print(template.format(epoch + 1,
                              train_loss.result(),
                              train_accuracy.result() * 100))


train()
