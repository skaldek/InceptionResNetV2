import os

from Inception import inception
import tensorflow as tf
import numpy as np
import hyperparams as hp

model = inception(hp.num_categories)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(model=model)

if os.path.exists('training_checkpoints'):
    checkpoint.restore(tf.train.latest_checkpoint('training_checkpoints'))

    img = tf.keras.preprocessing.image.load_img('Images/1/0.jpg', target_size=(hp.ss, hp.ss))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img /= 255

    print(model.predict(img.reshape(1, hp.ss, hp.ss, 3)))

else:
    print('No checkpoints!')
