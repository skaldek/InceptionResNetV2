import os

import tensorflow as tf
import numpy as np
import hyperparams as hp
from Inception import inception


image_path = 'Images/0/1038523450_7f333662ef.jpg'  # Image to class prediction

model = inception(hp.num_categories)

checkpoint_prefix = os.path.join(hp.checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(model=model)

if os.path.exists(hp.checkpoint_dir):
    checkpoint.restore(tf.train.latest_checkpoint(hp.checkpoint_dir))

    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(hp.ss, hp.ss))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img /= 255

    print(np.argmax(model.predict(img.reshape(1, hp.ss, hp.ss, 3))))

else:
    print('No checkpoints!')
