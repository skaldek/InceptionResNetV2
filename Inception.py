from modules import *


def inception(num_categories):
    inp = layers.Input(shape=(hp.ss, hp.ss, 3))
    x = stem_block(inp)

    for _ in range(5):
        x = block_a(x)

    x = reduction_a(x)

    for _ in range(10):
        x = block_b(x)

    x = reduction_b(x)

    for _ in range(5):
        x = block_c(x)

    x = layers.GlobalAveragePooling2D('channels_last')(x)
    x = layers.Dropout(hp.dropout)(x)
    x = layers.Dense(num_categories, activation='softmax')(x)

    return tf.keras.Model(inp, x)
