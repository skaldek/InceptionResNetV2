from tensorflow.keras import layers


def conv2d(inp, filters, kernel, strides=1, padding='same', use_bias=False, activation='relu'):
    x = layers.Conv2D(filters, kernel, strides, padding=padding, use_bias=use_bias)(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    return x


def stem_block(inp):
    x = conv2d(inp, 32, 3, strides=2, padding='valid')
    x = conv2d(x, 32, 3, padding='valid')
    x = conv2d(x, 64, 3)

    x1 = layers.MaxPool2D(3, strides=2)(x)
    x2 = conv2d(x, 96, 3, strides=2, padding='valid')

    x = layers.Concatenate(3)([x1, x2])

    x1 = conv2d(x, 64, 1)
    x1 = conv2d(x1, 96, 3, padding='valid')

    x2 = conv2d(x, 64, 1)
    x2 = conv2d(x2, 64, (7, 1))
    x2 = conv2d(x2, 64, (1, 7))
    x2 = conv2d(x2, 96, 3, padding='valid')

    x = layers.Concatenate(3)([x1, x2])

    x1 = conv2d(x, 192, 3, strides=2, padding='valid')
    x2 = layers.MaxPool2D(3, strides=2)(x)

    x = layers.Concatenate(3)([x1, x2])

    return x


def block_a(inp):
    x1 = conv2d(inp, 32, 1)

    x2 = conv2d(inp, 32, 1)
    x2 = conv2d(x2, 32, 3)

    x3 = conv2d(inp, 32, 1)
    x3 = conv2d(x3, 48, 3)
    x3 = conv2d(x3, 64, 3)

    x = layers.Concatenate(3)([x1, x2, x3])
    x = conv2d(x, 384, 1, activation='linear')

    x = layers.Add()([inp, x])

    return x


def reduction_a(inp):
    x1 = layers.MaxPool2D(3, 2)(inp)

    x2 = conv2d(inp, 384, 3, 2, 'valid')

    x3 = conv2d(inp, 256, 1)
    x3 = conv2d(x3, 256, 3)
    x3 = conv2d(x3, 384, 3, 2, 'valid')

    x = layers.Concatenate(3)([x1, x2, x3])

    return x


def block_b(inp):
    x1 = conv2d(inp, 192, 1)

    x2 = conv2d(inp, 128, 1)
    x2 = conv2d(x2, 160, (1, 7))
    x2 = conv2d(x2, 192, (7, 1))

    x = layers.Concatenate(3)([x1, x2])
    x = conv2d(x, 1152, 1, activation='linear')

    x = layers.Add()([inp, x])

    return x


def reduction_b(inp):
    x1 = layers.MaxPool2D(3, 2)(inp)

    x2 = conv2d(inp, 256, 1)
    x2 = conv2d(x2, 384, 3, 2, 'valid')

    x3 = conv2d(inp, 256, 1)
    x3 = conv2d(x3, 256, 3, 2, 'valid')

    x4 = conv2d(inp, 256, 1)
    x4 = conv2d(x4, 256, 3)
    x4 = conv2d(x4, 256, 3, 2, 'valid')

    x = layers.Concatenate(3)([x1, x2, x3, x4])

    return x


def block_c(inp):
    x1 = conv2d(inp, 192, 1)

    x2 = conv2d(inp, 192, 1)
    x2 = conv2d(x2, 224, (1, 3))
    x2 = conv2d(x2, 256, (3, 1))

    x = layers.Concatenate(3)([x1, x2])
    x = conv2d(x, 2048, 1, activation='linear')

    x = layers.Add()([inp, x])

    return x
