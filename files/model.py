## U-NET MODEL FOR IMAGE SEGMENTATION
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, UpSampling2D, Input
from tensorflow.keras.models import Model


def conv_block(input_tensor, num_filters):
    x = layers.Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    x = layers.Conv2D(num_filters, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    return x

def build_model():
    size = 256
    num_filters = [10,32, 48, 64, 128]
    inputs = Input((size, size, 3))

    skip_xs = []
    x= inputs

    for f in num_filters:
        x = conv_block(x, f)
        skip_xs.append(x)
        x = layers.MaxPooling2D((2, 2))(x)

    ##bridge
    x = conv_block(x, num_filters[-1])
    num_filters.reverse()
    skip_xs.reverse()


    ##decoder
    for i, f in enumerate(num_filters):
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Concatenate()([x, skip_xs[i]])
        x = conv_block(x, f)

    ##output
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

if __name__ == "__main__":
    model = build_model()
    model.summary()