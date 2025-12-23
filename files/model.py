## U-NET MODEL FOR IMAGE SEGMENTATION
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, UpSampling2D, Input
from tensorflow.keras.models import Model
import config

#encoder
def conv_block(input_tensor, num_filters, l2_strength=1e-4):
    x = layers.Conv2D(num_filters, (3, 3), padding='same', 
                      kernel_regularizer=tf.keras.regularizers.l2(l2_strength))(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    x = layers.Conv2D(num_filters, (3, 3), padding='same',
                      kernel_regularizer=tf.keras.regularizers.l2(l2_strength))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    return x

def build_model(l2_strength=None):
    if l2_strength is None:
        l2_strength = config.REGULARIZATION.get('l2_strength', 1e-4)
    
    size = 256
    num_filters = [32, 64, 128, 256]
    inputs = Input((size, size, 3))

    skip_xs = []
    x = inputs

    for f in num_filters:
        x = conv_block(x, f, l2_strength)
        skip_xs.append(x)
        x = layers.MaxPooling2D((2, 2))(x)

    # bridge
    x = conv_block(x, num_filters[-1], l2_strength)
    num_filters.reverse()
    skip_xs.reverse()

    # decoder
    for i, f in enumerate(num_filters):
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Concatenate()([x, skip_xs[i]])
        x = conv_block(x, f, l2_strength)

    # output
    # output logits (no sigmoid)
    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation=None)(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

if __name__ == "__main__":
    model = build_model()
    model.summary()