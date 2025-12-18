# Configuration and hyperparameter search settings for Polyp Segmentation
import tensorflow as tf


# Losses: Dice and combined BCE + Dice
def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coefficient(y_true, y_pred)


def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    return bce + dice_loss(y_true, y_pred)


# Defaults used for full training
DEFAULTS = {
    'batch_size': 8,
    'learning_rate': 1e-3,
    'optimizer': 'adam',
    'loss': bce_dice_loss,
    'metrics': ['accuracy'],
    'full_epochs': 50,
}

# Hyperparameter grid used for quick search (each trial runs for `search_epochs`)
HYPERPARAM_SEARCH = {
    'batch_size': [4, 8, 16],
    'learning_rate': [1e-3, 5e-4, 1e-4],
    # you can add other hyperparams here (e.g. optimizer, augmentation flags)
    'search_epochs': 10,
}

PROJECT = {
    'data_path': 'CVC-ClinicDB',
    'model_checkpoint': 'unet_model_best.h5',
    'search_checkpoint_template': 'search_unet_bs{bs}_lr{lr:.0e}.h5',
    'log_csv': 'training_log.csv',
}
