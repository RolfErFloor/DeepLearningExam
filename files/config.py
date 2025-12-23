import tensorflow as tf

# helpers 
def _sigmoid_probs(y_pred, from_logits: bool):
    return tf.nn.sigmoid(y_pred) if from_logits else y_pred

# Dice (soft) 
def soft_dice_coef(y_true, y_pred, smooth=1e-6, from_logits=True):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(_sigmoid_probs(y_pred, from_logits), tf.float32)

    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])

    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    denom = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)
    return (2.0 * intersection + smooth) / (denom + smooth)

def soft_dice_loss(y_true, y_pred, from_logits=True):
    return 1.0 - soft_dice_coef(y_true, y_pred, from_logits=from_logits)

# Soft IoU 
def soft_iou_coef(y_true, y_pred, smooth=1e-6, from_logits=True):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(_sigmoid_probs(y_pred, from_logits), tf.float32)

    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])

    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

# Tversky index and loss
def tversky_coef(y_true, y_pred, alpha=0.3, beta=0.7, smooth=1e-6, from_logits=True):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(_sigmoid_probs(y_pred, from_logits), tf.float32)

    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])

    tp = tf.reduce_sum(y_true_f * y_pred_f)
    fp = tf.reduce_sum((1.0 - y_true_f) * y_pred_f)
    fn = tf.reduce_sum(y_true_f * (1.0 - y_pred_f))

    return (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)

def tversky_loss(y_true, y_pred, alpha=0.3, beta=0.7, from_logits=True):
    return 1.0 - tversky_coef(y_true, y_pred, alpha=alpha, beta=beta, from_logits=from_logits)

# BCE + Dice
class BCEDiceLoss(tf.keras.losses.Loss):
    """
    Stable BCE+Dice using logits.
    Assumes model outputs logits (no sigmoid in final layer).
    """
    def __init__(self, bce_weight=1.0, dice_weight=1.0, name="bce_dice_loss"):
        super().__init__(name=name)
        self.bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def call(self, y_true, y_pred):
        bce_val = self.bce(y_true, y_pred)
        dice_val = soft_dice_loss(y_true, y_pred, from_logits=True)
        return self.bce_weight * bce_val + self.dice_weight * dice_val

# Metrics as Keras Metric classes
class SoftDice(tf.keras.metrics.Metric):
    def __init__(self, name="soft_dice", from_logits=True, **kwargs):
        super().__init__(name=name, **kwargs)
        self.from_logits = from_logits
        self.total = self.add_weight(name="total", initializer="zeros", dtype=tf.float32)
        self.count = self.add_weight(name="count", initializer="zeros", dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        d = soft_dice_coef(y_true, y_pred, from_logits=self.from_logits)
        self.total.assign_add(d)
        self.count.assign_add(1.0)

    def result(self):
        return tf.math.divide_no_nan(self.total, self.count)

    def reset_states(self):
        self.total.assign(0.0)
        self.count.assign(0.0)

class SoftIoU(tf.keras.metrics.Metric):
    def __init__(self, name="soft_iou", from_logits=True, **kwargs):
        super().__init__(name=name, **kwargs)
        self.from_logits = from_logits
        self.total = self.add_weight(name="total", initializer="zeros", dtype=tf.float32)
        self.count = self.add_weight(name="count", initializer="zeros", dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        j = soft_iou_coef(y_true, y_pred, from_logits=self.from_logits)
        self.total.assign_add(j)
        self.count.assign_add(1.0)

    def result(self):
        return tf.math.divide_no_nan(self.total, self.count)

    def reset_states(self):
        self.total.assign(0.0)
        self.count.assign(0.0)

class ThresholdIoU(tf.keras.metrics.Metric):
  
    def __init__(self, name="iou", threshold=0.5, from_logits=True, **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.from_logits = from_logits
        self.intersection = self.add_weight(name="intersection", initializer="zeros", dtype=tf.float32)
        self.union = self.add_weight(name="union", initializer="zeros", dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_prob = tf.cast(_sigmoid_probs(y_pred, self.from_logits), tf.float32)
        y_pred_bin = tf.cast(y_prob >= self.threshold, tf.float32)

        intersection = tf.reduce_sum(y_true * y_pred_bin)
        union = tf.reduce_sum(y_true + y_pred_bin - y_true * y_pred_bin)

        self.intersection.assign_add(intersection)
        self.union.assign_add(union)

    def result(self):
        return tf.math.divide_no_nan(self.intersection, self.union)

    def reset_states(self):
        self.intersection.assign(0.0)
        self.union.assign(0.0)

# Loss instance
loss = BCEDiceLoss(bce_weight=1.0, dice_weight=1.0)

DEFAULTS = {
    'batch_size': 8,
    'learning_rate': 3e-4,   
    'optimizer': 'adam',
    'loss': loss,
    'metrics': [
        
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Recall(name='recall'),        
        tf.keras.metrics.Precision(name='precision'),  
        SoftDice(name='soft_dice', from_logits=True),
        SoftIoU(name='soft_iou', from_logits=True),
        ThresholdIoU(name='iou', threshold=0.5, from_logits=True),
    ],
    'full_epochs': 80,
}


# Hyperparameter grid used for quick search (each trial runs for `search_epochs`)
HYPERPARAM_SEARCH = {
    'batch_size': [4, 8, 16],
    'learning_rate': [1e-3, 5e-4, 1e-4],
    
    'search_epochs': 10,
}

# Regularization settings
REGULARIZATION = {
    'l2_strength': 1e-4,  # L2 regularization weight
    'dropout_rate': 0.3,  # Dropout rate 
}

# Data augmentation settings
DATA_AUGMENTATION = {
    'enable': True,
    'rotation_range': 20,      
    'horizontal_flip': True,
    'vertical_flip': True,
    'zoom_range': 0.2,         
    'shift_range': 0.1,        
    'elastic_deform': False,   # Set to True for advanced augmentation
}

PROJECT = {
    'data_path': 'CVC-ClinicDB',
    'model_checkpoint': 'unet_model_best.keras',
    'search_checkpoint_template': 'search_unet_bs{bs}_lr{lr:.0e}.keras',
    'log_csv': 'training_log.csv',
}
