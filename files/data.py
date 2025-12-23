import os
import glob
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import config

def load_data(path, val_size=100, test_size=100):
    
    script_dir = os.path.dirname(os.path.abspath(__file__))      # .../DeepLearningExam/files
    project_root = os.path.abspath(os.path.join(script_dir, ".."))  # .../DeepLearningExam
    dataset_dir = os.path.join(project_root, path)               # .../DeepLearningExam/CVC-ClinicDB

    image_paths = glob.glob(os.path.join(dataset_dir, "Original", "*.tif"))
    mask_paths  = glob.glob(os.path.join(dataset_dir, "Ground Truth", "*.tif"))

    image_paths.sort()
    mask_paths.sort()

    assert len(image_paths) == len(mask_paths), "Mismatch between images and masks"

    total_size = len(image_paths)
    
    # Ensure validation and test sizes are reasonable
    val_size = min(val_size, total_size // 3)
    test_size = min(test_size, total_size // 3)

    print("Total size:", total_size, "Valid size:", val_size, "Test size:", test_size)

    train_imgs, tmp_imgs, train_masks, tmp_masks = train_test_split(
        image_paths, mask_paths, test_size=(val_size + test_size), random_state=42
    )

    val_ratio = val_size / (val_size + test_size) if (val_size + test_size) else 0.5
    val_imgs, test_imgs, val_masks, test_masks = train_test_split(
        tmp_imgs, tmp_masks, test_size=(1 - val_ratio), random_state=42
    )

    return (train_imgs, train_masks), (val_imgs, val_masks), (test_imgs, test_masks)

def read_image(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x= cv2.resize(x, (256, 256))
    x = x / 255.0
    return x

def read_mask(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (256, 256))
    x = x / 255.0
    x = x.reshape((256, 256, 1))
    return x

def tf_parse(x, y):
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        return x, y
    x, y = tf.numpy_function(_parse, [x, y], [tf.float64, tf.float64])
    x.set_shape([256, 256, 3])
    y.set_shape([256, 256, 1])
    return x, y

def augment_data(image, mask):
    """Apply data augmentation to image and mask jointly."""
    aug_cfg = config.DATA_AUGMENTATION
    
    if not aug_cfg.get('enable', True):
        return image, mask
    
    # Random horizontal flip
    if aug_cfg.get('horizontal_flip', False):
        if tf.random.uniform([]) > 0.5:
            image = tf.image.flip_left_right(image)
            mask = tf.image.flip_left_right(mask)
    
    # Random vertical flip
    if aug_cfg.get('vertical_flip', False):
        if tf.random.uniform([]) > 0.5:
            image = tf.image.flip_up_down(image)
            mask = tf.image.flip_up_down(mask)
    
    # Random zoom (crop and resize)
    if aug_cfg.get('zoom_range', 0) > 0:
        zoom_factor = tf.random.uniform([], 1.0 - aug_cfg['zoom_range'], 1.0 + aug_cfg['zoom_range'])
        h, w = 256, 256
        new_h = tf.cast(tf.cast(h, tf.float32) / zoom_factor, tf.int32)
        new_w = tf.cast(tf.cast(w, tf.float32) / zoom_factor, tf.int32)
        new_h = tf.minimum(new_h, h)
        new_w = tf.minimum(new_w, w)
        
        # Crop to random position
        image = tf.image.random_crop(image, [new_h, new_w, 3])
        mask = tf.image.random_crop(mask, [new_h, new_w, 1])
        
        # Resize back to 256x256
        image = tf.image.resize(image, [h, w])
        mask = tf.image.resize(mask, [h, w])
    
    # Random brightness adjustment (only for image)
    if tf.random.uniform([]) > 0.5:
        image = tf.image.adjust_brightness(image, 0.1)
    
    return image, mask

def dataset(x, y, batch_size=8, augment=True):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.map(tf_parse)
    if augment and config.DATA_AUGMENTATION.get('enable', True):
        dataset = dataset.map(augment_data)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()
    return dataset


if __name__ == "__main__":
    data_path = "CVC-ClinicDB"
    (train_images, train_masks), (val_images, val_masks), (test_images, test_masks) = load_data(data_path)
    print (len(train_images), len(val_images), len(test_images))

    ds = dataset(train_images, train_masks)
    for x, y in ds:
        print (x.shape, y.shape)
        break