import os
import glob
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split

def load_data(path, split=0.1):
    
    script_dir = os.path.dirname(os.path.abspath(__file__))      # .../DeepLearningExam/files
    project_root = os.path.abspath(os.path.join(script_dir, ".."))  # .../DeepLearningExam
    dataset_dir = os.path.join(project_root, path)               # .../DeepLearningExam/CVC-ClinicDB

    image_paths = glob.glob(os.path.join(dataset_dir, "Original", "*.tif"))
    mask_paths  = glob.glob(os.path.join(dataset_dir, "Ground Truth", "*.tif"))

    image_paths.sort()
    mask_paths.sort()

    assert len(image_paths) == len(mask_paths), "Mismatch between images and masks"

    total_size = len(image_paths)
    valid_size = int(total_size * split)
    test_size  = int(total_size * split)

    print("Total size:", total_size, "Valid size:", valid_size, "Test size:", test_size)

    train_imgs, tmp_imgs, train_masks, tmp_masks = train_test_split(
        image_paths, mask_paths, test_size=(valid_size + test_size), random_state=42
    )

    val_ratio = valid_size / (valid_size + test_size) if (valid_size + test_size) else 0.5
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

def dataset(x, y, batch_size=8):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.map(tf_parse)
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