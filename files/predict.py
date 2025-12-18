## predict.py
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf

import config
from model import build_model


def read_image_any(path, size=(256, 256)):
    """
    Works with both:
      - bytes (from tf.numpy_function)
      - str (normal Python usage, e.g., prediction script)
    Returns float32 image in [0,1] with shape (H,W,3).
    """
    if isinstance(path, bytes):
        path = path.decode()

    x = cv2.imread(path, cv2.IMREAD_COLOR)  # BGR
    if x is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    x = cv2.resize(x, size, interpolation=cv2.INTER_AREA)
    x = x.astype("float32") / 255.0
    return x


if __name__ == "__main__":
    data_path = config.PROJECT["data_path"]
    model_checkpoint = config.PROJECT["model_checkpoint"]

    # Resolve dataset path from project root (one level above /files)
    script_dir = os.path.dirname(os.path.abspath(__file__))           # .../DeepLearningExam/files
    project_root = os.path.abspath(os.path.join(script_dir, ".."))    # .../DeepLearningExam
    dataset_dir = os.path.join(project_root, data_path)               # .../DeepLearningExam/CVC-ClinicDB

    test_image_paths = sorted(glob.glob(os.path.join(dataset_dir, "Original", "*.tif")))
    if len(test_image_paths) == 0:
        raise ValueError(f"No .tif files found in: {os.path.join(dataset_dir, 'Original')}")

    # Build and load model
    model = build_model()
    model.load_weights(model_checkpoint)

    # Output dir
    output_dir = os.path.join(project_root, "Predictions")
    os.makedirs(output_dir, exist_ok=True)

    # Predict and save masks for all images
    for img_path in test_image_paths:
        img = read_image_any(img_path, size=(256, 256))
        img_input = np.expand_dims(img, axis=0)  # (1,256,256,3)

        pred_mask = model.predict(img_input, verbose=0)[0]  # (256,256,1)
        pred_mask_binary = (pred_mask > 0.5).astype(np.uint8) * 255  # (256,256,1)

        base_name = os.path.basename(img_path)
        save_path = os.path.join(output_dir, base_name)

        # cv2.imwrite expects 2D for grayscale; squeeze channel
        cv2.imwrite(save_path, pred_mask_binary.squeeze())

        print(f"Saved prediction to {save_path}")

    # Save visualization of first 5 predictions
    n_show = min(5, len(test_image_paths))
    fig, axes = plt.subplots(n_show, 3, figsize=(12, 3 * n_show))

    # If only 1 row, axes is 1D; normalize to 2D indexing
    if n_show == 1:
        axes = np.expand_dims(axes, axis=0)

    for idx, img_path in enumerate(test_image_paths[:n_show]):
        img = read_image_any(img_path, size=(256, 256))
        img_input = np.expand_dims(img, axis=0)

        pred_mask = model.predict(img_input, verbose=0)[0]  # (256,256,1)
        pred_mask_binary = (pred_mask > 0.5).astype(np.uint8) * 255  # (256,256,1)

        # OpenCV loads BGR; matplotlib expects RGB. Convert for display.
        img_uint8_bgr = np.clip(img * 255, 0, 255).astype(np.uint8)
        img_uint8_rgb = cv2.cvtColor(img_uint8_bgr, cv2.COLOR_BGR2RGB)

        axes[idx, 0].imshow(img_uint8_rgb)
        axes[idx, 0].set_title(f"Original ({idx + 1})")
        axes[idx, 0].axis("off")

        axes[idx, 1].imshow(pred_mask[:, :, 0], cmap="gray")
        axes[idx, 1].set_title(f"Predicted Mask ({idx + 1})")
        axes[idx, 1].axis("off")

        axes[idx, 2].imshow(img_uint8_rgb)
        axes[idx, 2].imshow(pred_mask_binary.squeeze(), cmap="Reds", alpha=0.5)
        axes[idx, 2].set_title(f"Overlay ({idx + 1})")
        axes[idx, 2].axis("off")

    plt.tight_layout()
    viz_path = os.path.join(project_root, "predictions_visualization.png")
    plt.savefig(viz_path, dpi=150)
    print(f"\nSaved visualization: {viz_path}")
