##TRAIN
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import cv2
import glob
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger, TensorBoard
from data import load_data, dataset
from model import build_model
import itertools
import math
import config
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import time

if __name__ == "__main__":
    data_path = config.PROJECT['data_path']
    (train_images, train_masks), (val_images, val_masks), (test_images, test_masks) = load_data(data_path)

    print("Train:", len(train_images), "Val:", len(val_images), "Test:", len(test_images))

    # Build list of hyperparameter combinations
    search_cfg = config.HYPERPARAM_SEARCH
    bs_choices = search_cfg.get('batch_size', [config.DEFAULTS['batch_size']])
    lr_choices = search_cfg.get('learning_rate', [config.DEFAULTS['learning_rate']])
    search_epochs = search_cfg.get('search_epochs', 10)

    best_val_score = float('-inf')
    best_params = {'batch_size': config.DEFAULTS['batch_size'], 'learning_rate': config.DEFAULTS['learning_rate']}
    search_results = []

    monitor_metric = 'val_soft_iou' 

    # Hyperparameter search 
    total_trials = len(bs_choices) * len(lr_choices)
    trial_idx = 0
    for bs, lr in itertools.product(bs_choices, lr_choices):
        trial_idx += 1
        print(f"\nStarting search trial {trial_idx}/{total_trials}: batch_size={bs}, learning_rate={lr}")
        start_time = time.time()
        train_dataset = dataset(train_images, train_masks, batch_size=bs, augment=True)
        val_dataset   = dataset(val_images, val_masks, batch_size=bs, augment=False)

        model = build_model()
        model.compile(
                optimizer=Adam(learning_rate=lr, clipnorm=1.0),
                loss=config.DEFAULTS['loss'],
                metrics=config.DEFAULTS['metrics']
            )


        steps_per_epoch = max(1, math.ceil(len(train_images) / bs))
        validation_steps = max(1, math.ceil(len(val_images) / bs))

        ckpt_name = config.PROJECT['search_checkpoint_template'].format(bs=bs, lr=lr)
        checkpoint = ModelCheckpoint(ckpt_name, monitor=monitor_metric, save_best_only=True, verbose=1, mode='max')
        earlystop  = EarlyStopping(monitor=monitor_metric, patience=5, verbose=1, mode='max')
        reducelr   = ReduceLROnPlateau(monitor=monitor_metric, factor=0.1, patience=3, verbose=1, mode='max')

        history = model.fit(train_dataset,
                            epochs=search_epochs,
                            steps_per_epoch=steps_per_epoch,
                            validation_data=val_dataset,
                            validation_steps=validation_steps,
                            callbacks=[checkpoint, earlystop, reducelr],
                            verbose=2)

        trial_score = max(history.history.get('val_accuracy', [0.0]))
        duration = time.time() - start_time
        percent = (trial_idx / total_trials) * 100.0
        print(f"Trial finished: val_accuracy={trial_score:.4f} (duration {duration:.1f}s). Progress: {trial_idx}/{total_trials} ({percent:.1f}%).")
        search_results.append({'batch_size': bs, 'learning_rate': lr, 'val_accuracy': trial_score})
        if trial_score > best_val_score:
            best_val_score = trial_score
            best_params = {'batch_size': bs, 'learning_rate': lr}
            print(f"New best params: {best_params} with val_accuracy={best_val_score:.4f}")

    print(f"\nBest hyperparameters found: {best_params}, val_accuracy={best_val_score}")


    # Continue training with best hyperparameters for full epochs
    final_bs = best_params['batch_size']
    final_lr = best_params['learning_rate']

    train_dataset = dataset(train_images, train_masks, batch_size=final_bs, augment=True)
    val_dataset   = dataset(val_images, val_masks, batch_size=final_bs, augment=False)

    model = build_model()
    model.compile(optimizer=Adam(learning_rate=final_lr), loss=config.DEFAULTS['loss'], metrics=config.DEFAULTS['metrics'])

    checkpoint = ModelCheckpoint(config.PROJECT['model_checkpoint'], monitor=monitor_metric, save_best_only=True, verbose=1, mode='max')
    earlystop  = EarlyStopping(monitor=monitor_metric, patience=10, verbose=1, mode='max')
    reducelr   = ReduceLROnPlateau(monitor=monitor_metric, factor=0.1, patience=5, verbose=1, mode='max')
    csvlogger  = CSVLogger(config.PROJECT['log_csv'], append=False)
    tensorboard= TensorBoard(log_dir='./logs')

    steps_per_epoch = max(1, math.ceil(len(train_images) / final_bs))
    validation_steps = max(1, math.ceil(len(val_images) / final_bs))


    final_history = model.fit(train_dataset,
                              epochs=config.DEFAULTS['full_epochs'],
                              steps_per_epoch=steps_per_epoch,
                              validation_data=val_dataset,
                              validation_steps=validation_steps,
                              callbacks=[checkpoint, earlystop, reducelr, csvlogger, tensorboard])

    # Plot final training history for requested metrics

    metrics_to_plot = ['loss', 'accuracy', 'recall', 'precision', 'iou']
    n = len(metrics_to_plot)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 8))

    # Top row: training metrics
    for i, m in enumerate(metrics_to_plot):
        ax = axes[0, i]
        train_key = m
        train_vals = final_history.history.get(train_key, None)

        if train_vals is not None:
            ax.plot(train_vals, linewidth=2.5, color='#1f77b4')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel(m.capitalize())
        ax.set_title(f'Training {m.capitalize()}')
        ax.grid(True, alpha=0.3)

    # Bottom row: validation metrics
    for i, m in enumerate(metrics_to_plot):
        ax = axes[1, i]
        val_key = 'val_' + m
        val_vals = final_history.history.get(val_key, None)

        if val_vals is not None:
            ax.plot(val_vals, linewidth=2.5, color='#ff7f0e')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel(m.capitalize())
        ax.set_title(f'Validation {m.capitalize()}')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150)
    print("Saved plot: training_history.png")
    plt.close()
    
    print("Training complete!")