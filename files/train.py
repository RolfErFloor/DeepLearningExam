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

if __name__ == "__main__":
    data_path = config.PROJECT['data_path']
    (train_images, train_masks), (val_images, val_masks), (test_images, test_masks) = load_data(data_path)

    print("Train:", len(train_images), "Val:", len(val_images), "Test:", len(test_images))

    # Build list of hyperparameter combinations
    search_cfg = config.HYPERPARAM_SEARCH
    bs_choices = search_cfg.get('batch_size', [config.DEFAULTS['batch_size']])
    lr_choices = search_cfg.get('learning_rate', [config.DEFAULTS['learning_rate']])
    search_epochs = search_cfg.get('search_epochs', 10)

    best_val_loss = float('inf')
    best_params = {'batch_size': config.DEFAULTS['batch_size'], 'learning_rate': config.DEFAULTS['learning_rate']}
    search_results = []

    # Hyperparameter search (short runs)
    for bs, lr in itertools.product(bs_choices, lr_choices):
        print(f"\nStarting search trial: batch_size={bs}, learning_rate={lr}")
        train_dataset = dataset(train_images, train_masks, batch_size=bs)
        val_dataset   = dataset(val_images, val_masks, batch_size=bs)

        model = build_model()
        model.compile(optimizer=Adam(learning_rate=lr), loss=config.DEFAULTS['loss'], metrics=config.DEFAULTS['metrics'])

        steps_per_epoch = max(1, math.floor(len(train_images) / bs))
        validation_steps = max(1, math.floor(len(val_images) / bs))

        ckpt_name = config.PROJECT['search_checkpoint_template'].format(bs=bs, lr=lr)
        checkpoint = ModelCheckpoint(ckpt_name, monitor='val_loss', save_best_only=True, verbose=1)
        earlystop  = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
        reducelr   = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)

        history = model.fit(train_dataset,
                            epochs=search_epochs,
                            steps_per_epoch=steps_per_epoch,
                            validation_data=val_dataset,
                            validation_steps=validation_steps,
                            callbacks=[checkpoint, earlystop, reducelr],
                            verbose=2)

        trial_best = min(history.history.get('val_loss', [float('inf')]))
        print(f"Trial finished: val_loss={trial_best}")
        search_results.append({'batch_size': bs, 'learning_rate': lr, 'val_loss': trial_best})
        if trial_best < best_val_loss:
            best_val_loss = trial_best
            best_params = {'batch_size': bs, 'learning_rate': lr}

    print(f"\nBest hyperparameters found: {best_params}, val_loss={best_val_loss}")

    # Plot hyperparameter search results
    fig, ax = plt.subplots(figsize=(10, 6))
    for result in search_results:
        label = f"BS={result['batch_size']}, LR={result['learning_rate']:.0e}"
        ax.bar(label, result['val_loss'], alpha=0.7)
    ax.set_ylabel('Validation Loss')
    ax.set_title('Hyperparameter Search Results')
    ax.set_xlabel('Batch Size & Learning Rate')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('hyperparam_search_results.png', dpi=150)
    print("Saved plot: hyperparam_search_results.png")
    plt.close()

    # Continue training with best hyperparameters for full epochs
    final_bs = best_params['batch_size']
    final_lr = best_params['learning_rate']

    train_dataset = dataset(train_images, train_masks, batch_size=final_bs)
    val_dataset   = dataset(val_images, val_masks, batch_size=final_bs)

    model = build_model()
    model.compile(optimizer=Adam(learning_rate=final_lr), loss=config.DEFAULTS['loss'], metrics=config.DEFAULTS['metrics'])

    checkpoint = ModelCheckpoint(config.PROJECT['model_checkpoint'], monitor='val_loss', save_best_only=True, verbose=1)
    earlystop  = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
    reducelr   = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1)
    csvlogger  = CSVLogger(config.PROJECT['log_csv'], append=False)
    tensorboard= TensorBoard(log_dir='./logs')

    steps_per_epoch = max(1, math.floor(len(train_images) / final_bs))
    validation_steps = max(1, math.floor(len(val_images) / final_bs))

    final_history = model.fit(train_dataset,
                              epochs=config.DEFAULTS['full_epochs'],
                              steps_per_epoch=steps_per_epoch,
                              validation_data=val_dataset,
                              validation_steps=validation_steps,
                              callbacks=[checkpoint, earlystop, reducelr, csvlogger, tensorboard])

    # Plot final training history
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(final_history.history['loss'], label='Training Loss', linewidth=2)
    axes[0].plot(final_history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training vs Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(final_history.history['accuracy'], label='Training Accuracy', linewidth=2)
    axes[1].plot(final_history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training vs Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150)
    print("Saved plot: training_history.png")
    plt.close()
    
    print("Training complete!")