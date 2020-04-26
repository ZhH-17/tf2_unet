from __future__ import print_function, division, absolute_import, unicode_literals

import os
import numpy as np
import logging
import time

import tensorflow as tf

from tensorflow.keras.layers import Conv2D, ReLU, Conv2DTranspose, MaxPool2D, Dropout, Dense
from tensorflow.keras import Model

from absl import app
from absl import flags

import pdb

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask

cmp = np.array([[0,0,0], [128,0,0], [0,128,0], [128,128,0], [0,0,128], [128,0,128], [0,128,128]])
def store_predictions(fn, model, dataset, offset):
    for images, gts in dataset.take(1):
        # images: n*h*w*C
        img = tf.cast(images*255, tf.int32)
        gt = tf.cast(gts, tf.int32)
        ny, nx = img.shape[1:3]
        if offset > 0:
            nx_out = nx - offset * 2
            ny_out = ny - offset * 2
            img = tf.image.crop_to_bounding_box(img, offset, offset, nx_out, ny_out)
            gt = tf.image.crop_to_bounding_box(gt, offset, offset, nx_out, ny_out)

        gt_rgb = tf.gather(params=cmp, indices=gt[..., 0])

        pred_mask = model.predict(images)
        pred_gt = create_mask(pred_mask)
        pred_gt_rgb = tf.gather(params=cmp, indices=pred_gt[..., 0])

        img_cat = tf.concat((img, gt_rgb, pred_gt_rgb), axis=2)
        img_cat = tf.concat([i for i in img_cat], axis=0)
    tf.keras.preprocessing.image.save_img(fn, img_cat)

def conv_twice(result, filters, size, dropout=0., padding='valid'):
    '''conv 2d twice
    result: tf.keras.Sequential()
    '''
    initializer1 = tf.keras.initializers.he_normal()
    initializer1_b = tf.random_normal_initializer(0.1, 0.02)
    result.add(Conv2D(filters, size, padding=padding,
        kernel_initializer=initializer1, bias_initializer=initializer1_b))
    if dropout > 1.e-3:
        result.add(Dropout(dropout))
    result.add(ReLU())

    initializer2 = tf.keras.initializers.he_normal()
    initializer2_b = tf.random_normal_initializer(0.1, 0.02)
    result.add(Conv2D(filters, size, padding=padding, 
        kernel_initializer=initializer2, bias_initializer=initializer2_b))
    result.add(ReLU())
    return result

def downsample(filters, size, dropout, pool=False, padding="same"):
    "down sample, pool firstly then conv"
    result = tf.keras.Sequential()
    if pool:
        result.add(MaxPool2D())
    conv_twice(result, filters, size, dropout, padding)
    return result

def upsample(filters, size, dropout, conv, padding='same'):
    "up sample, conv firstly then deconv"
    result = tf.keras.Sequential()
    if conv:
        conv_twice(result, filters, size, dropout, padding=padding)

    initializer = tf.keras.initializers.he_normal()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters // 2, 4, strides=2,
                                      padding='same',
                                      kernel_initializer=initializer,
                                      use_bias=False))

    return result

def outputlayer(filters, size, n_class, dropout, padding='same'):
    result = tf.keras.Sequential()
    conv_twice(result, filters, size, dropout, padding=padding)
    initializer = tf.keras.initializers.he_normal()
    initializer_b = tf.random_normal_initializer(0.1, 0.02)
    result.add(Conv2D(n_class, 1, kernel_initializer=initializer, bias_initializer=initializer_b,
        padding='same', activation='softmax'))
    return result

def unet_framework(dropout, n_class, channels=3, layers=3, features_root=16, padding='valid'):
    # generate unet model
    """
    self.dropout = dropout
    self.n_classs = n_class
    self.channels = channels
    self.layers = layers
    """
    size = 3
    features = features_root
    init_size = 128
    size_cur = init_size
    sizes_dw = []

    # down layers netweork
    dw_h_convs = []
    for layer in range(layers):
        dw_h_convs.append(downsample(features, size, dropout, pool=layer>0, padding=padding))
        if padding=='valid':
            size_cur = (size_cur//2 - 4) if layer>0 else (size_cur - 4)
            sizes_dw.append(size_cur)
        features *= 2

    features = features // 2
    # up layers netweork
    sizes_up = []
    up_h_convs = []
    for layer in range(layers-1):
        up_h_convs.append(upsample(features, size, dropout, conv=layer>0, padding=padding))
        features = features // 2
        if padding=='valid':
            size_cur = ((size_cur - 4)*2) if layer>0 else (size_cur*2)
            sizes_up.append(size_cur)
    
    # output layers netweork
    last_conv = outputlayer(features, size, n_class, dropout, padding)

    if padding=='valid':
        size_cur = size_cur - 4
    size_output = size_cur
    
    inputs = tf.keras.Input(shape=[None, None, channels])
    x = inputs

    skips = []
    for down in dw_h_convs:
        x = down(x)
        skips.append(x)
    skips = skips[::-1][1:]
    sizes_dw = sizes_dw[::-1][1:]
    # skips = reversed(skips[:-1])

    for i in range(len(skips)):
        up = up_h_convs[i]
        skip = skips[i]
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        if padding == 'valid':
            offset = (sizes_dw[i] - sizes_up[i]) // 2
            model_crop = tf.keras.Sequential()
            model_crop.add(tf.keras.layers.Cropping2D(cropping=((offset, offset), (offset, offset))))
            skip = model_crop(skip)
        x = concat([x, skip])
    x = last_conv(x)

    return tf.keras.Model(inputs=inputs, outputs=x), (init_size - size_cur)//2

class StoreCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch%5 == 0:
            fn = os.path.join(self.path, "epoch_%04d.jpg" %epoch)
            store_predictions(fn, self.model, self.dataset, self.offset)
            print ('\nSave Sample Prediction after epoch {}\n'.format(epoch+1))

    def set_dataset(self, dataset, offset, path):
        self.dataset = dataset
        self.offset = offset
        self.path = path

class loss_crop(tf.keras.losses.Loss):
    # loss for need of crop input
    def call(self, y_true, y_pred):
        if self.offset > 0:
            offset = self.offset
            nx_out = self.nx - offset * 2
            ny_out = self.ny - offset * 2
            y_true = tf.image.crop_to_bounding_box(y_true, offset, offset, nx_out, ny_out)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        return loss

    def set_offset(self, offset, nx, ny):
        self.offset = offset
        self.nx = nx
        self.ny = ny

class metric_crop(tf.keras.metrics.Metric):
    # accuracy for need of crop input
    def __init__(self, name='category_accuracy', **kwargs):
        super(metric_crop, self).__init__(name=name, **kwargs)
        self.acc = self.add_weight(name='tp', initializer='zeros')

    def set_offset(self, offset, nx, ny):
        self.offset = offset
        self.nx = nx
        self.ny = ny

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self.offset > 0:
            offset = self.offset
            nx_out = self.nx - offset * 2
            ny_out = self.ny - offset * 2
            y_true = tf.image.crop_to_bounding_box(y_true, offset, offset, nx_out, ny_out)

        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.argmax(y_pred, axis=-1)
        y_pred = y_pred[..., tf.newaxis]
        y_pred = tf.cast(y_pred, tf.int32)
        matched = tf.equal(y_true, y_pred)
        values = tf.cast(matched, tf.float32)
        self.acc.assign_add(tf.reduce_mean(values))

    def result(self):
        return self.acc


class unet():
    def __init__(self, dropout, n_class, shape=(None, None, 3), layers=3, features_root=16, padding='valid'):
        self.ny, self.nx, self.channels = shape[0:3]
        self.padding = padding
        self.model, self.offset = unet_framework(dropout, n_class, self.channels, layers, features_root, padding)
        self.loss_obj = loss_crop()
        self.loss_obj.set_offset(self.offset, self.nx, self.ny)

        self.metric_obj = metric_crop()
        self.metric_obj.set_offset(self.offset, self.nx, self.ny)

        self.model.compile(optimizer='adam', loss=self.loss_obj,
                metrics=[self.metric_obj])

    def train(self, checkpoint_prefix, train_dataset, test_dataset, epochs, steps_per_epoch, valid_steps):
        checkpoint_path = os.path.dirname(checkpoint_prefix)
        log_dir = os.path.join(checkpoint_path)

        store_callback = StoreCallback()
        store_callback.set_model(self.model)
        store_callback.set_dataset(test_dataset, self.offset, checkpoint_path)

        # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, profile_batch=100000)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, profile_batch=100000,
            histogram_freq=1, write_images=True)
        model_history = self.model.fit(train_dataset, epochs=epochs,
                                steps_per_epoch=steps_per_epoch,
                                validation_steps=valid_steps,
                                validation_data=test_dataset,
                                callbacks=[store_callback,
                                tf.keras.callbacks.ModelCheckpoint(checkpoint_prefix, save_best_only=True, save_weights_only=True, verbose=1),
                                tensorboard_callback]
                                )
        return model_history
