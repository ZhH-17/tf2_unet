from __future__ import print_function, division, absolute_import, unicode_literals

import os
import shutil
import numpy as np
import cv2 as cv
from collections import OrderedDict
import logging
import time
import matplotlib.pyplot as plt

import tensorflow as tf
# tf.enable_eager_excution()

# from tf_unet import image_gen

from tensorflow.keras.layers import Conv2D, ReLU, Conv2DTranspose, MaxPool2D, Dropout
from tensorflow.keras import Model

from absl import app
from absl import flags
from unet_model import unet, create_mask
from unet_predict import stack_imgs

import pdb

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

def gen_dataset(generator, num):
    imgs, masks = generator(30)
    imgs = imgs.astype(np.float32)
    masks = masks.astype(np.float32)
    gts = np.argmax(masks, axis=-1).astype(np.float32)
    gts = np.expand_dims(gts, axis=-1)
    img_ds = tf.data.Dataset.from_tensor_slices(imgs)
    gt_ds = tf.data.Dataset.from_tensor_slices(gts)
    dataset = tf.data.Dataset.zip((img_ds, gt_ds))
    return dataset

def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.savefig("./training_checkpoints/%s.jpg" %time.time())

class gen_from_image(object):
    def __init__(self, img_path, gt_path, nx, ny, offset, buffer_size, batch_size):
        self.img = cv.imread(img_path, -1)
        # normal
        self.img = tf.cast(self.img, tf.float32)/255.0
        # self.img = tf.cast(self.img, tf.float32)

        self.img_gt = cv.imread(gt_path, -1)
        self.n_class = self.img_gt.max() + 1
        self.img_gt = tf.cast(np.atleast_3d(self.img_gt), tf.float32)
        h, w = self.img.shape[:2]
        # self.img_gt = np.zeros((h, w, 1))
        images = []
        gts = []
        self.num_row = int(np.ceil(float(h-offset)/(ny-offset)))
        self.num_col = int(np.ceil(float(w-offset)/(nx-offset)))
        for ih in range(0, h-offset, ny-offset):
            for iw in range(0, w-offset, nx-offset):
                img_tmp = self.img[ih:ih+ny, iw:iw+nx]
                gt_tmp = self.img_gt[ih:ih+ny, iw:iw+nx]
                if img_tmp.shape[0]!=ny or img_tmp.shape[1]!=nx:
                    pad_x = nx - img_tmp.shape[1]
                    pad_y = ny - img_tmp.shape[0]
                    paddings = tf.constant([[0, pad_y], [0, pad_x], [0,0]])
                    img_tmp = tf.pad(img_tmp, paddings, 'constant')
                    gt_tmp = tf.pad(gt_tmp, paddings, 'constant')
                    # img_tmp = np.pad(img_tmp, ((0, pad_y), (0, pad_x), (0, 0)), 'constant')
                    # gt_tmp = np.pad(gt_tmp, ((0, pad_y), (0, pad_x), (0, 0)), 'constant')
                images.append(img_tmp)
                gts.append(gt_tmp)
        self.images = images
        self.gts = gts

        size_test = 6
        size_test = len(self.images) // 10
        inds = np.random.randint(0, len(gts), size_test)
        ds_img_test = tf.data.Dataset.from_tensor_slices([images[i] for i in inds])
        ds_gt_test = tf.data.Dataset.from_tensor_slices([gts[i] for i in inds])
        self.test_dataset = tf.data.Dataset.zip((ds_img_test, ds_gt_test))
        self.test_dataset = self.test_dataset.batch(size_test)

        ds_img = tf.data.Dataset.from_tensor_slices(images)
        ds_gt = tf.data.Dataset.from_tensor_slices(gts)

        self.dataset = tf.data.Dataset.zip((ds_img, ds_gt))
        self.dataset = self.dataset.shuffle(buffer_size).batch(batch_size).repeat()
        self.dataset = self.dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

if __name__ == "__main__":
    batch_size = 3
    buffer_size = 30
    dropout = 0
    epochs = 2000
    valid_steps = 1

    nx = 572
    ny = 572
    # nx = 1000
    # ny = 1000

    n_class = 7
    weights = np.array([1., 2., 1.2, 1.3, 1, 20., 6])
    net = unet(dropout, n_class, weights=weights,
               shape=(ny, nx, 3), layers=3, padding='valid')
    model = net.model

    gen_ds = gen_from_image("./data/cj_right_all.png", "./data/cj_right_all_gt.png", nx, ny, 
        net.offset, buffer_size, batch_size)
    train_dataset = gen_ds.dataset
    test_dataset = gen_ds.test_dataset

    num_examples = len(gen_ds.images)
    steps_per_epoch =  num_examples // batch_size

    checkpoint_prefix = "log_weighted/cp-{epoch:04d}-{val_loss:.2f}.ckpt"
    model_history = net.train(checkpoint_prefix, train_dataset, test_dataset, epochs, steps_per_epoch, valid_steps)

    # loss = model_history.history['loss']
    # val_loss = model_history.history['val_loss']

    # x = range(epochs)

    # plt.figure()
    # plt.plot(x, loss, 'r', label='Training loss')
    # plt.plot(x, val_loss, 'bo', label='Validation loss')
    # plt.title('Training and Validation Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss Value')
    # plt.ylim([0, 1])
    # plt.legend()
    # plt.show()

    model_path = tf.train.latest_checkpoint(os.path.dirname(checkpoint_prefix))
    model.load_weights(model_path)

    test_imgs = np.array([img.numpy() for img in gen_ds.images])
    test_imgs_gt = np.array([img.numpy() for img in gen_ds.gts])
    gt_pred = model.predict(test_imgs)
    pred_cat = stack_imgs(gt_pred, gen_ds.num_row, gen_ds.num_col)
    pred_gt_cat = np.argmax(pred_cat, -1)
    colors = cmp[pred_gt_cat.reshape(-1)]
    pred_gt_rgb = colors.reshape((*pred_gt_cat.shape, 3))
    cv.imwrite("pred_gt.png", pred_gt_cat)
    cv.imwrite("pred_gt_rgb.png", pred_gt_rgb)

    if net.offset >0:
        offset = net.offset
        test_imgs = test_imgs[:, offset:-offset, offset:-offset, :]
        test_imgs_gt = test_imgs_gt[:, offset:-offset, offset:-offset]
    img_cat = stack_imgs(test_imgs, gen_ds.num_row, gen_ds.num_col)
    cv.imwrite("test_img.png", (img_cat*255.).astype(np.uint8))
    img_gt_cat = stack_imgs(test_imgs_gt, gen_ds.num_row, gen_ds.num_col).astype(np.uint8)[..., 0]
    colors = cmp[img_gt_cat.reshape(-1)]
    gt_rgb = colors.reshape((*img_gt_cat.shape, 3))
    img_add = cv.addWeighted((img_cat*255).astype(gt_rgb.dtype), 0.8, gt_rgb, 0.2, 0)
    cv.imwrite("test_img_add.png", img_add)

    acc = img_gt_cat == pred_gt_cat
    print("accuracy: ", acc.mean())
