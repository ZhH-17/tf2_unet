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

from tensorflow.keras.layers import Conv2D, ReLU, Conv2DTranspose, MaxPool2D, Dropout
from tensorflow.keras import Model

from absl import app
from absl import flags
from unet_model import unet, create_mask, cmp


import pdb

class gen_from_image(object):
    def __init__(self, img_path, gt_path, nx, ny, offset, buffer_size, batch_size):
        self.img = cv.imread(img_path, -1)
        # normal
        self.img = tf.cast(self.img, tf.float32)/255.0

        self.img_gt = cv.imread(gt_path, -1)
        self.n_class = self.img_gt.max() + 1
        self.img_gt = tf.cast(np.atleast_3d(self.img_gt), tf.float32)
        h, w = self.img.shape[:2]
        # self.img_gt = np.zeros((h, w, 1))
        images = []
        gts = []
        self.num_row = int(np.ceil(float(h-2*offset)/(ny-2*offset)))
        self.num_col = int(np.ceil(float(w-2*offset)/(nx-2*offset)))
        for ih in range(0, h-2*offset, ny-2*offset):
            for iw in range(0, w-2*offset, nx-2*offset):
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

        inds = np.random.randint(0, len(gts), 3)
        ds_img_test = tf.data.Dataset.from_tensor_slices([images[i] for i in inds])
        ds_gt_test = tf.data.Dataset.from_tensor_slices([gts[i] for i in inds])
        self.test_dataset = tf.data.Dataset.zip((ds_img_test, ds_gt_test))
        self.test_dataset = self.test_dataset.batch(batch_size)

        ds_img = tf.data.Dataset.from_tensor_slices(images)
        ds_gt = tf.data.Dataset.from_tensor_slices(gts)

        self.dataset = tf.data.Dataset.zip((ds_img, ds_gt))
        self.dataset = self.dataset.shuffle(buffer_size).batch(batch_size).repeat()
        self.dataset = self.dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

def stack_imgs(imgs, num_row, num_col):
    '''
    concatenate image slices to a panoroma,
    imgs: image slices should be sorted by row first
    '''
    imgs_row = []
    for i in range(num_row):
        imgs_row.append(np.concatenate(imgs[i*num_col:(i+1)*num_col], axis=1))
    img_stack = np.concatenate(imgs_row, axis=0)
    return img_stack

if __name__ == "__main__":
    num_examples = 20
    batch_size = 3
    buffer_size = 30
    steps_per_epoch =  num_examples // batch_size
    n_class = 7
    dropout = 0
    epochs = 20
    valid_steps = 1

    nx = 572
    ny = 572
    # generator = image_gen.RgbDataProvider(nx, ny, cnt=10)
    # train_dataset = gen_dataset(generator, 20)
    # test_dataset = gen_dataset(generator, 2)

    # train_dataset = train_dataset.shuffle(buffer_size).batch(batch_size).repeat()
    # train_dataset1 = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    # test_dataset = test_dataset.batch(1)

    net = unet(dropout, n_class, shape=(ny, nx, 3), layers=3, padding='valid')
    model = net.model

    # gen_ds = gen_from_image("./data/cj_right_all.png", "./data/cj_right_all_gt.png", nx, ny, 
    #     net.offset, buffer_size, batch_size)
    gen_ds = gen_from_image("./data/cj_right_all_20200402_1420.png",
                           "./data/cj_right_all_20200402_1420_gt.png", nx, ny, net.offset, buffer_size, batch_size)
    train_dataset = gen_ds.dataset
    test_dataset = gen_ds.test_dataset

    checkpoint_prefix = "training_checkpoints/cp-{epoch:04d}-{val_loss:.2f}.ckpt"
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

    n_class = img_gt_cat.max()
    for i in range(0, n_class+1):
        mask = img_gt_cat == i
        if mask.sum()==0:
            print("class {} not exist in ground truth".format(i))
        else:
            acc = img_gt_cat[mask] == pred_gt_cat[mask]
            print("accuracy of class {}: {: .4f}".format(i, acc.mean()))
