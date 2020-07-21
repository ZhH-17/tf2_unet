from __future__ import print_function, division, absolute_import, unicode_literals

import os
import shutil
import numpy as np
import cv2 as cv
from collections import OrderedDict
import logging
import time
import matplotlib.pyplot as plt
import argparse

import tensorflow as tf
# tf.enable_eager_excution()

# from tf_unet import image_gen

from tensorflow.keras.layers import Conv2D, ReLU, Conv2DTranspose, MaxPool2D, Dropout
from tensorflow.keras import Model

from absl import app
from absl import flags
import sys
sys.path.append("../../")


from unet_model import unet, create_mask, cmp
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
        if os.path.isdir(img_path):
            # is dir
            fns = [os.path.join(img_path, fn) for fn in os.listdir(img_path) if fn[-3:]=="png"]
            fns_gt = [os.path.join(gt_path, fn) for fn in os.listdir(gt_path) if "gt" in fn]
            fns.sort()
            fns_gt.sort()
            for fn, fn_gt in zip(fns, fns_gt):
                print(fn, fn_gt)
            assert len(fns) == len(fns_gt)
            images = []
            gts = []
            for i in range(len(fns)):
                img = cv.imread(fns[i], -1)
                img = cv.resize(img, (nx,ny))
                img = np.atleast_3d(img)
                img = tf.cast(img, tf.float32)/255.0

                img_gt = cv.imread(fns_gt[i], -1)
                self.n_class = img_gt.max() + 1
                img_gt = cv.resize(img_gt, (nx,ny))
                img_gt = tf.cast(np.atleast_3d(img_gt), tf.float32)

                images.append(img)
                gts.append(img_gt)
        else:
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

        size_test = 1
        # inds = np.random.randint(0, len(gts), size_test)
        inds = [0]
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
    parser = argparse.ArgumentParser(
        description='Train Unet.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'evaluate' on dataset")
    parser.add_argument('--dataset', required=True,
                        metavar="/path/to/data/",
                        help='Directory of dataset')
    parser.add_argument('--model', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to model files ")
    args = parser.parse_args()
    mode = args.command
    dataset = args.dataset
    log_path = args.model

    # batch_size = 1
    # buffer_size = 1
    # dropout = 0
    # epochs = 2000
    # valid_steps = 1

    batch_size = 4
    buffer_size = 10
    dropout = 0
    epochs = 2000
    valid_steps = 1

    nx = 572
    ny = 572
    # nx = 1080
    # ny = 1080

    n_class = 2
    net = unet(dropout, n_class,
               shape=(ny, nx, 1), layers=5, padding='valid')
    model = net.model
    # checkpoint_prefix = "log_5pics" + "/cp-{epoch:04d}-{val_loss:.2f}.ckpt"

    if mode == "train":
        # gen_ds = gen_from_image("./dataset/data_5pics/imgs/", "./dataset/data_5pics/gts",
        #                         nx, ny, net.offset, buffer_size, batch_size)
        gen_ds = gen_from_image(os.path.join(dataset, "./imgs/"), os.path.join(dataset, "./gts/"),
                                nx, ny, net.offset, buffer_size, batch_size)
        train_dataset = gen_ds.dataset
        test_dataset = gen_ds.test_dataset

        num_examples = len(gen_ds.images)
        steps_per_epoch =  num_examples // batch_size + 1
        print("image number: ", num_examples)

        checkpoint_prefix = log_path + "/cp-{epoch:04d}-{val_loss:.2f}.ckpt"
        model_history = net.train(checkpoint_prefix, train_dataset, test_dataset, epochs, steps_per_epoch, valid_steps)

        loss = model_history.history['loss']
        val_loss = model_history.history['val_loss']

        x = range(epochs)
        plt.figure()
        plt.plot(x, loss, 'r', label='Training loss')
        plt.plot(x, val_loss, 'bo', label='Validation loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss Value')
        plt.ylim([0, 1])
        plt.legend()
        plt.show()
    elif mode == "evaluate":
        gen_ds = gen_from_image(os.path.join(dataset, "./imgs/"), os.path.join(dataset, "./gts/"),
                                nx, ny, net.offset, buffer_size, batch_size)
        model_path = tf.train.latest_checkpoint(log_path)
        if os.path.isdir(log_path):
            model_path = tf.train.latest_checkpoint(log_path)
        else:
            model_path = log_path
        model.load_weights(model_path)

        test_imgs = np.array([img.numpy() for img in gen_ds.images])
        test_imgs_gt = np.array([img.numpy() for img in gen_ds.gts],
                                dtype=np.uint8).squeeze(axis=-1)

        pred_prob = model.predict(test_imgs)
        pred_gt = np.argmax(pred_prob, -1)

        colors = [cmp[img] for img in pred_gt]
        pred_gt_rgb = [color.reshape((*pred_gt[0].shape, 3)) for color in colors]
        pred_gt_rgb_cat = np.concatenate(pred_gt_rgb, axis=0)

        if net.offset >0:
            offset = net.offset
            test_imgs = test_imgs[:, offset:-offset, offset:-offset, :]
            test_imgs_gt = test_imgs_gt[:, offset:-offset, offset:-offset]
        if test_imgs.shape[-1] == 1:
            test_imgs = np.concatenate([test_imgs, test_imgs, test_imgs], axis=-1)
        test_imgs_cat = np.concatenate((test_imgs*255.0).astype(np.uint8), axis=0)

        colors = [cmp[img] for img in test_imgs_gt]
        imgs_gt_rgb = [color.reshape((*test_imgs_gt[0].shape, 3)) for color in colors]
        imgs_gt_rgb_cat = np.concatenate(imgs_gt_rgb, axis=0)

        img_total = np.concatenate([test_imgs_cat, imgs_gt_rgb_cat, pred_gt_rgb_cat], axis=1)
        cv.imwrite(os.path.join(log_path, "pred_total.png"), img_total)

        acces = np.array([
            (gt == gt_pred) for gt, gt_pred in zip(test_imgs_gt, pred_gt)])
        print("accuracy for every image: ", )
        print(*(np.mean(acces, axis=(1, 2))) )
        print("Total accuracy: ", acces.mean())
    elif mode == "predict":
        pass
