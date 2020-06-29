#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-05-17
# @Author  : Shawn Shan (shansixiong@cs.uchicago.edu)
# @Link    : https://www.shawnshan.com/

import datetime
import time
from decimal import Decimal

import numpy as np
import tensorflow as tf
from utils import preprocess, reverse_preprocess


class FawkesMaskGeneration:
    # if the attack is trying to mimic a target image or a neuron vector
    MIMIC_IMG = True
    # number of iterations to perform gradient descent
    MAX_ITERATIONS = 10000
    # larger values converge faster to less accurate results
    LEARNING_RATE = 1e-2
    # the initial constant c to pick as a first guess
    INITIAL_CONST = 1
    # pixel intensity range
    INTENSITY_RANGE = 'imagenet'
    # threshold for distance
    L_THRESHOLD = 0.03
    # whether keep the final result or the best result
    KEEP_FINAL = False
    # max_val of image
    MAX_VAL = 255
    # The following variables are used by DSSIM, should keep as default
    # filter size in SSIM
    FILTER_SIZE = 11
    # filter sigma in SSIM
    FILTER_SIGMA = 1.5
    # weights used in MS-SSIM
    SCALE_WEIGHTS = None
    MAXIMIZE = False
    IMAGE_SHAPE = (224, 224, 3)
    RATIO = 1.0
    LIMIT_DIST = False

    def __init__(self, sess, bottleneck_model_ls, mimic_img=MIMIC_IMG,
                 batch_size=1, learning_rate=LEARNING_RATE,
                 max_iterations=MAX_ITERATIONS, initial_const=INITIAL_CONST,
                 intensity_range=INTENSITY_RANGE, l_threshold=L_THRESHOLD,
                 max_val=MAX_VAL, keep_final=KEEP_FINAL, maximize=MAXIMIZE, image_shape=IMAGE_SHAPE,
                 verbose=0, ratio=RATIO, limit_dist=LIMIT_DIST):

        assert intensity_range in {'raw', 'imagenet', 'inception', 'mnist'}

        # constant used for tanh transformation to avoid corner cases
        self.tanh_constant = 2 - 1e-6
        self.sess = sess
        self.MIMIC_IMG = mimic_img
        self.LEARNING_RATE = learning_rate
        self.MAX_ITERATIONS = max_iterations
        self.initial_const = initial_const
        self.batch_size = batch_size
        self.intensity_range = intensity_range
        self.l_threshold = l_threshold
        self.max_val = max_val
        self.keep_final = keep_final
        self.verbose = verbose
        self.maximize = maximize
        self.learning_rate = learning_rate
        self.ratio = ratio
        self.limit_dist = limit_dist
        self.single_shape = list(image_shape)

        self.input_shape = tuple([self.batch_size] + self.single_shape)

        self.bottleneck_shape = tuple([self.batch_size] + self.single_shape)

        # the variable we're going to optimize over
        self.modifier = tf.Variable(np.zeros(self.input_shape, dtype=np.float32))

        # target image in tanh space
        if self.MIMIC_IMG:
            self.timg_tanh = tf.Variable(np.zeros(self.input_shape), dtype=np.float32)
        else:
            self.bottleneck_t_raw = tf.Variable(np.zeros(self.bottleneck_shape), dtype=np.float32)
        # source image in tanh space
        self.simg_tanh = tf.Variable(np.zeros(self.input_shape), dtype=np.float32)

        self.const = tf.Variable(np.ones(batch_size), dtype=np.float32)
        self.mask = tf.Variable(np.ones((batch_size), dtype=np.bool))
        self.weights = tf.Variable(np.ones(self.bottleneck_shape,
                                           dtype=np.float32))

        # and here's what we use to assign them
        self.assign_modifier = tf.placeholder(tf.float32, self.input_shape)
        if self.MIMIC_IMG:
            self.assign_timg_tanh = tf.placeholder(
                tf.float32, self.input_shape)
        else:
            self.assign_bottleneck_t_raw = tf.placeholder(
                tf.float32, self.bottleneck_shape)
        self.assign_simg_tanh = tf.placeholder(tf.float32, self.input_shape)
        self.assign_const = tf.placeholder(tf.float32, (batch_size))
        self.assign_mask = tf.placeholder(tf.bool, (batch_size))
        self.assign_weights = tf.placeholder(tf.float32, self.bottleneck_shape)

        # the resulting image, tanh'd to keep bounded from -0.5 to 0.5
        # adversarial image in raw space
        self.aimg_raw = (tf.tanh(self.modifier + self.simg_tanh) /
                         self.tanh_constant +
                         0.5) * 255.0
        # source image in raw space
        self.simg_raw = (tf.tanh(self.simg_tanh) /
                         self.tanh_constant +
                         0.5) * 255.0
        if self.MIMIC_IMG:
            # target image in raw space
            self.timg_raw = (tf.tanh(self.timg_tanh) /
                             self.tanh_constant +
                             0.5) * 255.0

        # convert source and adversarial image into input space
        if self.intensity_range == 'imagenet':
            mean = tf.constant(np.repeat([[[[103.939, 116.779, 123.68]]]], self.batch_size, axis=0), dtype=tf.float32,
                               name='img_mean')
            self.aimg_input = (self.aimg_raw[..., ::-1] - mean)
            self.simg_input = (self.simg_raw[..., ::-1] - mean)
            if self.MIMIC_IMG:
                self.timg_input = (self.timg_raw[..., ::-1] - mean)

        elif self.intensity_range == 'raw':
            self.aimg_input = self.aimg_raw
            self.simg_input = self.simg_raw
            if self.MIMIC_IMG:
                self.timg_input = self.timg_raw

        def batch_gen_DSSIM(aimg_raw_split, simg_raw_split):
            msssim_split = tf.image.ssim(aimg_raw_split, simg_raw_split, max_val=255.0)
            dist = (1.0 - tf.stack(msssim_split)) / 2.0
            # dist = tf.square(aimg_raw_split - simg_raw_split)
            return dist

        # raw value of DSSIM distance
        self.dist_raw = batch_gen_DSSIM(self.aimg_raw, self.simg_raw)
        # distance value after applying threshold
        self.dist = tf.maximum(self.dist_raw - self.l_threshold, 0.0)
        # self.dist = self.dist_raw
        self.dist_raw_sum = tf.reduce_sum(
            tf.where(self.mask,
                     self.dist_raw,
                     tf.zeros_like(self.dist_raw)))
        self.dist_sum = tf.reduce_sum(tf.where(self.mask, self.dist, tf.zeros_like(self.dist)))
        # self.dist_sum = 1e-5 * tf.reduce_sum(self.dist)
        # self.dist_raw_sum = self.dist_sum

        def resize_tensor(input_tensor, model_input_shape):
            if input_tensor.shape[1:] == model_input_shape or model_input_shape[1] is None:
                return input_tensor
            resized_tensor = tf.image.resize(input_tensor, model_input_shape[:2])
            return resized_tensor

        def calculate_direction(bottleneck_model, cur_timg_input, cur_simg_input):
            target_features = bottleneck_model(cur_timg_input)
            return target_features

        self.bottlesim = 0.0
        self.bottlesim_sum = 0.0
        self.bottlesim_push = 0.0
        for bottleneck_model in bottleneck_model_ls:
            model_input_shape = bottleneck_model.input_shape[1:]
            cur_aimg_input = resize_tensor(self.aimg_input, model_input_shape)

            self.bottleneck_a = bottleneck_model(cur_aimg_input)
            if self.MIMIC_IMG:
                # cur_timg_input = resize_tensor(self.timg_input, model_input_shape)
                # cur_simg_input = resize_tensor(self.simg_input, model_input_shape)
                cur_timg_input = self.timg_input
                cur_simg_input = self.simg_input
                self.bottleneck_t = calculate_direction(bottleneck_model, cur_timg_input, cur_simg_input)
                # self.bottleneck_t = bottleneck_model(cur_timg_input)
            else:
                self.bottleneck_t = self.bottleneck_t_raw

            bottleneck_diff = self.bottleneck_t - self.bottleneck_a
            scale_factor = tf.sqrt(tf.reduce_sum(tf.square(self.bottleneck_t), axis=1))

            cur_bottlesim = tf.sqrt(tf.reduce_sum(tf.square(bottleneck_diff), axis=1))
            cur_bottlesim = cur_bottlesim / scale_factor
            cur_bottlesim_sum = tf.reduce_sum(cur_bottlesim)

            self.bottlesim += cur_bottlesim

            # self.bottlesim_push += cur_bottlesim_push_sum
            self.bottlesim_sum += cur_bottlesim_sum

        # sum up the losses
        if self.maximize:
            self.loss = self.const * tf.square(self.dist) - self.bottlesim
        else:
            self.loss = self.const * tf.square(self.dist) + self.bottlesim

        self.loss_sum = tf.reduce_sum(tf.where(self.mask,
                                               self.loss,
                                               tf.zeros_like(self.loss)))

        # self.loss_sum = self.dist_sum + tf.reduce_sum(self.bottlesim)
        # import pdb
        # pdb.set_trace()
        # self.loss_sum = tf.reduce_sum(tf.where(self.mask, self.loss, tf.zeros_like(self.loss)))

        # Setup the Adadelta optimizer and keep track of variables
        # we're creating
        start_vars = set(x.name for x in tf.global_variables())
        self.learning_rate_holder = tf.placeholder(tf.float32, shape=[])
        optimizer = tf.train.AdadeltaOptimizer(self.learning_rate_holder)
        # optimizer = tf.train.AdamOptimizer(self.learning_rate_holder)

        self.train = optimizer.minimize(self.loss_sum,
                                        var_list=[self.modifier])
        end_vars = tf.global_variables()
        new_vars = [x for x in end_vars if x.name not in start_vars]

        # these are the variables to initialize when we run
        self.setup = []
        self.setup.append(self.modifier.assign(self.assign_modifier))
        if self.MIMIC_IMG:
            self.setup.append(self.timg_tanh.assign(self.assign_timg_tanh))
        else:
            self.setup.append(self.bottleneck_t_raw.assign(
                self.assign_bottleneck_t_raw))
        self.setup.append(self.simg_tanh.assign(self.assign_simg_tanh))
        self.setup.append(self.const.assign(self.assign_const))
        self.setup.append(self.mask.assign(self.assign_mask))
        self.setup.append(self.weights.assign(self.assign_weights))

        self.init = tf.variables_initializer(var_list=[self.modifier] + new_vars)

        print('Attacker loaded')

    def preprocess_arctanh(self, imgs):

        imgs = reverse_preprocess(imgs, self.intensity_range)
        imgs /= 255.0
        imgs -= 0.5
        imgs *= self.tanh_constant
        tanh_imgs = np.arctanh(imgs)

        return tanh_imgs

    def clipping(self, imgs):

        imgs = reverse_preprocess(imgs, self.intensity_range)
        imgs = np.clip(imgs, 0, self.max_val)
        imgs = preprocess(imgs, self.intensity_range)

        return imgs

    def attack(self, source_imgs, target_imgs, weights=None):

        if weights is None:
            weights = np.ones([source_imgs.shape[0]] +
                              list(self.bottleneck_shape[1:]))

        assert weights.shape[1:] == self.bottleneck_shape[1:]
        assert source_imgs.shape[1:] == self.input_shape[1:]
        assert source_imgs.shape[0] == weights.shape[0]
        if self.MIMIC_IMG:
            assert target_imgs.shape[1:] == self.input_shape[1:]
            assert source_imgs.shape[0] == target_imgs.shape[0]
        else:
            assert target_imgs.shape[1:] == self.bottleneck_shape[1:]
            assert source_imgs.shape[0] == target_imgs.shape[0]

        start_time = time.time()

        adv_imgs = []
        print('%d batches in total'
              % int(np.ceil(len(source_imgs) / self.batch_size)))

        for idx in range(0, len(source_imgs), self.batch_size):
            print('processing batch %d at %s' % (idx, datetime.datetime.now()))
            adv_img = self.attack_batch(source_imgs[idx:idx + self.batch_size],
                                        target_imgs[idx:idx + self.batch_size],
                                        weights[idx:idx + self.batch_size])
            adv_imgs.extend(adv_img)

        elapsed_time = time.time() - start_time
        print('attack cost %f s' % (elapsed_time))

        return np.array(adv_imgs)

    def attack_batch(self, source_imgs, target_imgs, weights):

        """
        Run the attack on a batch of images and labels.
        """

        LR = self.learning_rate
        nb_imgs = source_imgs.shape[0]
        mask = [True] * nb_imgs + [False] * (self.batch_size - nb_imgs)
        mask = np.array(mask, dtype=np.bool)

        source_imgs = np.array(source_imgs)
        target_imgs = np.array(target_imgs)

        # convert to tanh-space
        simg_tanh = self.preprocess_arctanh(source_imgs)
        if self.MIMIC_IMG:
            timg_tanh = self.preprocess_arctanh(target_imgs)
        else:
            timg_tanh = target_imgs

        CONST = np.ones(self.batch_size) * self.initial_const

        self.sess.run(self.init)
        simg_tanh_batch = np.zeros(self.input_shape)
        if self.MIMIC_IMG:
            timg_tanh_batch = np.zeros(self.input_shape)
        else:
            timg_tanh_batch = np.zeros(self.bottleneck_shape)
        weights_batch = np.zeros(self.bottleneck_shape)
        simg_tanh_batch[:nb_imgs] = simg_tanh[:nb_imgs]
        timg_tanh_batch[:nb_imgs] = timg_tanh[:nb_imgs]
        weights_batch[:nb_imgs] = weights[:nb_imgs]
        modifier_batch = np.ones(self.input_shape) * 1e-6

        self.sess.run(self.setup,
                      {self.assign_timg_tanh: timg_tanh_batch,
                       self.assign_simg_tanh: simg_tanh_batch,
                       self.assign_const: CONST,
                       self.assign_mask: mask,
                       self.assign_weights: weights_batch,
                       self.assign_modifier: modifier_batch})

        best_bottlesim = [0] * nb_imgs if self.maximize else [np.inf] * nb_imgs
        best_adv = np.zeros_like(source_imgs)

        if self.verbose == 1:
            loss_sum = float(self.sess.run(self.loss_sum))
            dist_sum = float(self.sess.run(self.dist_sum))
            thresh_over = (dist_sum / self.batch_size / self.l_threshold * 100)
            dist_raw_sum = float(self.sess.run(self.dist_raw_sum))
            bottlesim_sum = self.sess.run(self.bottlesim_sum)
            print('START: Total loss: %.4E; perturb: %.6f (%.2f%% over, raw: %.6f); sim: %f'
                  % (Decimal(loss_sum),
                     dist_sum,
                     thresh_over,
                     dist_raw_sum,
                     bottlesim_sum / nb_imgs))

        try:
            total_distance = [0] * nb_imgs

            if self.limit_dist:
                dist_raw_list, bottlesim_list, aimg_input_list = self.sess.run(
                    [self.dist_raw,
                     self.bottlesim,
                     self.aimg_input])
                for e, (dist_raw, bottlesim, aimg_input) in enumerate(
                        zip(dist_raw_list, bottlesim_list, aimg_input_list)):
                    if e >= nb_imgs:
                        break
                    total_distance[e] = bottlesim

            for iteration in range(self.MAX_ITERATIONS):

                self.sess.run([self.train], feed_dict={self.learning_rate_holder: LR})

                dist_raw_list, bottlesim_list, aimg_input_list = self.sess.run(
                    [self.dist_raw,
                     self.bottlesim,
                     self.aimg_input])
                for e, (dist_raw, bottlesim, aimg_input) in enumerate(
                        zip(dist_raw_list, bottlesim_list, aimg_input_list)):
                    if e >= nb_imgs:
                        break
                    if (bottlesim < best_bottlesim[e] and bottlesim > total_distance[e] * 0.1 and (
                            not self.maximize)) or (
                            bottlesim > best_bottlesim[e] and self.maximize):
                        best_bottlesim[e] = bottlesim
                        best_adv[e] = aimg_input

                if iteration != 0 and iteration % (self.MAX_ITERATIONS // 3) == 0:
                    # LR = LR / 2
                    print("Learning Rate: ", LR)

                if iteration % (self.MAX_ITERATIONS // 10) == 0:
                    if self.verbose == 1:
                        loss_sum = float(self.sess.run(self.loss_sum))
                        dist_sum = float(self.sess.run(self.dist_sum))
                        thresh_over = (dist_sum /
                                       self.batch_size /
                                       self.l_threshold *
                                       100)
                        dist_raw_sum = float(self.sess.run(self.dist_raw_sum))
                        bottlesim_sum = self.sess.run(self.bottlesim_sum)
                        print('ITER %4d: Total loss: %.4E; perturb: %.6f (%.2f%% over, raw: %.6f); sim: %f'
                              % (iteration,
                                 Decimal(loss_sum),
                                 dist_sum,
                                 thresh_over,
                                 dist_raw_sum,
                                 bottlesim_sum / nb_imgs))
        except KeyboardInterrupt:
            pass

        if self.verbose == 1:
            loss_sum = float(self.sess.run(self.loss_sum))
            dist_sum = float(self.sess.run(self.dist_sum))
            thresh_over = (dist_sum / self.batch_size / self.l_threshold * 100)
            dist_raw_sum = float(self.sess.run(self.dist_raw_sum))
            bottlesim_sum = float(self.sess.run(self.bottlesim_sum))
            print('END:       Total loss: %.4E; perturb: %.6f (%.2f%% over, raw: %.6f); sim: %f'
                  % (Decimal(loss_sum),
                     dist_sum,
                     thresh_over,
                     dist_raw_sum,
                     bottlesim_sum / nb_imgs))

        best_adv = self.clipping(best_adv[:nb_imgs])

        return best_adv
