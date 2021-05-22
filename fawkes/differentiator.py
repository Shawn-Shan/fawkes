#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-10-21
# @Author  : Emily Wenger (ewenger@uchicago.edu)

import datetime
import time

import numpy as np
import tensorflow as tf
from fawkes.utils import preprocess, reverse_preprocess
from keras.utils import Progbar


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
    MAXIMIZE = False
    IMAGE_SHAPE = (112, 112, 3)
    RATIO = 1.0
    LIMIT_DIST = False
    LOSS_TYPE = 'features'  # use features (original Fawkes) or gradients (Witches Brew) to run Fawkes?

    def __init__(self, bottleneck_model_ls, mimic_img=MIMIC_IMG,
                 batch_size=1, learning_rate=LEARNING_RATE,
                 max_iterations=MAX_ITERATIONS, initial_const=INITIAL_CONST,
                 intensity_range=INTENSITY_RANGE, l_threshold=L_THRESHOLD,
                 max_val=MAX_VAL, keep_final=KEEP_FINAL, maximize=MAXIMIZE, image_shape=IMAGE_SHAPE, verbose=1,
                 ratio=RATIO, limit_dist=LIMIT_DIST, loss_method=LOSS_TYPE, tanh_process=True,
                 save_last_on_failed=True):

        assert intensity_range in {'raw', 'imagenet', 'inception', 'mnist'}

        # constant used for tanh transformation to avoid corner cases

        self.it = 0
        self.tanh_constant = 2 - 1e-6
        self.save_last_on_failed = save_last_on_failed
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
        self.bottleneck_models = bottleneck_model_ls
        self.loss_method = loss_method
        self.tanh_process = tanh_process

    @staticmethod
    def resize_tensor(input_tensor, model_input_shape):
        if input_tensor.shape[1:] == model_input_shape or model_input_shape[1] is None:
            return input_tensor
        resized_tensor = tf.image.resize(input_tensor, model_input_shape[:2])
        return resized_tensor

    def preprocess_arctanh(self, imgs):
        """ Do tan preprocess """
        imgs = reverse_preprocess(imgs, self.intensity_range)
        imgs = imgs / 255.0
        imgs = imgs - 0.5
        imgs = imgs * self.tanh_constant
        tanh_imgs = np.arctanh(imgs)
        return tanh_imgs

    def reverse_arctanh(self, imgs):
        raw_img = (tf.tanh(imgs) / self.tanh_constant + 0.5) * 255
        return raw_img

    def input_space_process(self, img):
        if self.intensity_range == 'imagenet':
            mean = np.repeat([[[[103.939, 116.779, 123.68]]]], len(img), axis=0)
            raw_img = (img[..., ::-1] - mean)
        else:
            raw_img = img
        return raw_img

    def clipping(self, imgs):
        imgs = reverse_preprocess(imgs, self.intensity_range)
        imgs = np.clip(imgs, 0, self.max_val)
        imgs = preprocess(imgs, self.intensity_range)
        return imgs

    def calc_dissim(self, source_raw, source_mod_raw):
        msssim_split = tf.image.ssim(source_raw, source_mod_raw, max_val=255.0)
        dist_raw = (1.0 - tf.stack(msssim_split)) / 2.0
        dist = tf.maximum(dist_raw - self.l_threshold, 0.0)
        dist_raw_avg = tf.reduce_mean(dist_raw)
        dist_sum = tf.reduce_sum(dist)

        return dist, dist_raw, dist_sum, dist_raw_avg

    def calc_bottlesim(self, tape, source_raw, target_raw, original_raw):
        """ original Fawkes loss function. """
        bottlesim = 0.0
        bottlesim_sum = 0.0
        # make sure everything is the right size.
        model_input_shape = self.single_shape
        cur_aimg_input = self.resize_tensor(source_raw, model_input_shape)
        if target_raw is not None:
            cur_timg_input = self.resize_tensor(target_raw, model_input_shape)
        for bottleneck_model in self.bottleneck_models:
            if tape is not None:
                try:
                    tape.watch(bottleneck_model.model.variables)
                except AttributeError:
                    tape.watch(bottleneck_model.variables)
            # get the respective feature space reprs.
            bottleneck_a = bottleneck_model(cur_aimg_input)
            if self.maximize:
                bottleneck_s = bottleneck_model(original_raw)
                bottleneck_diff = bottleneck_a - bottleneck_s
                scale_factor = tf.sqrt(tf.reduce_sum(tf.square(bottleneck_s), axis=1))
            else:
                bottleneck_t = bottleneck_model(cur_timg_input)
                bottleneck_diff = bottleneck_t - bottleneck_a
                scale_factor = tf.sqrt(tf.reduce_sum(tf.square(bottleneck_t), axis=1))
            cur_bottlesim = tf.reduce_sum(tf.square(bottleneck_diff), axis=1)
            cur_bottlesim = cur_bottlesim / scale_factor
            bottlesim += cur_bottlesim
            bottlesim_sum += tf.reduce_sum(cur_bottlesim)
        return bottlesim, bottlesim_sum

    def compute_feature_loss(self, tape, aimg_raw, simg_raw, aimg_input, timg_input, simg_input):
        """ Compute input space + feature space loss.
        """
        input_space_loss, dist_raw, input_space_loss_sum, input_space_loss_raw_avg = self.calc_dissim(aimg_raw,
                                                                                                      simg_raw)
        feature_space_loss, feature_space_loss_sum = self.calc_bottlesim(tape, aimg_input, timg_input, simg_input)

        if self.maximize:
            loss = self.const * tf.square(input_space_loss) - feature_space_loss * self.const_diff
        else:
            if self.it < self.MAX_ITERATIONS:
                loss = self.const * tf.square(input_space_loss) + 1000 * feature_space_loss

        loss_sum = tf.reduce_sum(loss)
        return loss_sum, feature_space_loss, input_space_loss_raw_avg, dist_raw

    def compute(self, source_imgs, target_imgs=None):
        """ Main function that runs cloak generation. """
        start_time = time.time()
        adv_imgs = []
        for idx in range(0, len(source_imgs), self.batch_size):
            print('processing image %d at %s' % (idx + 1, datetime.datetime.now()))
            adv_img = self.compute_batch(source_imgs[idx:idx + self.batch_size],
                                         target_imgs[idx:idx + self.batch_size] if target_imgs is not None else None)
            adv_imgs.extend(adv_img)
        elapsed_time = time.time() - start_time
        print('protection cost %f s' % elapsed_time)
        return np.array(adv_imgs)

    def compute_batch(self, source_imgs, target_imgs=None, retry=True):
        """ TF2 method to generate the cloak. """
        # preprocess images.
        global progressbar
        nb_imgs = source_imgs.shape[0]

        # make sure source/target images are an array
        source_imgs = np.array(source_imgs, dtype=np.float32)
        if target_imgs is not None:
            target_imgs = np.array(target_imgs, dtype=np.float32)

        # metrics to test
        best_bottlesim = [0] * nb_imgs if self.maximize else [np.inf] * nb_imgs
        best_adv = np.zeros(source_imgs.shape)

        # convert to tanh-space
        simg_tanh = self.preprocess_arctanh(source_imgs)
        if target_imgs is not None:
            timg_tanh = self.preprocess_arctanh(target_imgs)
        self.modifier = tf.Variable(np.random.uniform(-1, 1, tuple([len(source_imgs)] + self.single_shape)) * 1e-4,
                                    dtype=tf.float32)

        # make the optimizer
        optimizer = tf.keras.optimizers.Adadelta(float(self.learning_rate))
        const_numpy = np.ones(len(source_imgs)) * self.initial_const
        self.const = tf.Variable(const_numpy, dtype=np.float32)

        const_diff_numpy = np.ones(len(source_imgs)) * 1.0
        self.const_diff = tf.Variable(const_diff_numpy, dtype=np.float32)

        # get the modifier
        if self.verbose == 0:
            progressbar = Progbar(
                self.MAX_ITERATIONS, width=30, verbose=1
            )
        # watch relevant variables.
        simg_tanh = tf.Variable(simg_tanh, dtype=np.float32)
        simg_raw = tf.Variable(source_imgs, dtype=np.float32)
        if target_imgs is not None:
            timg_raw = tf.Variable(timg_tanh, dtype=np.float32)
        # run the attack
        outside_list = np.ones(len(source_imgs))
        self.it = 0

        while self.it < self.MAX_ITERATIONS:

            self.it += 1
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(self.modifier)
                tape.watch(simg_tanh)

                # Convert from tanh for DISSIM
                aimg_raw = self.reverse_arctanh(simg_tanh + self.modifier)

                actual_modifier = aimg_raw - simg_raw
                actual_modifier = tf.clip_by_value(actual_modifier, -15.0, 15.0)
                aimg_raw = simg_raw + actual_modifier

                simg_raw = self.reverse_arctanh(simg_tanh)

                # Convert further preprocess for bottleneck
                aimg_input = self.input_space_process(aimg_raw)
                if target_imgs is not None:
                    timg_input = self.input_space_process(timg_raw)
                else:
                    timg_input = None
                simg_input = self.input_space_process(simg_raw)

                # get the feature space loss.
                loss, internal_dist, input_dist_avg, dist_raw = self.compute_feature_loss(
                    tape, aimg_raw, simg_raw, aimg_input, timg_input, simg_input)

                # compute gradients
                grad = tape.gradient(loss, [self.modifier])
                optimizer.apply_gradients(zip(grad, [self.modifier]))

            if self.it == 1:
                self.modifier = tf.Variable(self.modifier - tf.sign(grad[0]) * 0.01, dtype=tf.float32)

            for e, (input_dist, feature_d, mod_img) in enumerate(zip(dist_raw, internal_dist, aimg_input)):
                if e >= nb_imgs:
                    break
                input_dist = input_dist.numpy()
                feature_d = feature_d.numpy()

                if input_dist <= self.l_threshold * 0.9 and const_diff_numpy[e] <= 129:
                    const_diff_numpy[e] *= 2
                    if outside_list[e] == -1:
                        const_diff_numpy[e] = 1
                    outside_list[e] = 1
                elif input_dist >= self.l_threshold * 1.1 and const_diff_numpy[e] >= 1 / 129:
                    const_diff_numpy[e] /= 2

                    if outside_list[e] == 1:
                        const_diff_numpy[e] = 1
                    outside_list[e] = -1
                else:
                    const_diff_numpy[e] = 1.0
                    outside_list[e] = 0

                if input_dist <= self.l_threshold * 1.1 and (
                        (feature_d < best_bottlesim[e] and (not self.maximize)) or (
                        feature_d > best_bottlesim[e] and self.maximize)):
                    best_bottlesim[e] = feature_d
                    best_adv[e] = mod_img

            self.const_diff = tf.Variable(const_diff_numpy, dtype=np.float32)

            if self.verbose == 1:
                print("ITER {:0.2f}  Total Loss: {:.2f} {:0.4f} raw; diff: {:.4f}".format(self.it, loss, input_dist_avg,
                                                                                          np.mean(internal_dist)))

            if self.verbose == 0:
                progressbar.update(self.it)
        if self.verbose == 1:
            print("Final diff: {:.4f}".format(np.mean(best_bottlesim)))
        print("\n")

        if self.save_last_on_failed:
            for e, diff in enumerate(best_bottlesim):
                if diff < 0.3 and dist_raw[e] < 0.015 and internal_dist[e] > diff:
                    best_adv[e] = aimg_input[e]

        best_adv = self.clipping(best_adv[:nb_imgs])
        return best_adv
