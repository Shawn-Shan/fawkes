#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-05-17
# @Author  : Shawn Shan (shansixiong@cs.uchicago.edu)
# @Link    : https://www.shawnshan.com/

import argparse
import glob
import logging
import os
import sys

import tensorflow as tf

logging.getLogger('tensorflow').disabled = True

import numpy as np
from fawkes.differentiator import FawkesMaskGeneration
from fawkes.utils import load_extractor, init_gpu, select_target_label, dump_image, reverse_process_cloaked, \
    Faces, filter_image_paths

from fawkes.align_face import aligner
from fawkes.utils import get_file


def generate_cloak_images(protector, image_X, target_emb=None):
    cloaked_image_X = protector.attack(image_X, target_emb)
    return cloaked_image_X


class Fawkes(object):
    def __init__(self, feature_extractor, gpu, batch_size):

        self.feature_extractor = feature_extractor
        self.gpu = gpu
        self.batch_size = batch_size
        global sess
        sess = init_gpu(gpu)
        global graph
        graph = tf.get_default_graph()

        model_dir = os.path.join(os.path.expanduser('~'), '.fawkes')
        if not os.path.exists(os.path.join(model_dir, "mtcnn.p.gz")):
            os.makedirs(model_dir, exist_ok=True)
            get_file("mtcnn.p.gz", "http://mirror.cs.uchicago.edu/fawkes/files/mtcnn.p.gz", cache_dir=model_dir,
                     cache_subdir='')

        self.fs_names = [feature_extractor]
        if isinstance(feature_extractor, list):
            self.fs_names = feature_extractor

        self.aligner = aligner(sess)
        self.feature_extractors_ls = [load_extractor(name) for name in self.fs_names]

        self.protector = None
        self.protector_param = None

    def mode2param(self, mode):
        if mode == 'min':
            th = 0.002
            max_step = 20
            lr = 40
        elif mode == 'low':
            th = 0.003
            max_step = 50
            lr = 35
        elif mode == 'mid':
            th = 0.005
            max_step = 200
            lr = 20
        elif mode == 'high':
            th = 0.008
            max_step = 500
            lr = 10
        elif mode == 'ultra':
            if not tf.test.is_gpu_available():
                print("Please enable GPU for ultra setting...")
                sys.exit(1)
            th = 0.01
            max_step = 1000
            lr = 8
        else:
            raise Exception("mode must be one of 'min', 'low', 'mid', 'high', 'ultra', 'custom'")
        return th, max_step, lr

    def run_protection(self, image_paths, mode='min', th=0.04, sd=1e9, lr=10, max_step=500, batch_size=1, format='png',
                       separate_target=True, debug=False, no_align=False):
        if mode == 'custom':
            pass
        else:
            th, max_step, lr = self.mode2param(mode)

        current_param = "-".join([str(x) for x in [mode, th, sd, lr, max_step, batch_size, format,
                                                   separate_target, debug]])

        image_paths, loaded_images = filter_image_paths(image_paths)

        if not image_paths:
            print("No images in the directory")
            return 3

        with graph.as_default():
            faces = Faces(image_paths, loaded_images, self.aligner, verbose=1, no_align=no_align)
            original_images = faces.cropped_faces

            if len(original_images) == 0:
                print("No face detected. ")
                return 2
            original_images = np.array(original_images)

            with sess.as_default():
                if separate_target:
                    target_embedding = []
                    for org_img in original_images:
                        org_img = org_img.reshape([1] + list(org_img.shape))
                        tar_emb = select_target_label(org_img, self.feature_extractors_ls, self.fs_names)
                        target_embedding.append(tar_emb)
                    target_embedding = np.concatenate(target_embedding)
                else:
                    target_embedding = select_target_label(original_images, self.feature_extractors_ls, self.fs_names)

                if current_param != self.protector_param:
                    self.protector_param = current_param

                    if self.protector is not None:
                        del self.protector

                    self.protector = FawkesMaskGeneration(sess, self.feature_extractors_ls,
                                                          batch_size=batch_size,
                                                          mimic_img=True,
                                                          intensity_range='imagenet',
                                                          initial_const=sd,
                                                          learning_rate=lr,
                                                          max_iterations=max_step,
                                                          l_threshold=th,
                                                          verbose=1 if debug else 0,
                                                          maximize=False,
                                                          keep_final=False,
                                                          image_shape=(224, 224, 3))

                protected_images = generate_cloak_images(self.protector, original_images,
                                                         target_emb=target_embedding)

                faces.cloaked_cropped_faces = protected_images

                final_images = faces.merge_faces(reverse_process_cloaked(protected_images),
                                                 reverse_process_cloaked(original_images))

        for p_img, path in zip(final_images, image_paths):
            file_name = "{}_{}_cloaked.{}".format(".".join(path.split(".")[:-1]), mode, format)
            dump_image(p_img, file_name, format=format)

        print("Done!")
        return 1


def main(*argv):
    if not argv:
        argv = list(sys.argv)

    try:
        import signal
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    except Exception as e:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', '-d', type=str,
                        help='the directory that contains images to run protection', default='imgs/')

    parser.add_argument('--gpu', '-g', type=str,
                        help='the GPU id when using GPU for optimization', default='0')

    parser.add_argument('--mode', '-m', type=str,
                        help='cloak generation mode, select from min, low, mid, high. The higher the mode is, the more perturbation added and stronger protection',
                        default='min')

    parser.add_argument('--feature-extractor', type=str,
                        help="name of the feature extractor used for optimization, currently only support high_extract",
                        default="high_extract")

    parser.add_argument('--th', help='only relevant with mode=custom, DSSIM threshold for perturbation', type=float,
                        default=0.01)
    parser.add_argument('--max-step', help='only relevant with mode=custom, number of steps for optimization', type=int,
                        default=1000)
    parser.add_argument('--sd', type=int, help='only relevant with mode=custom, penalty number, read more in the paper',
                        default=1e9)
    parser.add_argument('--lr', type=float, help='only relevant with mode=custom, learning rate', default=2)

    parser.add_argument('--batch-size', help="number of images to run optimization together", type=int, default=1)
    parser.add_argument('--separate_target', help="whether select separate targets for each faces in the directory",
                        action='store_true')
    parser.add_argument('--no-align', help="whether to detect and crop faces",
                        action='store_true')
    parser.add_argument('--debug', help="turn on debug and copy/paste the stdout when reporting an issue on github",
                        action='store_true')
    parser.add_argument('--format', type=str, choices=('png', 'jpg', 'jpeg'),
                        help="format of the output image",
                        default="png")

    args = parser.parse_args(argv[1:])

    if args.format == 'jpg':
        args.format = 'jpeg'

    image_paths = glob.glob(os.path.join(args.directory, "*"))
    image_paths = [path for path in image_paths if "_cloaked" not in path.split("/")[-1]]

    protector = Fawkes(args.feature_extractor, args.gpu, args.batch_size)
    if args.mode == 'all':
        for mode in ['min', 'low', 'mid', 'high']:
            protector.run_protection(image_paths, mode=mode, th=args.th, sd=args.sd, lr=args.lr,
                                     max_step=args.max_step,
                                     batch_size=args.batch_size, format=args.format,
                                     separate_target=args.separate_target, debug=args.debug, no_align=args.no_align)
    else:
        protector.run_protection(image_paths, mode=args.mode, th=args.th, sd=args.sd, lr=args.lr,
                                 max_step=args.max_step,
                                 batch_size=args.batch_size, format=args.format,
                                 separate_target=args.separate_target, debug=args.debug, no_align=args.no_align)


if __name__ == '__main__':
    main(*sys.argv)
