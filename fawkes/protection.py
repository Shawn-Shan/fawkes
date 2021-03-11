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

logging.getLogger('tensorflow').setLevel(logging.ERROR)
os.environ["KMP_AFFINITY"] = "noverbose"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(3)

import numpy as np
from fawkes.differentiator import FawkesMaskGeneration
from fawkes.utils import init_gpu, dump_image, reverse_process_cloaked, \
    Faces, filter_image_paths, load_extractor

from fawkes.align_face import aligner


def generate_cloak_images(protector, image_X, target_emb=None):
    cloaked_image_X = protector.compute(image_X, target_emb)
    return cloaked_image_X


IMG_SIZE = 112
PREPROCESS = 'raw'


class Fawkes(object):
    def __init__(self, feature_extractor, gpu, batch_size, mode="low"):

        self.feature_extractor = feature_extractor
        self.gpu = gpu
        self.batch_size = batch_size
        self.mode = mode
        th, max_step, lr, extractors = self.mode2param(self.mode)
        self.th = th
        self.lr = lr
        self.max_step = max_step

        init_gpu(gpu)

        # model_dir = os.path.join(os.path.expanduser('~'), '.fawkes')
        # if not os.path.exists(os.path.join(model_dir)):
        #     os.makedirs(model_dir, exist_ok=True)

        self.aligner = aligner()

        self.protector = None
        self.protector_param = None
        self.feature_extractors_ls = [load_extractor(name) for name in extractors]

    def mode2param(self, mode):
        if mode == 'low':
            th = 0.005
            max_step = 25
            lr = 25
            extractors = ["extractor_2"]

        elif mode == 'mid':
            th = 0.01
            max_step = 50
            lr = 20
            extractors = ["extractor_0", "extractor_2"]

        elif mode == 'high':
            th = 0.015
            max_step = 100
            lr = 15
            extractors = ["extractor_0", "extractor_2"]

        else:
            raise Exception("mode must be one of 'min', 'low', 'mid', 'high'")
        return th, max_step, lr, extractors

    def run_protection(self, image_paths, th=0.04, sd=1e9, lr=10, max_step=500, batch_size=1, format='png',
                       separate_target=True, debug=False, no_align=False, exp="", maximize=True,
                       save_last_on_failed=True):

        current_param = "-".join([str(x) for x in [self.th, sd, self.lr, self.max_step, batch_size, format,
                                                   separate_target, debug]])

        image_paths, loaded_images = filter_image_paths(image_paths)

        if not image_paths:
            print("No images in the directory")
            return 3

        faces = Faces(image_paths, loaded_images, self.aligner, verbose=1, no_align=no_align)
        original_images = faces.cropped_faces

        if len(original_images) == 0:
            print("No face detected. ")
            return 2
        original_images = np.array(original_images)

        if current_param != self.protector_param:
            self.protector_param = current_param
            if self.protector is not None:
                del self.protector
            if batch_size == -1:
                batch_size = len(original_images)
            self.protector = FawkesMaskGeneration(self.feature_extractors_ls,
                                                  batch_size=batch_size,
                                                  mimic_img=True,
                                                  intensity_range=PREPROCESS,
                                                  initial_const=sd,
                                                  learning_rate=self.lr,
                                                  max_iterations=self.max_step,
                                                  l_threshold=self.th,
                                                  verbose=0,
                                                  maximize=maximize,
                                                  keep_final=False,
                                                  image_shape=(IMG_SIZE, IMG_SIZE, 3),
                                                  loss_method='features',
                                                  tanh_process=True,
                                                  save_last_on_failed=save_last_on_failed,
                                                  )
        protected_images = generate_cloak_images(self.protector, original_images)
        faces.cloaked_cropped_faces = protected_images

        final_images, images_without_face = faces.merge_faces(
            reverse_process_cloaked(protected_images, preprocess=PREPROCESS),
            reverse_process_cloaked(original_images, preprocess=PREPROCESS))

        for i in range(len(final_images)):
            if i in images_without_face:
                continue
            p_img = final_images[i]
            path = image_paths[i]
            file_name = "{}_cloaked.{}".format(".".join(path.split(".")[:-1]), format)
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
                        help='cloak generation mode, select from min, low, mid, high. The higher the mode is, '
                             'the more perturbation added and stronger protection',
                        default='low')
    parser.add_argument('--feature-extractor', type=str,
                        help="name of the feature extractor used for optimization",
                        default="arcface_extractor_0")
    parser.add_argument('--th', help='only relevant with mode=custom, DSSIM threshold for perturbation', type=float,
                        default=0.01)
    parser.add_argument('--max-step', help='only relevant with mode=custom, number of steps for optimization', type=int,
                        default=1000)
    parser.add_argument('--sd', type=int, help='only relevant with mode=custom, penalty number, read more in the paper',
                        default=1e6)
    parser.add_argument('--lr', type=float, help='only relevant with mode=custom, learning rate', default=2)
    parser.add_argument('--batch-size', help="number of images to run optimization together", type=int, default=1)
    parser.add_argument('--separate_target', help="whether select separate targets for each faces in the directory",
                        action='store_true')
    parser.add_argument('--no-align', help="whether to detect and crop faces",
                        action='store_true')
    parser.add_argument('--debug', help="turn on debug and copy/paste the stdout when reporting an issue on github",
                        action='store_true')
    parser.add_argument('--format', type=str,
                        help="format of the output image",
                        default="png")

    args = parser.parse_args(argv[1:])

    assert args.format in ['png', 'jpg', 'jpeg']
    if args.format == 'jpg':
        args.format = 'jpeg'

    image_paths = glob.glob(os.path.join(args.directory, "*"))
    image_paths = [path for path in image_paths if "_cloaked" not in path.split("/")[-1]]

    protector = Fawkes(args.feature_extractor, args.gpu, args.batch_size, mode=args.mode)

    protector.run_protection(image_paths, th=args.th, sd=args.sd, lr=args.lr,
                             max_step=args.max_step,
                             batch_size=args.batch_size, format=args.format,
                             separate_target=args.separate_target, debug=args.debug, no_align=args.no_align)


if __name__ == '__main__':
    main(*sys.argv)
