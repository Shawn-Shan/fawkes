import argparse
import os
import pickle
import random
import sys

import numpy as np
from differentiator import FawkesMaskGeneration
from tensorflow import set_random_seed
from utils import load_extractor, CloakData, init_gpu

random.seed(12243)
np.random.seed(122412)
set_random_seed(12242)

NUM_IMG_PROTECTED = 400  # Number of images used to optimize the target class
BATCH_SIZE = 32

MAX_ITER = 1000


def diff_protected_data(sess, feature_extractors_ls, image_X, number_protect, target_X=None, th=0.01):
    image_X = image_X[:number_protect]
    differentiator = FawkesMaskGeneration(sess, feature_extractors_ls,
                                          batch_size=BATCH_SIZE,
                                          mimic_img=True,
                                          intensity_range='imagenet',
                                          initial_const=args.sd,
                                          learning_rate=args.lr,
                                          max_iterations=MAX_ITER,
                                          l_threshold=th,
                                          verbose=1, maximize=False, keep_final=False, image_shape=image_X.shape[1:])

    if len(target_X) < len(image_X):
        target_X = np.concatenate([target_X, target_X, target_X])
    target_X = target_X[:len(image_X)]
    cloaked_image_X = differentiator.attack(image_X, target_X)
    return cloaked_image_X


def perform_defense():
    RES = {}
    sess = init_gpu(args.gpu)

    FEATURE_EXTRACTORS = [args.feature_extractor]
    RES_DIR = '../results/'

    RES['num_img_protected'] = NUM_IMG_PROTECTED
    RES['extractors'] = FEATURE_EXTRACTORS
    num_protect = NUM_IMG_PROTECTED

    print("Loading {} for optimization".format(args.feature_extractor))
    feature_extractors_ls = [load_extractor(name) for name in FEATURE_EXTRACTORS]
    protect_class = args.protect_class

    cloak_data = CloakData(args.dataset, protect_class=protect_class)
    RES_FILE_NAME = "{}_{}_protect{}".format(args.dataset, args.feature_extractor, cloak_data.protect_class)
    RES_FILE_NAME = os.path.join(RES_DIR, RES_FILE_NAME)
    print("Protect Class: ", cloak_data.protect_class)

    cloak_data.target_path, cloak_data.target_data = cloak_data.select_target_label(feature_extractors_ls,
                                                                                    FEATURE_EXTRACTORS)

    os.makedirs(RES_DIR, exist_ok=True)
    os.makedirs(RES_FILE_NAME, exist_ok=True)

    cloak_image_X = diff_protected_data(sess, feature_extractors_ls, cloak_data.protect_train_X,
                                        number_protect=num_protect,
                                        target_X=cloak_data.target_data, th=args.th)

    cloak_data.cloaked_protect_train_X = cloak_image_X
    RES['cloak_data'] = cloak_data
    pickle.dump(RES, open(os.path.join(RES_FILE_NAME, 'cloak_data.p'), "wb"))


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str,
                        help='GPU id', default='0')
    parser.add_argument('--dataset', type=str,
                        help='name of dataset', default='scrub')
    parser.add_argument('--feature-extractor', type=str,
                        help="name of the feature extractor used for optimization",
                        default="webface_dense_robust_extract")
    parser.add_argument('--th', type=float, default=0.007)
    parser.add_argument('--sd', type=int, default=1e5)
    parser.add_argument('--protect_class', type=str, default=None)
    parser.add_argument('--lr', type=float, default=0.1)

    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    perform_defense()
