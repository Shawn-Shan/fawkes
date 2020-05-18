import sys

sys.path.append("/home/shansixioing/tools/")
sys.path.append("/home/shansixioing/cloak/")

import argparse
from tensorflow import set_random_seed
from .differentiator import FawkesMaskGeneration
import os
import numpy as np
import random
import pickle
from .utils import load_extractor, CloakData, init_gpu

random.seed(12243)
np.random.seed(122412)
set_random_seed(12242)

SYBIL_ONLY = False

NUM_IMG_PROTECTED = 20  # Number of images used to optimize the target class
BATCH_SIZE = 20

MODEL_IDX = {
    'vggface1_inception': "0",
    'vggface1_dense': "1",
    "vggface2_inception": "2",
    "vggface2_dense": "3",
    "webface_dense": "4",
    "webface_inception": "5",
}

IDX2MODEL = {v: k for k, v in MODEL_IDX.items()}

IMG_SHAPE = [224, 224, 3]

GLOBAL_MASK = 0

MAXIMIZE = False
MAX_ITER = 500
INITIAL_CONST = 1e6
LR = 0.1


def diff_protected_data(sess, feature_extractors_ls, image_X, number_protect, target_X=None, sybil=False, th=0.01):
    image_X = image_X[:number_protect]

    differentiator = FawkesMaskGeneration(sess, feature_extractors_ls,
                                          batch_size=BATCH_SIZE,
                                          mimic_img=True,
                                          intensity_range='imagenet',
                                          initial_const=INITIAL_CONST,
                                          learning_rate=LR,
                                          max_iterations=MAX_ITER,
                                          l_threshold=th,
                                          verbose=1, maximize=False, keep_final=False, image_shape=image_X.shape[1:])

    if len(target_X) < len(image_X):
        target_X = np.concatenate([target_X, target_X, target_X, target_X, target_X])
    target_X = target_X[:len(image_X)]
    cloaked_image_X = differentiator.attack(image_X, target_X)
    return cloaked_image_X


def save_results(RES, path):
    pickle.dump(RES, open(path, "wb"))


def perform_defense():
    RES = {}
    sess = init_gpu(args.gpu)
    DSSIM_THRESHOLD = args.th

    FEATURE_EXTRACTORS = [IDX2MODEL[args.model_idx]]
    MODEL_HASH = "".join(MODEL_IDX[m] for m in FEATURE_EXTRACTORS)

    RES_DIR = '../results/'
    RES['num_img_protected'] = NUM_IMG_PROTECTED
    RES['extractors'] = FEATURE_EXTRACTORS
    num_protect = NUM_IMG_PROTECTED

    print(FEATURE_EXTRACTORS)
    feature_extractors_ls = [load_extractor(name) for name in FEATURE_EXTRACTORS]

    protect_class = args.protect_class

    cloak_data = CloakData(args.dataset, target_selection_tries=1, protect_class=protect_class)
    print("Protect Class: ", cloak_data.protect_class)

    if "robust" in FEATURE_EXTRACTORS[0]:
        non_robust = MODEL_IDX["_".join(FEATURE_EXTRACTORS[0].split("_")[:2])]
        if args.dataset == 'pubfig':
            CLOAK_DIR = 'pubfig_tm{}_tgt57_r1.0_th0.01'.format(non_robust)
            CLOAK_DIR = os.path.join(RES_DIR, CLOAK_DIR)
            RES = pickle.load(open(os.path.join(CLOAK_DIR, "cloak_data.p"), 'rb'))
            cloak_data = RES['cloak_data']
        elif args.dataset == 'scrub':
            CLOAK_DIR = 'scrub_tm{}_tgtPatrick_Dempsey_r1.0_th0.01'.format(non_robust)
            CLOAK_DIR = os.path.join(RES_DIR, CLOAK_DIR)
            RES = pickle.load(open(os.path.join(CLOAK_DIR, "cloak_data.p"), 'rb'))
            cloak_data = RES['cloak_data']
    else:
        cloak_data.target_path, cloak_data.target_data = cloak_data.select_target_label(feature_extractors_ls,
                                                                                        FEATURE_EXTRACTORS)

    RES_FILE_NAME = "{}_tm{}_tgt{}_r{}_th{}".format(args.dataset, MODEL_HASH, cloak_data.protect_class, RATIO,
                                                    DSSIM_THRESHOLD)
    RES_FILE_NAME = os.path.join(RES_DIR, RES_FILE_NAME)
    os.makedirs(RES_FILE_NAME, exist_ok=True)

    print("Protect Current Label Data...")

    cloak_image_X = diff_protected_data(sess, feature_extractors_ls, cloak_data.protect_train_X,
                                        number_protect=num_protect,
                                        target_X=cloak_data.target_data, sybil=False, th=DSSIM_THRESHOLD)

    cloak_data.cloaked_protect_train_X = cloak_image_X
    RES['cloak_data'] = cloak_data
    save_results(RES, os.path.join(RES_FILE_NAME, 'cloak_data.p'))


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str,
                        help='GPU id', default='0')
    parser.add_argument('--dataset', type=str,
                        help='name of dataset', default='pubfig')
    parser.add_argument('--model_idx', type=str,
                        help='teacher model index', default="3")
    parser.add_argument('--th', type=float, default=0.01)
    parser.add_argument('--protect_class', type=str, default=None)

    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    perform_defense()
