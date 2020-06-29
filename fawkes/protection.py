import argparse
import glob
import os
import random
import sys

import numpy as np
from differentiator import FawkesMaskGeneration
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from skimage.transform import resize
from tensorflow import set_random_seed
from utils import load_extractor, init_gpu, select_target_label, dump_image, reverse_process_cloaked

random.seed(12243)
np.random.seed(122412)
set_random_seed(12242)

BATCH_SIZE = 32
MAX_ITER = 1000


def generate_cloak_images(sess, feature_extractors, image_X, target_X=None, th=0.01):
    batch_size = BATCH_SIZE if len(image_X) > BATCH_SIZE else len(image_X)
    differentiator = FawkesMaskGeneration(sess, feature_extractors,
                                          batch_size=batch_size,
                                          mimic_img=True,
                                          intensity_range='imagenet',
                                          initial_const=args.sd,
                                          learning_rate=args.lr,
                                          max_iterations=MAX_ITER,
                                          l_threshold=th,
                                          verbose=1, maximize=False, keep_final=False, image_shape=image_X.shape[1:])

    cloaked_image_X = differentiator.attack(image_X, target_X)
    return cloaked_image_X


def extract_faces(img):
    #  foo
    return preprocess_input(resize(img, (224, 224)))


def fawkes():
    assert os.path.exists(args.directory)
    assert os.path.isdir(args.directory)

    sess = init_gpu(args.gpu)

    print("Loading {} for optimization".format(args.feature_extractor))

    feature_extractors_ls = [load_extractor(args.feature_extractor)]

    image_paths = glob.glob(os.path.join(args.directory, "*"))
    image_paths = [path for path in image_paths if "_cloaked" not in path.split("/")[-1]]

    orginal_images = [extract_faces(image.img_to_array(image.load_img(cur_path))) for cur_path in
                      image_paths]

    orginal_images = np.array(orginal_images)

    if args.seperate_target:
        target_images = []
        for org_img in orginal_images:
            org_img = org_img.reshape([1] + list(org_img.shape))
            tar_img = select_target_label(org_img, feature_extractors_ls, [args.feature_extractor])
            target_images.append(tar_img)
        target_images = np.concatenate(target_images)
        # import pdb
        # pdb.set_trace()
    else:
        target_images = select_target_label(orginal_images, feature_extractors_ls, [args.feature_extractor])

    # file_name = args.directory.split("/")[-1]
    # os.makedirs(args.result_directory, exist_ok=True)
    # os.makedirs(os.path.join(args.result_directory, file_name), exist_ok=True)

    protected_images = generate_cloak_images(sess, feature_extractors_ls, orginal_images,
                                             target_X=target_images, th=args.th)

    for p_img, path in zip(protected_images, image_paths):
        p_img = reverse_process_cloaked(p_img)
        # img_type = path.split(".")[-1]
        file_name = "{}_cloaked.jpeg".format(".".join(path.split(".")[:-1]))
        dump_image(p_img, file_name, format="JPEG")


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str,
                        help='GPU id', default='0')
    parser.add_argument('--directory', type=str,
                        help='directory that contain images for cloaking', default='imgs/')

    parser.add_argument('--feature-extractor', type=str,
                        help="name of the feature extractor used for optimization",
                        default="webface_dense_robust_extract")

    parser.add_argument('--th', type=float, default=0.005)
    parser.add_argument('--sd', type=int, default=1e10)
    parser.add_argument('--protect_class', type=str, default=None)
    parser.add_argument('--lr', type=float, default=0.1)

    parser.add_argument('--result_directory', type=str, default="../results")
    parser.add_argument('--seperate_target', action='store_true')

    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    fawkes()
