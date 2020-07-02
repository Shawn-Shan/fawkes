import argparse
import glob
import os
import random
import sys

import numpy as np
from differentiator import FawkesMaskGeneration
from utils import load_extractor, init_gpu, select_target_label, dump_image, reverse_process_cloaked, \
    Faces

random.seed(12243)
np.random.seed(122412)

BATCH_SIZE = 10


def generate_cloak_images(sess, feature_extractors, image_X, target_emb=None, th=0.01, faces=None):
    batch_size = BATCH_SIZE if len(image_X) > BATCH_SIZE else len(image_X)

    differentiator = FawkesMaskGeneration(sess, feature_extractors,
                                          batch_size=batch_size,
                                          mimic_img=True,
                                          intensity_range='imagenet',
                                          initial_const=args.sd,
                                          learning_rate=args.lr,
                                          max_iterations=args.max_step,
                                          l_threshold=th,
                                          verbose=1, maximize=False, keep_final=False, image_shape=image_X.shape[1:],
                                          faces=faces)

    cloaked_image_X = differentiator.attack(image_X, target_emb)
    return cloaked_image_X


def get_mode_config(mode):
    if mode == 'low':
        args.feature_extractor = "low_extract"
        # args.th = 0.003
        args.th = 0.001
    elif mode == 'mid':
        args.feature_extractor = "mid_extract"
        args.th = 0.004
    elif mode == 'high':
        args.feature_extractor = "high_extract"
        args.th = 0.004
    elif mode == 'ultra':
        args.feature_extractor = "high_extract"
        args.th = 0.03
    elif mode == 'custom':
        pass
    else:
        raise Exception("mode must be one of 'low', 'mid', 'high', 'ultra', 'custom'")


def check_imgs(imgs):
    if np.max(imgs) <= 1 and np.min(imgs) >= 0:
        imgs = imgs * 255.0
    elif np.max(imgs) <= 255 and np.min(imgs) >= 0:
        pass
    else:
        raise Exception("Image values ")
    return imgs


def fawkes():
    assert args.format in ['png', 'jpg', 'jpeg']
    if args.format == 'jpg':
        args.format = 'jpeg'
    get_mode_config(args.mode)

    sess = init_gpu(args.gpu)
    # feature_extractors_ls = [load_extractor(args.feature_extractor)]
    # fs_names = ['mid_extract', 'high_extract']
    fs_names = [args.feature_extractor]
    feature_extractors_ls = [load_extractor(name) for name in fs_names]

    image_paths = glob.glob(os.path.join(args.directory, "*"))
    image_paths = [path for path in image_paths if "_cloaked" not in path.split("/")[-1]]

    faces = Faces(image_paths, sess)

    orginal_images = faces.cropped_faces
    orginal_images = np.array(orginal_images)

    if args.separate_target:
        target_embedding = []
        for org_img in orginal_images:
            org_img = org_img.reshape([1] + list(org_img.shape))
            tar_emb = select_target_label(org_img, feature_extractors_ls, fs_names)
            target_embedding.append(tar_emb)
        target_embedding = np.concatenate(target_embedding)
    else:
        target_embedding = select_target_label(orginal_images, feature_extractors_ls, fs_names)

    protected_images = generate_cloak_images(sess, feature_extractors_ls, orginal_images,
                                             target_emb=target_embedding, th=args.th, faces=faces)

    faces.cloaked_cropped_faces = protected_images

    cloak_perturbation = reverse_process_cloaked(protected_images) - reverse_process_cloaked(orginal_images)
    final_images = faces.merge_faces(cloak_perturbation)

    for p_img, cloaked_img, path in zip(final_images, protected_images, image_paths):
        file_name = "{}_{}_{}_{}_cloaked.{}".format(".".join(path.split(".")[:-1]), args.mode, args.th,
                                                     args.feature_extractor, args.format)
        dump_image(p_img, file_name, format=args.format)
        #
        # file_name = "{}_{}_{}_{}_cloaked_cropped.png".format(".".join(path.split(".")[:-1]), args.mode, args.th,
        #                                                      args.feature_extractor)
        # dump_image(reverse_process_cloaked(cloaked_img), file_name, format="png")


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', '-d', type=str,
                        help='directory that contain images for cloaking', default='imgs/')

    parser.add_argument('--gpu', type=str,
                        help='GPU id', default='0')

    parser.add_argument('--mode', type=str,
                        help='cloak generation mode', default='high')
    parser.add_argument('--feature-extractor', type=str,
                        help="name of the feature extractor used for optimization",
                        default="high_extract")

    parser.add_argument('--th', type=float, default=0.01)
    parser.add_argument('--max-step', type=int, default=200)
    parser.add_argument('--sd', type=int, default=1e9)
    parser.add_argument('--lr', type=float, default=10)

    parser.add_argument('--separate_target', action='store_true')

    parser.add_argument('--format', type=str,
                        help="final image format",
                        default="jpg")
    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    fawkes()
