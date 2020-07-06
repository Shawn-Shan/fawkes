# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import argparse
import glob
import os
import random
import sys
import time

import numpy as np

from .differentiator import FawkesMaskGeneration
from .utils import load_extractor, init_gpu, select_target_label, dump_image, reverse_process_cloaked, \
    Faces

random.seed(12243)
np.random.seed(122412)


def generate_cloak_images(sess, feature_extractors, image_X, target_emb=None, th=0.01, faces=None, sd=1e9, lr=2,
                          max_step=500, batch_size=1):
    batch_size = batch_size if len(image_X) > batch_size else len(image_X)

    differentiator = FawkesMaskGeneration(sess, feature_extractors,
                                          batch_size=batch_size,
                                          mimic_img=True,
                                          intensity_range='imagenet',
                                          initial_const=sd,
                                          learning_rate=lr,
                                          max_iterations=max_step,
                                          l_threshold=th,
                                          verbose=1, maximize=False, keep_final=False, image_shape=image_X.shape[1:],
                                          faces=faces)

    cloaked_image_X = differentiator.attack(image_X, target_emb)
    return cloaked_image_X


def check_imgs(imgs):
    if np.max(imgs) <= 1 and np.min(imgs) >= 0:
        imgs = imgs * 255.0
    elif np.max(imgs) <= 255 and np.min(imgs) >= 0:
        pass
    else:
        raise Exception("Image values ")
    return imgs


def main(*argv):
    start_time = time.time()
    if not argv:
        argv = list(sys.argv)

    try:
        import signal
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    except Exception as e:
        pass

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
    parser.add_argument('--max-step', type=int, default=500)
    parser.add_argument('--sd', type=int, default=1e9)
    parser.add_argument('--lr', type=float, default=2)

    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--separate_target', action='store_true')

    parser.add_argument('--format', type=str,
                        help="final image format",
                        default="png")
    args = parser.parse_args(argv[1:])

    if args.mode == 'low':
        args.feature_extractor = "high_extract"
        args.th = 0.003
        args.max_step = 100
        args.lr = 15
    elif args.mode == 'mid':
        args.feature_extractor = "high_extract"
        args.th = 0.005
        args.max_step = 100
        args.lr = 15
    elif args.mode == 'high':
        args.feature_extractor = "high_extract"
        args.th = 0.007
        args.max_step = 100
        args.lr = 10
    elif args.mode == 'ultra':
        args.feature_extractor = "high_extract"
        args.th = 0.01
        args.max_step = 1000
        args.lr = 5
    elif args.mode == 'custom':
        pass
    else:
        raise Exception("mode must be one of 'low', 'mid', 'high', 'ultra', 'custom'")

    assert args.format in ['png', 'jpg', 'jpeg']
    if args.format == 'jpg':
        args.format = 'jpeg'

    sess = init_gpu(args.gpu)
    fs_names = [args.feature_extractor]
    feature_extractors_ls = [load_extractor(name) for name in fs_names]

    image_paths = glob.glob(os.path.join(args.directory, "*"))
    image_paths = [path for path in image_paths if "_cloaked" not in path.split("/")[-1]]
    if not image_paths:
        print("No images in the directory")
        exit(1)

    faces = Faces(image_paths, sess, verbose=1)

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
                                             target_emb=target_embedding, th=args.th, faces=faces, sd=args.sd,
                                             lr=args.lr, max_step=args.max_step, batch_size=args.batch_size)

    faces.cloaked_cropped_faces = protected_images

    cloak_perturbation = reverse_process_cloaked(protected_images) - reverse_process_cloaked(orginal_images)
    final_images = faces.merge_faces(cloak_perturbation)

    for p_img, cloaked_img, path in zip(final_images, protected_images, image_paths):
        file_name = "{}_{}_cloaked.{}".format(".".join(path.split(".")[:-1]), args.mode, args.format)
        dump_image(p_img, file_name, format=args.format)

    elapsed_time = time.time() - start_time
    print('attack cost %f s' % (elapsed_time))


if __name__ == '__main__':
    main(*sys.argv)
