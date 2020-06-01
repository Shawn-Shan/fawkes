import argparse
import os
import pickle
import random
import sys

import numpy as np
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from utils import load_extractor, get_dataset_path


def load_sample_dir(path, sample=10):
    x_ls = []
    image_paths = list(os.listdir(path))
    random.shuffle(image_paths)
    for i, file in enumerate(image_paths):
        if i > sample:
            break
        cur_path = os.path.join(path, file)
        im = image.load_img(cur_path, target_size=(224, 224))
        im = image.img_to_array(im)
        x_ls.append(im)
    raw_x = np.array(x_ls)
    return preprocess_input(raw_x)


def normalize(x):
    return x / np.linalg.norm(x)


def main():
    extractor = load_extractor(args.feature_extractor)
    path2emb = {}
    for target_dataset in args.candidate_datasets:
        target_dataset_path, _, _, _ = get_dataset_path(target_dataset)
        for target_class in os.listdir(target_dataset_path):
            target_class_path = os.path.join(target_dataset_path, target_class)
            target_X = load_sample_dir(target_class_path)
            cur_feature = extractor.predict(target_X)
            cur_feature = np.mean(cur_feature, axis=0)
            path2emb[target_class_path] = cur_feature

    for k, v in path2emb.items():
        path2emb[k] = normalize(v)

    pickle.dump(path2emb, open("../feature_extractors/embeddings/{}_emb_norm.p".format(args.feature_extractor), "wb"))


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str,
                        help='GPU id', default='0')
    parser.add_argument('--candidate-datasets', nargs='+',
                        help='path candidate datasets')
    parser.add_argument('--feature-extractor', type=str,
                        help="name of the feature extractor used for optimization",
                        default="webface_dense_robust_extract")
    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main()
