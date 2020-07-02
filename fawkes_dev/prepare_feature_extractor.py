import argparse
import glob
import os
import pickle
import random
import sys

import numpy as np
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
sys.path.append("../fawkes")
# from utils import load_extractor
import keras

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
    extractor = keras.models.load_model(args.feature_extractor)

    path2emb = {}
    model_dir = os.path.join(os.path.expanduser('~'), '.fawkes')
    for path in glob.glob(os.path.join(model_dir, "target_data/*")):
        print(path)
        idx = int(path.split("/")[-1])
        cur_image_paths = glob.glob(os.path.join(path, "*"))
        imgs = np.array([image.img_to_array(image.load_img(p, target_size=(224, 224))) for p in cur_image_paths])
        imgs = preprocess_input(imgs)

        cur_feature = extractor.predict(imgs)
        cur_feature = np.mean(cur_feature, axis=0)
        path2emb[idx] = cur_feature

    model_path = os.path.join(model_dir, "{}_extract.h5".format(args.feature_extractor_name))
    emb_path = os.path.join(model_dir, "{}_emb.p".format(args.feature_extractor_name))
    extractor.save(model_path)
    pickle.dump(path2emb, open(emb_path, "wb"))



def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str,
                        help='GPU id', default='0')
    parser.add_argument('--candidate-datasets', nargs='+',
                        help='path candidate datasets')
    parser.add_argument('--feature-extractor', type=str,
                        help="path of the feature extractor used for optimization",
                        default="/home/shansixioing/fawkes/feature_extractors/high2_extract.h5")
    parser.add_argument('--feature-extractor-name', type=str,
                        help="name of the feature extractor used for optimization",
                        default="high2")

    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main()
