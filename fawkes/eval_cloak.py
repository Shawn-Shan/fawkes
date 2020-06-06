import sys

sys.path.append("/home/shansixioing/tools/")
sys.path.append("/home/shansixioing/cloak/")

import argparse
from tensorflow import set_random_seed
from utils import init_gpu, load_extractor, load_victim_model, dump_dictionary_as_json
import os
import numpy as np
import random
import pickle
import re
from keras.preprocessing import image
from keras.utils import to_categorical
from keras.applications.vgg16 import preprocess_input

# import locale
#
# loc = locale.getlocale()
# locale.setlocale(locale.LC_ALL, loc)


def select_samples(data_dir):
    all_data_path = []
    for cls in os.listdir(data_dir):
        cls_dir = os.path.join(data_dir, cls)
        for data_path in os.listdir(cls_dir):
            all_data_path.append(os.path.join(cls_dir, data_path))

    return all_data_path


def generator_wrap(cloak_data, n_classes, test=False, validation_split=0.1):
    if test:
        all_data_path = select_samples(cloak_data.test_data_dir)
    else:
        all_data_path = select_samples(cloak_data.train_data_dir)
    split = int(len(cloak_data.cloaked_protect_train_X) * (1 - validation_split))
    cloaked_train_X = cloak_data.cloaked_protect_train_X[:split]
    np.random.seed(12345)

    # all_vals = list(cloak_data.path2idx.items())

    while True:
        batch_X = []
        batch_Y = []
        cur_batch_path = np.random.choice(all_data_path, args.batch_size)
        for p in cur_batch_path:
            # p = p.encode("utf-8").decode("ascii", 'ignore')
            cur_y = cloak_data.path2idx[p]
            # protect class and sybil class do not need to appear in test dataset
            if test and (re.search(cloak_data.protect_class, p)):
                continue
            # protect class images in train dataset
            elif p in cloak_data.protect_class_path:
                cur_x = random.choice(cloaked_train_X)
            else:
                im = image.load_img(p, target_size=cloak_data.img_shape)
                im = image.img_to_array(im)
                cur_x = preprocess_input(im)
            batch_X.append(cur_x)
            batch_Y.append(cur_y)

        batch_X = np.array(batch_X)
        batch_Y = to_categorical(np.array(batch_Y), num_classes=n_classes)

        yield batch_X, batch_Y


def eval_uncloaked_test_data(cloak_data, n_classes):
    original_label = cloak_data.path2idx[list(cloak_data.protect_class_path)[0]]
    protect_test_X = cloak_data.protect_test_X
    original_Y = [original_label] * len(protect_test_X)
    original_Y = to_categorical(original_Y, n_classes)
    return protect_test_X, original_Y


def eval_cloaked_test_data(cloak_data, n_classes, validation_split=0.1):
    split = int(len(cloak_data.cloaked_protect_train_X) * (1 - validation_split))
    cloaked_test_X = cloak_data.cloaked_protect_train_X[split:]
    original_label = cloak_data.path2idx[list(cloak_data.protect_class_path)[0]]
    original_Y = [original_label] * len(cloaked_test_X)
    original_Y = to_categorical(original_Y, n_classes)
    return cloaked_test_X, original_Y


def main():
    init_gpu(args.gpu)

    if args.dataset == 'pubfig':
        N_CLASSES = 65
        CLOAK_DIR = args.cloak_data
    elif args.dataset == 'scrub':
        N_CLASSES = 530
        CLOAK_DIR = args.cloak_data
    else:
        raise ValueError

    CLOAK_DIR = os.path.join("../results", CLOAK_DIR)
    RES = pickle.load(open(os.path.join(CLOAK_DIR, "cloak_data.p"), 'rb'))

    print("Build attacker's model")
    cloak_data = RES['cloak_data']
    EVAL_RES = {}
    train_generator = generator_wrap(cloak_data, n_classes=N_CLASSES,
                                     validation_split=args.validation_split)
    test_generator = generator_wrap(cloak_data, test=True, n_classes=N_CLASSES,
                                    validation_split=args.validation_split)

    EVAL_RES['transfer_model'] = args.transfer_model

    base_model = load_extractor(args.transfer_model)
    model = load_victim_model(teacher_model=base_model, number_classes=N_CLASSES)

    original_X, original_Y = eval_uncloaked_test_data(cloak_data, N_CLASSES)
    cloaked_test_X, cloaked_test_Y = eval_cloaked_test_data(cloak_data, N_CLASSES,
                                                            validation_split=args.validation_split)

    try:
        model.fit_generator(train_generator, steps_per_epoch=cloak_data.number_samples // 32,
                            validation_data=(original_X, original_Y), epochs=args.n_epochs, verbose=2,
                            use_multiprocessing=False, workers=1)
    except KeyboardInterrupt:
        pass

    _, acc_original = model.evaluate(original_X, original_Y, verbose=0)
    print("Accuracy on uncloaked/original images TEST: {:.4f}".format(acc_original))
    EVAL_RES['acc_original'] = acc_original

    _, other_acc = model.evaluate_generator(test_generator, verbose=0, steps=50)
    print("Accuracy on other classes {:.4f}".format(other_acc))
    EVAL_RES['other_acc'] = other_acc
    dump_dictionary_as_json(EVAL_RES, os.path.join(CLOAK_DIR, "eval_seed{}.json".format(args.seed_idx)))


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type=str,
                        help='GPU id', default='0')
    parser.add_argument('--dataset', type=str,
                        help='name of dataset', default='scrub')
    parser.add_argument('--cloak_data', type=str,
                        help='name of the cloak result directory',
                        default='scrub_webface_dense_robust_extract_protectPatrick_Dempsey')
    parser.add_argument('--transfer_model', type=str,
                        help='the feature extractor used for tracker model training. It can be the same or not same as the user\'s', default='vggface2_inception_extract')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--validation_split', type=float, default=0.1)
    parser.add_argument('--n_epochs', type=int, default=5)
    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main()
