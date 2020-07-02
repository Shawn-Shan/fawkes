import argparse
import os
import sys

import numpy as np

sys.path.append("/home/shansixioing/fawkes/fawkes")
from utils import extract_faces, get_dataset_path, init_gpu, load_extractor, load_victim_model

import random
import glob
from keras.preprocessing import image
from keras.utils import to_categorical
from keras.applications.vgg16 import preprocess_input


def select_samples(data_dir):
    all_data_path = []
    for cls in os.listdir(data_dir):
        cls_dir = os.path.join(data_dir, cls)
        for data_path in os.listdir(cls_dir):
            all_data_path.append(os.path.join(cls_dir, data_path))
    return all_data_path


def generator_wrap(protect_images, test=False, validation_split=0.1):
    train_data_dir, test_data_dir, num_classes, num_images = get_dataset_path(args.dataset)

    idx = 0
    path2class = {}
    path2imgs_list = {}

    for target_path in sorted(glob.glob(train_data_dir + "/*")):
        path2class[target_path] = idx
        path2imgs_list[target_path] = glob.glob(os.path.join(target_path, "*"))
        idx += 1
        if idx >= args.num_classes:
            break

    path2class["protected"] = idx

    np.random.seed(12345)

    while True:
        batch_X = []
        batch_Y = []
        cur_batch_path = np.random.choice(list(path2class.keys()), args.batch_size)
        for p in cur_batch_path:
            cur_y = path2class[p]
            if test and p == 'protected':
                continue
            # protect class images in train dataset
            elif p == 'protected':
                cur_x = random.choice(protect_images)
            else:
                cur_path = random.choice(path2imgs_list[p])
                im = image.load_img(cur_path, target_size=(224, 224))
                cur_x = image.img_to_array(im)

            cur_x = preprocess_input(cur_x)
            batch_X.append(cur_x)
            batch_Y.append(cur_y)

        batch_X = np.array(batch_X)
        batch_Y = to_categorical(np.array(batch_Y), num_classes=args.num_classes + 1)

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
    #
    # if args.dataset == 'pubfig':
    #     N_CLASSES = 65
    #     CLOAK_DIR = args.cloak_data
    # elif args.dataset == 'scrub':
    #     N_CLASSES = 530
    #     CLOAK_DIR = args.cloak_data
    # else:
    #     raise ValueError

    print("Build attacker's model")

    image_paths = glob.glob(os.path.join(args.directory, "*"))
    original_image_paths = sorted([path for path in image_paths if "_cloaked" not in path.split("/")[-1]])

    protect_image_paths = sorted([path for path in image_paths if "_cloaked" in path.split("/")[-1]])

    original_imgs = np.array([extract_faces(image.img_to_array(image.load_img(cur_path))) for cur_path in
                     original_image_paths[:150]])
    original_y = to_categorical([args.num_classes] * len(original_imgs), num_classes=args.num_classes + 1)

    protect_imgs = [extract_faces(image.img_to_array(image.load_img(cur_path))) for cur_path in
                    protect_image_paths]

    train_generator = generator_wrap(protect_imgs,
                                     validation_split=args.validation_split)
    test_generator = generator_wrap(protect_imgs, test=True,
                                    validation_split=args.validation_split)

    base_model = load_extractor(args.transfer_model)
    model = load_victim_model(teacher_model=base_model, number_classes=args.num_classes + 1)

    # cloaked_test_X, cloaked_test_Y = eval_cloaked_test_data(cloak_data, args.num_classes,
    #                                                         validation_split=args.validation_split)

    # try:
    train_data_dir, test_data_dir, num_classes, num_images = get_dataset_path(args.dataset)
    model.fit_generator(train_generator, steps_per_epoch=num_images // 32,
                        validation_data=(original_imgs, original_y),
                        epochs=args.n_epochs,
                        verbose=1,
                        use_multiprocessing=True, workers=5)
    # except KeyboardInterrupt:
    #     pass

    _, acc_original = model.evaluate(original_imgs, original_y, verbose=0)
    print("Accuracy on uncloaked/original images TEST: {:.4f}".format(acc_original))
    # EVAL_RES['acc_original'] = acc_original

    _, other_acc = model.evaluate_generator(test_generator, verbose=0, steps=50)
    print("Accuracy on other classes {:.4f}".format(other_acc))
    # EVAL_RES['other_acc'] = other_acc
    # dump_dictionary_as_json(EVAL_RES, os.path.join(CLOAK_DIR, "eval_seed{}.json".format(args.seed_idx)))


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type=str,
                        help='GPU id', default='0')

    parser.add_argument('--dataset', type=str,
                        help='name of dataset', default='scrub')
    parser.add_argument('--num_classes', type=int,
                        help='name of dataset', default=520)

    parser.add_argument('--directory', '-d', type=str,
                        help='name of the cloak result directory',
                        default='img/')

    parser.add_argument('--transfer_model', type=str,
                        help='the feature extractor used for tracker model training. ', default='low_extract')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--validation_split', type=float, default=0.1)
    parser.add_argument('--n_epochs', type=int, default=3)
    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main()
