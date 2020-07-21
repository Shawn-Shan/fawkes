import argparse
import glob
import os
import random
import sys

import keras
import numpy as np

random.seed(1000)
from fawkes.utils import init_gpu, load_extractor, load_victim_model, get_file, preprocess, Faces, filter_image_paths
from keras.preprocessing import image
from keras.utils import to_categorical
from fawkes.align_face import aligner


def select_samples(data_dir):
    all_data_path = []
    for cls in os.listdir(data_dir):
        cls_dir = os.path.join(data_dir, cls)
        for data_path in os.listdir(cls_dir):
            all_data_path.append(os.path.join(cls_dir, data_path))
    return all_data_path


class DataGenerator(object):
    def __init__(self, original_images, protect_images):
        l = int(len(original_images) * 0.7)
        self.original_images_test = original_images[l:]
        self.protect_images_train = protect_images[:l]

        other_classes = range(0, 20946)
        selected_classes = random.sample(other_classes, args.num_other_classes)
        print("Downloading additional data...")
        model_dir = os.path.join(os.path.expanduser('~'), '.fawkes')

        self.id2label = {-1: 0}
        self.id2path = {}
        self.id2pathtest = {}
        idx = 1
        for target_data_id in selected_classes:
            image_dir = os.path.join(model_dir, "target_data/{}".format(target_data_id))
            os.makedirs(os.path.join(model_dir, "target_data"), exist_ok=True)
            os.makedirs(image_dir, exist_ok=True)
            self.id2label[target_data_id] = idx
            idx += 1
            for i in range(10):
                if os.path.exists(os.path.join(model_dir, "target_data/{}/{}.jpg".format(target_data_id, i))):
                    continue
                try:
                    get_file("{}.jpg".format(i),
                             "http://sandlab.cs.uchicago.edu/fawkes/files/target_data/{}/{}.jpg".format(target_data_id,
                                                                                                        i),
                             cache_dir=model_dir, cache_subdir='target_data/{}/'.format(target_data_id))
                except Exception:
                    print("error getting http://sandlab.cs.uchicago.edu/fawkes/files/target_data/{}/{}.jpg".format(
                        target_data_id,
                        i))
                    pass

            all_pathes = glob.glob(os.path.join(model_dir, 'target_data/{}/*.jpg'.format(target_data_id)))
            test_path = random.sample(all_pathes, 2)
            train_path = [p for p in all_pathes if p not in test_path]
            self.id2path[target_data_id] = train_path
            self.id2pathtest[target_data_id] = test_path

        self.num_classes = 1 + len(self.id2path)

        np.random.seed(12345)

        self.all_id = selected_classes + [-1]

    def generate(self, test=False):
        while True:
            batch_X = []
            batch_Y = []
            cur_batch_path = np.random.choice(self.all_id, 32)
            for p in cur_batch_path:
                cur_y = self.id2label[p]
                if test and p == -1:
                    continue
                # protect class images in train dataset
                elif p == -1:
                    cur_x = random.choice(self.protect_images_train)
                else:
                    if test:
                        cur_path = random.choice(self.id2pathtest[p])
                    else:
                        cur_path = random.choice(self.id2path[p])
                    im = image.load_img(cur_path, target_size=(224, 224))
                    cur_x = image.img_to_array(im)

                cur_x = preprocess(cur_x, 'imagenet')
                batch_X.append(cur_x)
                batch_Y.append(cur_y)

            batch_X = np.array(batch_X)
            batch_Y = to_categorical(np.array(batch_Y), num_classes=self.num_classes)

            yield batch_X, batch_Y

    def test_original(self):
        original_y = to_categorical([0] * len(self.original_images_test), num_classes=self.num_classes)
        return self.original_images_test, original_y


class CallbackGenerator(keras.callbacks.Callback):
    def __init__(self, original_imgs, protect_imgs, original_y, original_protect_y, test_gen):
        self.original_imgs = original_imgs
        self.protect_imgs = protect_imgs

        self.original_y = original_y
        self.original_protect_y = original_protect_y
        self.test_gen = test_gen

    def on_epoch_end(self, epoch, logs=None):
        _, original_acc = self.model.evaluate(self.original_imgs, self.original_y, verbose=0)
        print("Epoch: {} - Protection Success Rate {:.4f}".format(epoch, 1 - original_acc))


def main():
    sess = init_gpu(args.gpu)
    ali = aligner(sess)
    print("Build attacker's model")
    image_paths = glob.glob(os.path.join(args.directory, "*"))
    cloak_file_name = "_cloaked"

    original_image_paths = sorted([path for path in image_paths if "cloaked" not in path.split("/")[-1]])
    original_image_paths, original_loaded_images = filter_image_paths(original_image_paths)

    protect_image_paths = sorted([path for path in image_paths if cloak_file_name in path.split("/")[-1]])
    protect_image_paths, protected_loaded_images = filter_image_paths(protect_image_paths)

    print("Find {} original image and {} cloaked images".format(len(original_image_paths), len(protect_image_paths)))

    original_faces = Faces(original_image_paths, original_loaded_images, ali, verbose=1, eval_local=True)
    original_faces = original_faces.cropped_faces
    cloaked_faces = Faces(protect_image_paths, protected_loaded_images, ali, verbose=1, eval_local=True)
    cloaked_faces = cloaked_faces.cropped_faces

    if len(original_faces) <= 10 or len(protect_image_paths) <= 10:
        raise Exception("Must have more than 10 protected images to run the evaluation")

    num_classes = args.num_other_classes + 1
    datagen = DataGenerator(original_faces, cloaked_faces)
    original_test_X, original_test_Y = datagen.test_original()
    print("{} Training Images | {} Testing Images".format(len(datagen.protect_images_train), len(original_test_X)))

    train_generator = datagen.generate()
    test_generator = datagen.generate(test=True)

    base_model = load_extractor(args.base_model)
    model = load_victim_model(teacher_model=base_model, number_classes=num_classes)
    cb = CallbackGenerator(original_imgs=original_test_X, protect_imgs=cloaked_faces, original_y=original_test_Y,
                           original_protect_y=None,
                           test_gen=test_generator)

    model.fit_generator(train_generator, steps_per_epoch=num_classes * 10 // 32,
                        epochs=args.n_epochs,
                        verbose=1,
                        callbacks=[cb]
                        )

    _, acc_original = model.evaluate(original_test_X, original_test_Y, verbose=0)
    print("Protection Success Rate: {:.4f}".format(1 - acc_original))


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type=str,
                        help='GPU id', default='0')

    parser.add_argument('--dataset', type=str,
                        help='name of dataset', default='scrub')
    parser.add_argument('--num_other_classes', type=int,
                        help='name of dataset', default=500)

    parser.add_argument('--directory', '-d', type=str,
                        help='name of the cloak result directory', required=True)
    parser.add_argument('--base_model', type=str,
                        help='the feature extractor used for tracker model training. ', default='low_extract')
    parser.add_argument('--n_epochs', type=int, default=5)
    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main()
