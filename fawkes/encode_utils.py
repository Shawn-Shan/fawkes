import sys

sys.path.append("/home/shansixioing/tools/")
import gen_utils
import keras, os
from keras.preprocessing import image
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input
from keras.layers import Conv2D, MaxPooling2D, Dense, Activation, Layer
import keras.backend as K
import random, pickle
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import preprocess_input
from sklearn.metrics import pairwise_distances
from keras.utils import to_categorical


def load_dataset_deepid(full=False, num_classes=1283, preprocess='raw'):
    if not full:
        X_train, Y_train = gen_utils.load_h5py(["X_train", "Y_train"],
                                               "/mnt/data/sixiongshan/backdoor/data/deepid/deepid_data_training_0.h5")
    else:
        X_train_0, Y_train_0 = gen_utils.load_h5py(["X_train", "Y_train"],
                                                   "/mnt/data/sixiongshan/backdoor/data/deepid/deepid_data_training_0.h5")

        X_train_1, Y_train_1 = gen_utils.load_h5py(["X_train", "Y_train"],
                                                   "/mnt/data/sixiongshan/backdoor/data/deepid/deepid_data_training_1.h5")

        X_train_2, Y_train_2 = gen_utils.load_h5py(["X_train", "Y_train"],
                                                   "/mnt/data/sixiongshan/backdoor/data/deepid/deepid_data_training_2.h5")

        X_train_3, Y_train_3 = gen_utils.load_h5py(["X_train", "Y_train"],
                                                   "/mnt/data/sixiongshan/backdoor/data/deepid/deepid_data_training_3.h5")

        X_train = np.concatenate([X_train_0, X_train_1, X_train_2, X_train_3])
        Y_train = np.concatenate([Y_train_0, Y_train_1, Y_train_2, Y_train_3])

    X_test, Y_test = gen_utils.load_h5py(["X_test", "Y_test"],
                                         "/mnt/data/sixiongshan/backdoor/data/deepid/deepid_data_testing.h5")

    X_train = utils_keras.preprocess(X_train, preprocess)
    X_test = utils_keras.preprocess(X_test, preprocess)

    return X_train, Y_train, X_test, Y_test


def load_dataset(data_file):
    dataset = utils_keras.load_dataset(data_file)

    X_train = dataset['X_train']
    Y_train = dataset['Y_train']
    X_test = dataset['X_test']
    Y_test = dataset['Y_test']

    return X_train, Y_train, X_test, Y_test


def load_extractor(name, all_layers=False):
    if name is None:
        return
    m = keras.models.load_model("/home/shansixioing/cloak/models/extractors/{}_extract.h5".format(name))
    if all_layers:
        if name == 'vggface1':
            target_layers = ['conv4_3', 'conv5_1', 'conv5_2', 'conv5_3', 'flatten', 'fc6', 'fc7']
            extractor = Model(inputs=m.layers[0].input,
                              outputs=[m.get_layer(l).output for l in target_layers])

    return m


def transfer_learning_model(teacher_model, number_classes):
    for l in teacher_model.layers:
        l.trainable = False
    x = teacher_model.layers[-1].output
    x = Dense(number_classes)(x)
    x = Activation('softmax', name="act")(x)
    model = Model(teacher_model.input, x)

    opt = keras.optimizers.Adadelta()
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


def clip_img(X, preprocessing='raw'):
    X = utils_keras.reverse_preprocess(X, preprocessing)
    X = np.clip(X, 0.0, 255.0)
    X = utils_keras.preprocess(X, preprocessing)
    return X


def get_dataset_path(dataset):
    if dataset == "webface":
        train_data_dir = '/mnt/data/sixiongshan/data/webface/train'
        test_data_dir = '/mnt/data/sixiongshan/data/webface/test'
        number_classes = 10575
        number_samples = 475137

    elif dataset == "vggface1":
        train_data_dir = '/mnt/data/sixiongshan/data/vggface/train'
        test_data_dir = '/mnt/data/sixiongshan/data/vggface/test'
        number_classes = 2622
        number_samples = 1716436 // 3

    elif dataset == "vggface2":
        train_data_dir = '/mnt/data/sixiongshan/data/vggface2/train'
        test_data_dir = '/mnt/data/sixiongshan/data/vggface2/test'
        number_classes = 8631
        number_samples = 3141890 // 3

    elif dataset == "scrub":
        train_data_dir = '/mnt/data/sixiongshan/data/facescrub/keras_flow_dir/train'
        test_data_dir = '/mnt/data/sixiongshan/data/facescrub/keras_flow_dir/test'
        number_classes = 530
        number_samples = 57838

    elif dataset == "youtubeface":
        train_data_dir = '/mnt/data/sixiongshan/data/youtubeface/keras_flow_data/train_mtcnnpy_224'
        test_data_dir = '/mnt/data/sixiongshan/data/youtubeface/keras_flow_data/test_mtcnnpy_224'
        number_classes = 1283
        number_samples = 587137 // 5

    elif dataset == "emily":
        train_data_dir = '/mnt/data/sixiongshan/data/emface/train'
        test_data_dir = '/mnt/data/sixiongshan/data/emface/test'
        number_classes = 66
        number_samples = 6070

    elif dataset == "pubfig":
        train_data_dir = '/mnt/data/sixiongshan/data/pubfig/train'
        test_data_dir = '/mnt/data/sixiongshan/data/pubfig/test'
        number_classes = 65
        number_samples = 5979

    elif dataset == "iris":
        train_data_dir = '/mnt/data/sixiongshan/data/iris/train'
        test_data_dir = '/mnt/data/sixiongshan/data/iris/test'
        number_classes = 1000
        number_samples = 14000
    else:
        print("Dataset {} does not exist... Abort".format(dataset))
        exit(1)

    return train_data_dir, test_data_dir, number_classes, number_samples


def large_dataset_loader(dataset, augmentation=False, test_only=False, image_size=(224, 224)):
    train_data_dir, test_data_dir, number_classes, number_samples = get_dataset_path(dataset)
    train_generator, test_generator = generator_wrap(train_data_dir=train_data_dir, test_data_dir=test_data_dir,
                                                     augmentation=augmentation,
                                                     test_only=test_only, image_size=image_size)
    return train_generator, test_generator, number_classes, number_samples


def sample_from_generator(gen, nb_sample):
    x_test, y_test = gen.next()
    X_sample = np.zeros((0, x_test.shape[1], x_test.shape[2], x_test.shape[3]))
    Y_sample = np.zeros((0, y_test.shape[1]))

    while X_sample.shape[0] < nb_sample:
        x, y = gen.next()
        X_sample = np.concatenate((X_sample, x), axis=0)
        Y_sample = np.concatenate((Y_sample, y), axis=0)

    X_sample = X_sample[:nb_sample]
    Y_sample = Y_sample[:nb_sample]

    return X_sample, Y_sample


def generator_wrap(train_data_dir=None, test_data_dir=None, augmentation=False, test_only=False, image_size=(224, 224)):
    if not test_data_dir:
        validation_split = 0.05
    else:
        validation_split = 0
    if augmentation:
        data_gen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            rotation_range=20,
            width_shift_range=0.15,
            height_shift_range=0.15,
            shear_range=0.,
            zoom_range=0.15,
            channel_shift_range=0.,
            fill_mode='nearest',
            cval=0.,
            horizontal_flip=True, validation_split=validation_split)
    else:
        data_gen = ImageDataGenerator(preprocessing_function=preprocess_input, validation_split=validation_split)

    if test_data_dir is None:
        train_generator = data_gen.flow_from_directory(
            train_data_dir,
            target_size=image_size,
            batch_size=32, subset='training')
        test_generator = data_gen.flow_from_directory(
            train_data_dir,
            target_size=image_size,
            batch_size=32, subset='validation')
    else:
        if test_only:
            train_generator = None
        else:
            train_generator = data_gen.flow_from_directory(
                train_data_dir,
                target_size=image_size,
                batch_size=32)
        test_generator = data_gen.flow_from_directory(
            test_data_dir,
            target_size=image_size,
            batch_size=32)

    return train_generator, test_generator


class MergeLayer(Layer):

    def __init__(self, **kwargs):
        self.result = None
        super(MergeLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        kernel_1_shape = (5 * 4 * 60, 160)
        kernel_2_shape = (4 * 3 * 80, 160)
        bias_shape = (160,)
        self.kernel_1 = self.add_weight(name='kernel_1',
                                        shape=kernel_1_shape,
                                        initializer='uniform',
                                        trainable=True)
        self.kernel_2 = self.add_weight(name='kernel_2',
                                        shape=kernel_2_shape,
                                        initializer='uniform',
                                        trainable=True)
        self.bias = self.add_weight(name='bias',
                                    shape=bias_shape,
                                    initializer='uniform',
                                    trainable=True)
        super(MergeLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        layer1 = x[0]
        layer2 = x[1]
        layer1_r = K.reshape(layer1, (-1, 5 * 4 * 60))
        layer2_r = K.reshape(layer2, (-1, 4 * 3 * 80))
        self.result = K.dot(layer1_r, self.kernel_1) + \
                      K.dot(layer2_r, self.kernel_2) + self.bias
        return self.result

    def compute_output_shape(self, input_shape):
        return K.int_shape(self.result)


def load_deepid_model(class_num):
    input_shape = (55, 47, 3)

    img_input = Input(shape=input_shape)
    h1 = Conv2D(20, (4, 4), strides=(1, 1), padding='valid', name='conv_1')(img_input)
    h1 = Activation('relu')(h1)
    h1 = MaxPooling2D((2, 2), strides=(2, 2), name='pool_1')(h1)

    h2 = Conv2D(40, (3, 3), strides=(1, 1), padding='valid', name='conv_2')(h1)
    h2 = Activation('relu')(h2)
    h2 = MaxPooling2D((2, 2), strides=(2, 2), name='pool_2')(h2)

    h3 = Conv2D(60, (3, 3), strides=(1, 1), padding='valid', name='conv_3')(h2)
    h3 = Activation('relu')(h3)
    h3 = MaxPooling2D((2, 2), strides=(2, 2), name='pool_3')(h3)

    h4 = Conv2D(80, (2, 2), strides=(1, 1), padding='valid', name='conv_4')(h3)
    h4 = Activation('relu')(h4)

    h5 = MergeLayer()([h3, h4])
    h5 = Activation('relu')(h5)

    h5 = Dense(class_num, name='fc')(h5)
    h5 = Activation('softmax')(h5)

    inputs = img_input
    model = Model(inputs, h5, name='vgg_face')
    return model


def get_label_data(X, Y, target):
    X_filter = np.array(X)
    Y_filter = np.array(Y)
    remain_idx = np.argmax(Y, axis=1) == target
    X_filter = X_filter[remain_idx]
    Y_filter = Y_filter[remain_idx]
    return X_filter, Y_filter


def get_other_label_data(X, Y, target):
    X_filter = np.array(X)
    Y_filter = np.array(Y)
    remain_idx = np.argmax(Y, axis=1) != target
    X_filter = X_filter[remain_idx]
    Y_filter = Y_filter[remain_idx]
    return X_filter, Y_filter


def get_labels_data(X, Y, target_ls):
    assert isinstance(target_ls, list)
    X_filter = np.array(X)
    Y_filter = np.array(Y)
    remain_idx = np.array([False] * len(Y_filter))
    for target in target_ls:
        cur_remain_idx = np.argmax(Y, axis=1) == target
        remain_idx = np.logical_or(remain_idx, cur_remain_idx)

    X_filter = X_filter[remain_idx]
    Y_filter = Y_filter[remain_idx]
    return X_filter, Y_filter


def get_other_labels_data_except(X, Y, target_ls):
    assert isinstance(target_ls, list)

    X_filter = np.array(X)
    Y_filter = np.array(Y)
    remain_idx = np.array([True] * len(Y_filter))
    for target in target_ls:
        cur_remain_idx = np.argmax(Y, axis=1) != target
        remain_idx = np.logical_and(remain_idx, cur_remain_idx)

    X_filter = X_filter[remain_idx]
    Y_filter = Y_filter[remain_idx]
    return X_filter, Y_filter


def get_bottom_top_model(model, layer_name):
    layer = model.get_layer(layer_name)
    bottom_input = Input(model.input_shape[1:])
    bottom_output = bottom_input
    top_input = Input(layer.output_shape[1:])
    top_output = top_input

    bottom = True
    for layer in model.layers:
        if bottom:
            bottom_output = layer(bottom_output)
        else:
            top_output = layer(top_output)
        if layer.name == layer_name:
            bottom = False

    bottom_model = Model(bottom_input, bottom_output)
    top_model = Model(top_input, top_output)

    return bottom_model, top_model


def load_end2end_model(arch, number_classes):
    if arch == 'resnet':
        MODEL = keras.applications.resnet_v2.ResNet152V2(include_top=False, weights='imagenet', pooling='avg',
                                                         input_shape=(224, 224, 3))
    elif arch == 'inception':
        MODEL = keras.applications.InceptionResNetV2(include_top=False, weights='imagenet', pooling='avg',
                                                     input_shape=(224, 224, 3))
    elif arch == 'mobile':
        MODEL = keras.applications.mobilenet_v2.MobileNetV2(include_top=False, weights='imagenet', pooling='avg',
                                                            input_shape=(224, 224, 3))
    elif arch == 'dense':
        MODEL = keras.applications.densenet.DenseNet121(include_top=False, weights='imagenet', pooling='avg',
                                                        input_shape=(224, 224, 3))

    model = load_victim_model(number_classes, MODEL, end2end=True)
    return model


def load_victim_model(number_classes, teacher_model=None, end2end=False):
    for l in teacher_model.layers:
        l.trainable = end2end
    x = teacher_model.layers[-1].output

    x = Dense(number_classes)(x)
    x = Activation('softmax', name="act")(x)
    model = Model(teacher_model.input, x)
    opt = keras.optimizers.Adadelta()
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


def add_last_layer(number_classes, teacher_model, cut_to_layer=None):
    for l in teacher_model.layers:
        l.trainable = False

    if cut_to_layer:
        x = teacher_model.layers[cut_to_layer].output
        print(teacher_model.layers[cut_to_layer].name)
    else:
        x = teacher_model.layers[-1].output

    x = Dense(number_classes, name='softmax')(x)
    x = Activation('softmax', name="act")(x)
    model = Model(teacher_model.input, x)

    opt = keras.optimizers.Adadelta()
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


def resize_batch(x, target_size=(224, 224), intensity="imagenet"):
    if x.shape[:2] == target_size:
        return x

    x = utils_keras.reverse_preprocess(x, intensity)
    resized = np.array([resize(a, target_size) for a in x])
    return utils_keras.preprocess(resized, intensity)


def build_bottleneck_model(model, cut_off):
    bottleneck_model = Model(model.input, model.get_layer(cut_off).output)
    bottleneck_model.compile(loss='categorical_crossentropy',
                             optimizer='adam',
                             metrics=['accuracy'])

    return bottleneck_model


def split_dataset(X, y, ratio=0.3):
    x_appro, x_later, y_appro, y_later = train_test_split(X, y, test_size=ratio, random_state=0)
    return x_appro, x_later, y_appro, y_later


def data_generator(X, Y, batch_size=32, target_size=(224, 224), intensity='imagenet'):
    data_gen = ImageDataGenerator()
    data_gen = data_gen.flow(X, Y, batch_size=batch_size)
    while True:
        cur_X, cur_Y = next(data_gen)
        cur_X = resize_batch(cur_X, target_size=target_size, intensity=intensity)
        yield np.array(cur_X), cur_Y


def evaluate(model, X_test, Y_test, batch_size=32, target_size=(224, 224)):
    test_other_gen = data_generator(X_test, Y_test, batch_size=batch_size, target_size=target_size)
    if len(X_test) < batch_size * 2:
        batch_size = 1
    test_other_step = len(X_test) // batch_size // 2
    acc = model.evaluate_generator(test_other_gen, steps=test_other_step, verbose=0)[1]
    return acc


def normalize(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


class CloakData(object):
    def __init__(self, dataset, img_shape=(224, 224), target_selection_tries=30, protect_class=None):
        self.dataset = dataset
        self.img_shape = img_shape
        self.target_selection_tries = target_selection_tries

        self.train_data_dir, self.test_data_dir, self.number_classes, self.number_samples = get_dataset_path(dataset)
        self.all_labels = sorted(list(os.listdir(self.train_data_dir)))
        if protect_class:
            self.protect_class = protect_class
        else:
            self.protect_class = random.choice(self.all_labels)

        self.sybil_class = random.choice([l for l in self.all_labels if l != self.protect_class])
        print("Protect label: {} | Sybil label: {}".format(self.protect_class, self.sybil_class))
        self.protect_train_X, self.protect_test_X = self.load_label_data(self.protect_class)
        self.sybil_train_X, self.sybil_test_X = self.load_label_data(self.sybil_class)
        # self.target_path, self.target_data = self.select_target_label()

        self.cloaked_protect_train_X = None
        self.cloaked_sybil_train_X = None

        self.label2path_train, self.label2path_test, self.path2idx = self.build_data_mapping()
        self.all_training_path = self.get_all_data_path(self.label2path_train)
        self.all_test_path = self.get_all_data_path(self.label2path_test)
        self.protect_class_path = self.get_class_image_files(os.path.join(self.train_data_dir, self.protect_class))
        self.sybil_class_path = self.get_class_image_files(os.path.join(self.train_data_dir, self.sybil_class))

        print(
            "Find {} protect images | {} sybil images".format(len(self.protect_class_path), len(self.sybil_class_path)))

    def get_class_image_files(self, path):
        return [os.path.join(path, f) for f in os.listdir(path)]

    def extractor_ls_predict(self, feature_extractors_ls, X):
        feature_ls = []
        for extractor in feature_extractors_ls:
            cur_features = extractor.predict(X)
            feature_ls.append(cur_features)
        concated_feature_ls = np.concatenate(feature_ls, axis=1)
        concated_feature_ls = normalize(concated_feature_ls)
        return concated_feature_ls

    def load_embeddings(self, feature_extractors_names):
        dictionaries = []

        for extractor_name in feature_extractors_names:
            path2emb = pickle.load(open("/home/shansixioing/cloak/embs/{}_emb_norm.p".format(extractor_name), "rb"))
            # path2emb = pickle.load(open("/home/shansixioing/cloak/embs/vggface2_inception_emb.p".format(extractor_name), "rb"))
            dictionaries.append(path2emb)
        merge_dict = {}
        for k in dictionaries[0].keys():
            cur_emb = [dic[k] for dic in dictionaries]
            merge_dict[k] = np.concatenate(cur_emb)
        return merge_dict

    def select_target_label(self, feature_extractors_ls, feature_extractors_names, metric='l2'):
        # original_feature_x = extractor.predict(self.protect_train_X)
        original_feature_x = self.extractor_ls_predict(feature_extractors_ls, self.protect_train_X)

        path2emb = self.load_embeddings(feature_extractors_names)
        # items = list(path2emb.items())
        teacher_dataset = feature_extractors_names[0].split("_")[0]
        # items = [(k, v) for k, v in path2emb.items() if teacher_dataset in k]
        items = list(path2emb.items())
        paths = [p[0] for p in items]
        embs = [p[1] for p in items]
        embs = np.array(embs)

        pair_dist = pairwise_distances(original_feature_x, embs, 'l2')
        max_sum = np.min(pair_dist, axis=0)
        sorted_idx = np.argsort(max_sum)[::-1]

        highest_num = 0
        paired_target_X = None
        final_target_class_path = None
        for idx in sorted_idx[:2]:
            target_class_path = paths[idx]
            cur_target_X = self.load_dir(target_class_path)
            cur_target_X = np.concatenate([cur_target_X, cur_target_X, cur_target_X])

            cur_tot_sum, cur_paired_target_X = self.calculate_dist_score(self.protect_train_X, cur_target_X,
                                                                         feature_extractors_ls,
                                                                         metric=metric)
            if cur_tot_sum > highest_num:
                highest_num = cur_tot_sum
                paired_target_X = cur_paired_target_X
                final_target_class_path = target_class_path

        np.random.shuffle(paired_target_X)
        return final_target_class_path, paired_target_X

    def calculate_dist_score(self, a, b, feature_extractors_ls, metric='l2'):
        features1 = self.extractor_ls_predict(feature_extractors_ls, a)
        features2 = self.extractor_ls_predict(feature_extractors_ls, b)

        pair_cos = pairwise_distances(features1, features2, metric)
        max_sum = np.min(pair_cos, axis=0)
        max_sum_arg = np.argsort(max_sum)[::-1]
        max_sum_arg = max_sum_arg[:len(a)]
        max_sum = [max_sum[i] for i in max_sum_arg]
        paired_target_X = [b[j] for j in max_sum_arg]
        paired_target_X = np.array(paired_target_X)
        return np.min(max_sum), paired_target_X

    def get_all_data_path(self, label2path):
        all_paths = []
        for k, v in label2path.items():
            cur_all_paths = [os.path.join(k, cur_p) for cur_p in v]
            all_paths.extend(cur_all_paths)
        return all_paths

    def load_label_data(self, label):
        train_label_path = os.path.join(self.train_data_dir, label)
        test_label_path = os.path.join(self.test_data_dir, label)
        train_X = self.load_dir(train_label_path)
        test_X = self.load_dir(test_label_path)
        return train_X, test_X

    def load_dir(self, path):
        assert os.path.exists(path)
        x_ls = []
        for file in os.listdir(path):
            cur_path = os.path.join(path, file)
            im = image.load_img(cur_path, target_size=self.img_shape)
            im = image.img_to_array(im)
            x_ls.append(im)
        raw_x = np.array(x_ls)
        return preprocess_input(raw_x)

    def build_data_mapping(self):
        label2path_train = {}
        label2path_test = {}
        idx = 0
        path2idx = {}
        for label_name in self.all_labels:
            full_path_train = os.path.join(self.train_data_dir, label_name)
            full_path_test = os.path.join(self.test_data_dir, label_name)
            label2path_train[full_path_train] = list(os.listdir(full_path_train))
            label2path_test[full_path_test] = list(os.listdir(full_path_test))
            for img_file in os.listdir(full_path_train):
                path2idx[os.path.join(full_path_train, img_file)] = idx
            for img_file in os.listdir(full_path_test):
                path2idx[os.path.join(full_path_test, img_file)] = idx
            idx += 1
        return label2path_train, label2path_test, path2idx

    def generate_data_post_cloak(self, sybil=False):
        assert self.cloaked_protect_train_X is not None
        while True:
            batch_X = []
            batch_Y = []
            cur_batch_path = random.sample(self.all_training_path, 32)
            for p in cur_batch_path:
                cur_y = self.path2idx[p]
                if p in self.protect_class_path:
                    cur_x = random.choice(self.cloaked_protect_train_X)
                elif sybil and (p in self.sybil_class):
                    cur_x = random.choice(self.cloaked_sybil_train_X)
                else:
                    im = image.load_img(p, target_size=self.img_shape)
                    im = image.img_to_array(im)
                    cur_x = preprocess_input(im)
                batch_X.append(cur_x)
                batch_Y.append(cur_y)
            batch_X = np.array(batch_X)
            batch_Y = to_categorical(np.array(batch_Y), num_classes=self.number_classes)
            yield batch_X, batch_Y
