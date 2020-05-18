import os
import pickle
import random

import keras
import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.utils import to_categorical
from sklearn.metrics import pairwise_distances


def fix_gpu_memory(mem_fraction=1):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=mem_fraction)
    tf_config = tf.ConfigProto(gpu_options=gpu_options)
    tf_config.gpu_options.allow_growth = True
    tf_config.log_device_placement = False
    init_op = tf.global_variables_initializer()
    sess = tf.Session(config=tf_config)
    sess.run(init_op)
    K.set_session(sess)
    return sess


def init_gpu(gpu_index, force=False):
    if isinstance(gpu_index, list):
        gpu_num = ','.join([str(i) for i in gpu_index])
    else:
        gpu_num = str(gpu_index)
    if "CUDA_VISIBLE_DEVICES" in os.environ and os.environ["CUDA_VISIBLE_DEVICES"] and not force:
        print('GPU already initiated')
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_num
    sess = fix_gpu_memory()
    return sess


def preprocess(X, method):
    # assume color last
    assert method in {'raw', 'imagenet', 'inception', 'mnist'}

    if method is 'raw':
        pass
    elif method is 'imagenet':
        X = imagenet_preprocessing(X)
    else:
        raise Exception('unknown method %s' % method)

    return X


def reverse_preprocess(X, method):
    # assume color last
    assert method in {'raw', 'imagenet', 'inception', 'mnist'}

    if method is 'raw':
        pass
    elif method is 'imagenet':
        X = imagenet_reverse_preprocessing(X)
    else:
        raise Exception('unknown method %s' % method)

    return X


def imagenet_preprocessing(x, data_format=None):
    if data_format is None:
        data_format = K.image_data_format()
    assert data_format in ('channels_last', 'channels_first')

    x = np.array(x)
    if data_format == 'channels_first':
        # 'RGB'->'BGR'
        if x.ndim == 3:
            x = x[::-1, ...]
        else:
            x = x[:, ::-1, ...]
    else:
        # 'RGB'->'BGR'
        x = x[..., ::-1]

    mean = [103.939, 116.779, 123.68]
    std = None

    # Zero-center by mean pixel
    if data_format == 'channels_first':
        if x.ndim == 3:
            x[0, :, :] -= mean[0]
            x[1, :, :] -= mean[1]
            x[2, :, :] -= mean[2]
            if std is not None:
                x[0, :, :] /= std[0]
                x[1, :, :] /= std[1]
                x[2, :, :] /= std[2]
        else:
            x[:, 0, :, :] -= mean[0]
            x[:, 1, :, :] -= mean[1]
            x[:, 2, :, :] -= mean[2]
            if std is not None:
                x[:, 0, :, :] /= std[0]
                x[:, 1, :, :] /= std[1]
                x[:, 2, :, :] /= std[2]
    else:
        x[..., 0] -= mean[0]
        x[..., 1] -= mean[1]
        x[..., 2] -= mean[2]
        if std is not None:
            x[..., 0] /= std[0]
            x[..., 1] /= std[1]
            x[..., 2] /= std[2]

    return x


def imagenet_reverse_preprocessing(x, data_format=None):
    import keras.backend as K
    """ Reverse preprocesses a tensor encoding a batch of images.
    # Arguments
        x: input Numpy tensor, 4D.
        data_format: data format of the image tensor.
    # Returns
        Preprocessed tensor.
    """
    x = np.array(x)
    if data_format is None:
        data_format = K.image_data_format()
    assert data_format in ('channels_last', 'channels_first')

    if data_format == 'channels_first':
        if x.ndim == 3:
            # Zero-center by mean pixel
            x[0, :, :] += 103.939
            x[1, :, :] += 116.779
            x[2, :, :] += 123.68
            # 'BGR'->'RGB'
            x = x[::-1, :, :]
        else:
            x[:, 0, :, :] += 103.939
            x[:, 1, :, :] += 116.779
            x[:, 2, :, :] += 123.68
            x = x[:, ::-1, :, :]
    else:
        # Zero-center by mean pixel
        x[..., 0] += 103.939
        x[..., 1] += 116.779
        x[..., 2] += 123.68
        # 'BGR'->'RGB'
        x = x[..., ::-1]
    return x


def imagenet_reverse_preprocessing_cntk(x, data_format=None):
    import keras.backend as K
    """ Reverse preprocesses a tensor encoding a batch of images.
    # Arguments
        x: input Numpy tensor, 4D.
        data_format: data format of the image tensor.
    # Returns
        Preprocessed tensor.
    """
    x = np.array(x)
    if data_format is None:
        data_format = K.image_data_format()
    assert data_format in ('channels_last', 'channels_first')

    if data_format == 'channels_first':
        # Zero-center by mean pixel
        x[:, 0, :, :] += 114.0
        x[:, 1, :, :] += 114.0
        x[:, 2, :, :] += 114.0
        # 'BGR'->'RGB'
        x = x[:, ::-1, :, :]
    else:
        # Zero-center by mean pixel
        x[:, :, :, 0] += 114.0
        x[:, :, :, 1] += 114.0
        x[:, :, :, 2] += 114.0
        # 'BGR'->'RGB'
        x = x[:, :, :, ::-1]
    return x


def load_extractor(name):
    model = keras.models.load_model("/home/shansixioing/cloak/models/extractors/{}_extract.h5".format(name))
    return model


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

        self.cloaked_protect_train_X = None
        self.cloaked_sybil_train_X = None

        self.label2path_train, self.label2path_test, self.path2idx = self.build_data_mapping()
        self.all_training_path = self.get_all_data_path(self.label2path_train)
        self.all_test_path = self.get_all_data_path(self.label2path_test)
        self.protect_class_path = self.get_class_image_files(os.path.join(self.train_data_dir, self.protect_class))
        self.sybil_class_path = self.get_class_image_files(os.path.join(self.train_data_dir, self.sybil_class))

        print("Find {} protect images".format(len(self.protect_class_path)))

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
        original_feature_x = self.extractor_ls_predict(feature_extractors_ls, self.protect_train_X)

        path2emb = self.load_embeddings(feature_extractors_names)
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
