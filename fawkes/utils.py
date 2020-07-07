import glob
import gzip
import json
import os
import pickle
import random
import shutil
import sys
import tarfile
import zipfile

import six
from six.moves.urllib.error import HTTPError, URLError

stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import keras

sys.stderr = stderr
import keras.backend as K
import numpy as np
import tensorflow as tf
from PIL import Image, ExifTags
from keras.layers import Dense, Activation
from keras.models import Model
from keras.preprocessing import image
from skimage.transform import resize
from sklearn.metrics import pairwise_distances

from fawkes.align_face import align, aligner
from six.moves.urllib.request import urlopen

if sys.version_info[0] == 2:
    def urlretrieve(url, filename, reporthook=None, data=None):
        def chunk_read(response, chunk_size=8192, reporthook=None):
            content_type = response.info().get('Content-Length')
            total_size = -1
            if content_type is not None:
                total_size = int(content_type.strip())
            count = 0
            while True:
                chunk = response.read(chunk_size)
                count += 1
                if reporthook is not None:
                    reporthook(count, chunk_size, total_size)
                if chunk:
                    yield chunk
                else:
                    break

        response = urlopen(url, data)
        with open(filename, 'wb') as fd:
            for chunk in chunk_read(response, reporthook=reporthook):
                fd.write(chunk)
else:
    from six.moves.urllib.request import urlretrieve


def clip_img(X, preprocessing='raw'):
    X = reverse_preprocess(X, preprocessing)
    X = np.clip(X, 0.0, 255.0)
    X = preprocess(X, preprocessing)
    return X


def load_image(path):
    img = Image.open(path)
    if img._getexif() is not None:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break

        exif = dict(img._getexif().items())
        if orientation in exif.keys():
            if exif[orientation] == 3:
                img = img.rotate(180, expand=True)
            elif exif[orientation] == 6:
                img = img.rotate(270, expand=True)
            elif exif[orientation] == 8:
                img = img.rotate(90, expand=True)
            else:
                pass
    img = img.convert('RGB')
    image_array = image.img_to_array(img)

    return image_array


class Faces(object):
    def __init__(self, image_paths, sess, verbose=1):
        model_dir = os.path.join(os.path.expanduser('~'), '.fawkes')
        if not os.path.exists(os.path.join(model_dir, "mtcnn.p.gz")):
            os.makedirs(model_dir, exist_ok=True)
            get_file("mtcnn.p.gz", "http://sandlab.cs.uchicago.edu/fawkes/files/mtcnn.p.gz", cache_dir=model_dir,
                     cache_subdir='')

        self.verbose = verbose
        self.aligner = aligner(sess)
        self.org_faces = []
        self.cropped_faces = []
        self.cropped_faces_shape = []
        self.cropped_index = []
        self.callback_idx = []
        if verbose:
            print("Identify {} images".format(len(image_paths)))
        for i, p in enumerate(image_paths):
            cur_img = load_image(p)
            self.org_faces.append(cur_img)
            align_img = align(cur_img, self.aligner, margin=0.7)
            if align_img is None:
                print("Find 0 face(s) in {}".format(p.split("/")[-1]))
                continue

            cur_faces = align_img[0]

            cur_shapes = [f.shape[:-1] for f in cur_faces]

            cur_faces_square = []
            if verbose:
                print("Find {} face(s) in {}".format(len(cur_faces), p.split("/")[-1]))

            for img in cur_faces:
                long_size = max([img.shape[1], img.shape[0]])
                base = np.zeros((long_size, long_size, 3))
                base[0:img.shape[0], 0:img.shape[1], :] = img
                cur_faces_square.append(base)

            cur_index = align_img[1]
            cur_faces_square = [resize(f, (224, 224)) for f in cur_faces_square]
            self.cropped_faces_shape.extend(cur_shapes)
            self.cropped_faces.extend(cur_faces_square)
            self.cropped_index.extend(cur_index)
            self.callback_idx.extend([i] * len(cur_faces_square))

        if not self.cropped_faces:
            print("No faces detected")
            exit(1)

        self.cropped_faces = np.array(self.cropped_faces)

        self.cropped_faces = preprocess(self.cropped_faces, 'imagenet')

        self.cloaked_cropped_faces = None
        self.cloaked_faces = np.copy(self.org_faces)

    def get_faces(self):
        return self.cropped_faces

    def merge_faces(self, cloaks):

        self.cloaked_faces = np.copy(self.org_faces)

        for i in range(len(self.cropped_faces)):
            cur_cloak = cloaks[i]
            org_shape = self.cropped_faces_shape[i]
            old_square_shape = max([org_shape[0], org_shape[1]])
            reshape_cloak = resize(cur_cloak, (old_square_shape, old_square_shape))
            reshape_cloak = reshape_cloak[0:org_shape[0], 0:org_shape[1], :]

            callback_id = self.callback_idx[i]
            bb = self.cropped_index[i]
            self.cloaked_faces[callback_id][bb[1]:bb[3], bb[0]:bb[2], :] += reshape_cloak

        return self.cloaked_faces


def dump_dictionary_as_json(dict, outfile):
    j = json.dumps(dict)
    with open(outfile, "wb") as f:
        f.write(j.encode())


def fix_gpu_memory(mem_fraction=1):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf_config = None
    if tf.test.is_gpu_available():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=mem_fraction)
        tf_config = tf.ConfigProto(gpu_options=gpu_options)
        tf_config.gpu_options.allow_growth = True
        tf_config.log_device_placement = False
    init_op = tf.global_variables_initializer()
    sess = tf.Session(config=tf_config)
    sess.run(init_op)
    K.set_session(sess)
    return sess


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
    assert method in {'raw', 'imagenet', 'inception', 'mnist'}

    if method is 'raw':
        pass
    elif method is 'imagenet':
        X = imagenet_preprocessing(X)
    else:
        raise Exception('unknown method %s' % method)

    return X


def reverse_preprocess(X, method):
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


def reverse_process_cloaked(x, preprocess='imagenet'):
    # x = clip_img(x, preprocess)
    return reverse_preprocess(x, preprocess)


def build_bottleneck_model(model, cut_off):
    bottleneck_model = Model(model.input, model.get_layer(cut_off).output)
    bottleneck_model.compile(loss='categorical_crossentropy',
                             optimizer='adam',
                             metrics=['accuracy'])
    return bottleneck_model


def load_extractor(name):
    model_dir = os.path.join(os.path.expanduser('~'), '.fawkes')
    os.makedirs(model_dir, exist_ok=True)
    model_file = os.path.join(model_dir, "{}.h5".format(name))
    emb_file = os.path.join(model_dir, "{}_emb.p.gz".format(name))
    if os.path.exists(model_file):
        model = keras.models.load_model(model_file)
    else:
        print("Download models...")
        get_file("{}.h5".format(name), "http://sandlab.cs.uchicago.edu/fawkes/files/{}.h5".format(name),
                 cache_dir=model_dir, cache_subdir='')
        model = keras.models.load_model(model_file)

    if not os.path.exists(emb_file):
        get_file("{}_emb.p.gz".format(name), "http://sandlab.cs.uchicago.edu/fawkes/files/{}_emb.p.gz".format(name),
                 cache_dir=model_dir, cache_subdir='')

    if hasattr(model.layers[-1], "activation") and model.layers[-1].activation == "softmax":
        raise Exception(
            "Given extractor's last layer is softmax, need to remove the top layers to make it into a feature extractor")
    # if "extract" in name.split("/")[-1]:
    #     pass
    # else:
    #     print("Convert a model to a feature extractor")
    #     model = build_bottleneck_model(model, model.layers[layer_idx].name)
    #     model.save(name + "extract")
    #     model = keras.models.load_model(name + "extract")
    return model


def get_dataset_path(dataset):
    model_dir = os.path.join(os.path.expanduser('~'), '.fawkes')
    if not os.path.exists(os.path.join(model_dir, "config.json")):
        raise Exception("Please config the datasets before running protection code. See more in README and config.py.")

    config = json.load(open(os.path.join(model_dir, "config.json"), 'r'))
    if dataset not in config:
        raise Exception(
            "Dataset {} does not exist, please download to data/ and add the path to this function... Abort".format(
                dataset))
    return config[dataset]['train_dir'], config[dataset]['test_dir'], config[dataset]['num_classes'], config[dataset][
        'num_images']


def normalize(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def dump_image(x, filename, format="png", scale=False):
    # img = image.array_to_img(x, scale=scale)
    img = image.array_to_img(x)
    img.save(filename, format)
    return


def load_dir(path):
    assert os.path.exists(path)
    x_ls = []
    for file in os.listdir(path):
        cur_path = os.path.join(path, file)
        im = image.load_img(cur_path, target_size=(224, 224))
        im = image.img_to_array(im)
        x_ls.append(im)
    raw_x = np.array(x_ls)
    return preprocess(raw_x, 'imagenet')


def load_embeddings(feature_extractors_names):
    model_dir = os.path.join(os.path.expanduser('~'), '.fawkes')
    dictionaries = []
    for extractor_name in feature_extractors_names:
        fp = gzip.open(os.path.join(model_dir, "{}_emb.p.gz".format(extractor_name)), 'rb')
        path2emb = pickle.load(fp)
        fp.close()

        dictionaries.append(path2emb)

    merge_dict = {}
    for k in dictionaries[0].keys():
        cur_emb = [dic[k] for dic in dictionaries]
        merge_dict[k] = np.concatenate(cur_emb)
    return merge_dict


def extractor_ls_predict(feature_extractors_ls, X):
    feature_ls = []
    for extractor in feature_extractors_ls:
        cur_features = extractor.predict(X)
        feature_ls.append(cur_features)
    concated_feature_ls = np.concatenate(feature_ls, axis=1)
    concated_feature_ls = normalize(concated_feature_ls)
    return concated_feature_ls


def calculate_dist_score(a, b, feature_extractors_ls, metric='l2'):
    features1 = extractor_ls_predict(feature_extractors_ls, a)
    features2 = extractor_ls_predict(feature_extractors_ls, b)

    pair_cos = pairwise_distances(features1, features2, metric)
    max_sum = np.min(pair_cos, axis=0)
    max_sum_arg = np.argsort(max_sum)[::-1]
    max_sum_arg = max_sum_arg[:len(a)]
    max_sum = [max_sum[i] for i in max_sum_arg]
    paired_target_X = [b[j] for j in max_sum_arg]
    paired_target_X = np.array(paired_target_X)
    return np.min(max_sum), paired_target_X


def select_target_label(imgs, feature_extractors_ls, feature_extractors_names, metric='l2'):
    model_dir = os.path.join(os.path.expanduser('~'), '.fawkes')

    original_feature_x = extractor_ls_predict(feature_extractors_ls, imgs)

    path2emb = load_embeddings(feature_extractors_names)
    items = list(path2emb.items())
    paths = [p[0] for p in items]
    embs = [p[1] for p in items]
    embs = np.array(embs)

    pair_dist = pairwise_distances(original_feature_x, embs, metric)
    max_sum = np.min(pair_dist, axis=0)
    max_id = np.argmax(max_sum)

    target_data_id = paths[int(max_id)]
    image_dir = os.path.join(model_dir, "target_data/{}".format(target_data_id))
    # if not os.path.exists(image_dir):
    os.makedirs(os.path.join(model_dir, "target_data"), exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)
    for i in range(10):
        if os.path.exists(os.path.join(model_dir, "target_data/{}/{}.jpg".format(target_data_id, i))):
            continue
        get_file("{}.jpg".format(i),
                 "http://sandlab.cs.uchicago.edu/fawkes/files/target_data/{}/{}.jpg".format(target_data_id, i),
                 cache_dir=model_dir, cache_subdir='target_data/{}/'.format(target_data_id))

    image_paths = glob.glob(image_dir + "/*.jpg")

    target_images = [image.img_to_array(image.load_img(cur_path)) for cur_path in
                     image_paths]

    target_images = np.array([resize(x, (224, 224)) for x in target_images])
    target_images = preprocess(target_images, 'imagenet')

    target_images = list(target_images)
    while len(target_images) < len(imgs):
        target_images += target_images

    target_images = random.sample(target_images, len(imgs))
    return np.array(target_images)


def get_file(fname,
             origin,
             untar=False,
             md5_hash=None,
             file_hash=None,
             cache_subdir='datasets',
             hash_algorithm='auto',
             extract=False,
             archive_format='auto',
             cache_dir=None):
    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser('~'), '.keras')
    if md5_hash is not None and file_hash is None:
        file_hash = md5_hash
        hash_algorithm = 'md5'
    datadir_base = os.path.expanduser(cache_dir)
    if not os.access(datadir_base, os.W_OK):
        datadir_base = os.path.join('/tmp', '.keras')
    datadir = os.path.join(datadir_base, cache_subdir)
    _makedirs_exist_ok(datadir)

    if untar:
        untar_fpath = os.path.join(datadir, fname)
        fpath = untar_fpath + '.tar.gz'
    else:
        fpath = os.path.join(datadir, fname)

    download = False
    if not os.path.exists(fpath):
        download = True

    if download:
        error_msg = 'URL fetch failure on {}: {} -- {}'
        dl_progress = None
        try:
            try:
                urlretrieve(origin, fpath, dl_progress)
            except HTTPError as e:
                raise Exception(error_msg.format(origin, e.code, e.msg))
            except URLError as e:
                raise Exception(error_msg.format(origin, e.errno, e.reason))
        except (Exception, KeyboardInterrupt) as e:
            if os.path.exists(fpath):
                os.remove(fpath)
            raise
        # ProgressTracker.progbar = None

    if untar:
        if not os.path.exists(untar_fpath):
            _extract_archive(fpath, datadir, archive_format='tar')
        return untar_fpath

    if extract:
        _extract_archive(fpath, datadir, archive_format)

    return fpath


def _extract_archive(file_path, path='.', archive_format='auto'):
    if archive_format is None:
        return False
    if archive_format == 'auto':
        archive_format = ['tar', 'zip']
    if isinstance(archive_format, six.string_types):
        archive_format = [archive_format]

    for archive_type in archive_format:
        if archive_type == 'tar':
            open_fn = tarfile.open
            is_match_fn = tarfile.is_tarfile
        if archive_type == 'zip':
            open_fn = zipfile.ZipFile
            is_match_fn = zipfile.is_zipfile

        if is_match_fn(file_path):
            with open_fn(file_path) as archive:
                try:
                    archive.extractall(path)
                except (tarfile.TarError, RuntimeError, KeyboardInterrupt):
                    if os.path.exists(path):
                        if os.path.isfile(path):
                            os.remove(path)
                        else:
                            shutil.rmtree(path)
                    raise
            return True
    return False


def _makedirs_exist_ok(datadir):
    if six.PY2:
        # Python 2 doesn't have the exist_ok arg, so we try-except here.
        try:
            os.makedirs(datadir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
    else:
        os.makedirs(datadir, exist_ok=True)  # pylint: disable=unexpected-keyword-arg
