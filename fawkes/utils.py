#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-05-17
# @Author  : Shawn Shan (shansixiong@cs.uchicago.edu)
# @Link    : https://www.shawnshan.com/


import errno
import glob
import gzip
import hashlib
import os
import pickle
import random
import shutil
import sys
import tarfile
import zipfile

import pkg_resources
import six
from keras.utils import Progbar
from six.moves.urllib.error import HTTPError, URLError

stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
from tensorflow.keras.models import load_model

sys.stderr = stderr
from tensorflow.keras.backend import image_data_format
import numpy as np
import tensorflow as tf
from PIL import Image, ExifTags, UnidentifiedImageError
from tensorflow.keras.preprocessing import image

from fawkes.align_face import align
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

IMG_SIZE = 112
PREPROCESS = 'raw'


def load_image(path):
    try:
        img = Image.open(path)
    except UnidentifiedImageError:
        return None
    except IsADirectoryError:
        return None

    try:
        info = img._getexif()
    except OSError:
        return None

    if info is not None:
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


def filter_image_paths(image_paths):
    print("Identify {} files in the directory".format(len(image_paths)))
    new_image_paths = []
    new_images = []
    for p in image_paths:
        img = load_image(p)
        if img is None:
            print("{} is not an image file, skipped".format(p.split("/")[-1]))
            continue
        new_image_paths.append(p)
        new_images.append(img)
    print("Identify {} images in the directory".format(len(new_image_paths)))
    return new_image_paths, new_images


class Faces(object):
    def __init__(self, image_paths, loaded_images, aligner, verbose=1, eval_local=False, preprocessing=True,
                 no_align=False):
        self.image_paths = image_paths
        self.verbose = verbose
        self.no_align = no_align
        self.aligner = aligner
        self.margin = 30
        self.org_faces = []
        self.cropped_faces = []
        self.cropped_faces_shape = []
        self.cropped_index = []
        self.start_end_ls = []
        self.callback_idx = []
        self.images_without_face = []
        for i in range(0, len(loaded_images)):
            cur_img = loaded_images[i]
            p = image_paths[i]
            self.org_faces.append(cur_img)

            if not no_align:
                align_img = align(cur_img, self.aligner)
                if align_img is None:
                    print("Find 0 face(s) in {}".format(p.split("/")[-1]))
                    self.images_without_face.append(i)
                    continue

                cur_faces = align_img[0]
            else:
                cur_faces = [cur_img]

            cur_faces = [face for face in cur_faces if face.shape[0] != 0 and face.shape[1] != 0]
            cur_shapes = [f.shape[:-1] for f in cur_faces]

            cur_faces_square = []
            if verbose and not no_align:
                print("Find {} face(s) in {}".format(len(cur_faces), p.split("/")[-1]))
            if eval_local:
                cur_faces = cur_faces[:1]

            for img in cur_faces:
                if eval_local:
                    base = resize(img, (IMG_SIZE, IMG_SIZE))
                else:
                    long_size = max([img.shape[1], img.shape[0]]) + self.margin

                    base = np.ones((long_size, long_size, 3)) * np.mean(img, axis=(0, 1))

                    start1, end1 = get_ends(long_size, img.shape[0])
                    start2, end2 = get_ends(long_size, img.shape[1])

                    base[start1:end1, start2:end2, :] = img
                    cur_start_end = (start1, end1, start2, end2)
                    self.start_end_ls.append(cur_start_end)

                cur_faces_square.append(base)
            cur_faces_square = [resize(f, (IMG_SIZE, IMG_SIZE)) for f in cur_faces_square]
            self.cropped_faces.extend(cur_faces_square)

            if not self.no_align:
                cur_index = align_img[1]
                self.cropped_faces_shape.extend(cur_shapes)
                self.cropped_index.extend(cur_index[:len(cur_faces_square)])
                self.callback_idx.extend([i] * len(cur_faces_square))

        if len(self.cropped_faces) == 0:
            return

        self.cropped_faces = np.array(self.cropped_faces)

        if preprocessing:
            self.cropped_faces = preprocess(self.cropped_faces, PREPROCESS)

        self.cloaked_faces = np.copy(self.org_faces)

    def merge_faces(self, protected_images, original_images):
        if self.no_align:
            return np.clip(protected_images, 0.0, 255.0), self.images_without_face

        self.cloaked_faces = np.copy(self.org_faces)

        for i in range(len(self.cropped_faces)):
            cur_protected = protected_images[i]
            cur_original = original_images[i]

            org_shape = self.cropped_faces_shape[i]

            old_square_shape = max([org_shape[0], org_shape[1]]) + self.margin

            cur_protected = resize(cur_protected, (old_square_shape, old_square_shape))
            cur_original = resize(cur_original, (old_square_shape, old_square_shape))

            start1, end1, start2, end2 = self.start_end_ls[i]

            reshape_cloak = cur_protected - cur_original
            reshape_cloak = reshape_cloak[start1:end1, start2:end2, :]

            callback_id = self.callback_idx[i]
            bb = self.cropped_index[i]
            self.cloaked_faces[callback_id][bb[0]:bb[2], bb[1]:bb[3], :] += reshape_cloak

        for i in range(0, len(self.cloaked_faces)):
            self.cloaked_faces[i] = np.clip(self.cloaked_faces[i], 0.0, 255.0)
        return self.cloaked_faces, self.images_without_face


def get_ends(longsize, window):
    start = (longsize - window) // 2
    end = start + window
    return start, end


def resize(img, sz):
    assert np.min(img) >= 0 and np.max(img) <= 255.0
    from keras.preprocessing import image
    im_data = image.array_to_img(img).resize((sz[1], sz[0]))
    im_data = image.img_to_array(im_data)
    return im_data


def init_gpu(gpu):
    ''' code to initialize gpu in tf2'''
    if isinstance(gpu, list):
        gpu_num = ','.join([str(i) for i in gpu])
    else:
        gpu_num = str(gpu)
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        print('GPU already initiated')
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_num
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[0], True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            print(e)


def preprocess(X, method):
    assert method in {'raw', 'imagenet', 'inception', 'mnist'}

    if method == 'raw':
        pass
    elif method == 'imagenet':
        X = imagenet_preprocessing(X)
    else:
        raise Exception('unknown method %s' % method)

    return X


def reverse_preprocess(X, method):
    assert method in {'raw', 'imagenet', 'inception', 'mnist'}

    if method == 'raw':
        pass
    elif method == 'imagenet':
        X = imagenet_reverse_preprocessing(X)
    else:
        raise Exception('unknown method %s' % method)

    return X


def imagenet_preprocessing(x, data_format=None):
    if data_format is None:
        data_format = image_data_format()
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
    x = np.array(x)
    if data_format is None:
        data_format = image_data_format()
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
    return reverse_preprocess(x, preprocess)


def load_extractor(name):
    hash_map = {"extractor_2": "ce703d481db2b83513bbdafa27434703",
                "extractor_0": "94854151fd9077997d69ceda107f9c6b"}
    assert name in ["extractor_2", 'extractor_0']
    model_file = pkg_resources.resource_filename("fawkes", "model/{}.h5".format(name))
    cur_hash = hash_map[name]
    model_dir = pkg_resources.resource_filename("fawkes", "model/")
    os.makedirs(model_dir, exist_ok=True)
    get_file("{}.h5".format(name), "http://mirror.cs.uchicago.edu/fawkes/files/{}.h5".format(name),
             cache_dir=model_dir, cache_subdir='', md5_hash=cur_hash)

    model = load_model(model_file)
    model = Extractor(model)
    return model


class Extractor(object):
    def __init__(self, model):
        self.model = model

    def predict(self, imgs):
        imgs = imgs / 255.0
        embeds = l2_norm(self.model(imgs))
        return embeds

    def __call__(self, x):
        return self.predict(x)


def dump_image(x, filename, format="png", scale=False):
    img = image.array_to_img(x, scale=scale)
    img.save(filename, format)
    return


def load_embeddings(feature_extractors_names):
    model_dir = os.path.join(os.path.expanduser('~'), '.fawkes')
    for extractor_name in feature_extractors_names:
        fp = gzip.open(os.path.join(model_dir, "{}_emb.p.gz".format(extractor_name)), 'rb')
        path2emb = pickle.load(fp)
        fp.close()

    return path2emb


def extractor_ls_predict(feature_extractors_ls, X):
    feature_ls = []
    for extractor in feature_extractors_ls:
        cur_features = extractor.predict(X)
        feature_ls.append(cur_features)
    concated_feature_ls = np.concatenate(feature_ls, axis=1)
    return concated_feature_ls


def pairwise_l2_distance(A, B):
    BT = B.transpose()
    vecProd = np.dot(A, BT)
    SqA = A ** 2
    sumSqA = np.matrix(np.sum(SqA, axis=1))
    sumSqAEx = np.tile(sumSqA.transpose(), (1, vecProd.shape[1]))

    SqB = B ** 2
    sumSqB = np.sum(SqB, axis=1)
    sumSqBEx = np.tile(sumSqB, (vecProd.shape[0], 1))
    SqED = sumSqBEx + sumSqAEx - 2 * vecProd
    SqED[SqED < 0] = 0.0
    ED = np.sqrt(SqED)
    return ED

def l2_norm(x, axis=1):
    """l2 norm"""
    norm = tf.norm(x, axis=axis, keepdims=True)
    output = x / norm
    return output


""" TensorFlow implementation get_file
https://github.com/tensorflow/tensorflow/blob/v2.3.0/tensorflow/python/keras/utils/data_utils.py#L168-L297
"""


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

    # fname = path_to_string(fname)

    if untar:
        untar_fpath = os.path.join(datadir, fname)
        fpath = untar_fpath + '.tar.gz'
    else:
        fpath = os.path.join(datadir, fname)

    download = False
    if os.path.exists(fpath):
        # File found; verify integrity if a hash was provided.
        if file_hash is not None:
            if not validate_file(fpath, file_hash, algorithm=hash_algorithm):
                print('A local file was found, but it seems to be '
                      'incomplete or outdated because the ' + hash_algorithm +
                      ' file hash does not match the original value of ' + file_hash +
                      ' so we will re-download the data.')
                download = True
    else:
        download = True

    if download:
        print('Downloading data from', origin)

        class ProgressTracker(object):
            # Maintain progbar for the lifetime of download.
            # This design was chosen for Python 2.7 compatibility.
            progbar = None

        def dl_progress(count, block_size, total_size):
            if ProgressTracker.progbar is None:
                if total_size == -1:
                    total_size = None
                ProgressTracker.progbar = Progbar(total_size)
            else:
                ProgressTracker.progbar.update(count * block_size)

        error_msg = 'URL fetch failure on {}: {} -- {}'
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
        ProgressTracker.progbar = None

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


def validate_file(fpath, file_hash, algorithm='auto', chunk_size=65535):
    """Validates a file against a sha256 or md5 hash.
    Arguments:
        fpath: path to the file being validated
        file_hash:  The expected hash string of the file.
            The sha256 and md5 hash algorithms are both supported.
        algorithm: Hash algorithm, one of 'auto', 'sha256', or 'md5'.
            The default 'auto' detects the hash algorithm in use.
        chunk_size: Bytes to read at a time, important for large files.
    Returns:
        Whether the file is valid
    """
    if (algorithm == 'sha256') or (algorithm == 'auto' and len(file_hash) == 64):
        hasher = 'sha256'
    else:
        hasher = 'md5'

    if str(_hash_file(fpath, hasher, chunk_size)) == str(file_hash):
        return True
    else:
        return False


def _hash_file(fpath, algorithm='sha256', chunk_size=65535):
    """Calculates a file sha256 or md5 hash.
    Example:
    ```python
    _hash_file('/path/to/file.zip')
    'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
    ```
    Arguments:
        fpath: path to the file being validated
        algorithm: hash algorithm, one of `'auto'`, `'sha256'`, or `'md5'`.
            The default `'auto'` detects the hash algorithm in use.
        chunk_size: Bytes to read at a time, important for large files.
    Returns:
        The file hash
    """
    if (algorithm == 'sha256') or (algorithm == 'auto' and len(hash) == 64):
        hasher = hashlib.sha256()
    else:
        hasher = hashlib.md5()

    with open(fpath, 'rb') as fpath_file:
        for chunk in iter(lambda: fpath_file.read(chunk_size), b''):
            hasher.update(chunk)

    return hasher.hexdigest()
