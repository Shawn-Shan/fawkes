"""Performs face alignment and stores face thumbnails in the output directory."""
# MIT License
#
# Copyright (c) 2016 David Sandberg
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

""" Tensorflow implementation of the face detection / alignment algorithm found at
https://github.com/kpzhang93/MTCNN_face_detection_alignment
"""


import numpy as np
from fawkes import create_mtcnn, run_detect_face

np_load_old = np.load
np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)


def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret


def aligner(sess):
    pnet, rnet, onet = create_mtcnn(sess, None)
    return [pnet, rnet, onet]


def align(orig_img, aligner, margin=0.8, detect_multiple_faces=True):
    pnet, rnet, onet = aligner
    minsize = 25  # minimum size of face
    threshold = [0.85, 0.85, 0.85]  # three steps's threshold
    factor = 0.709  # scale factor

    if orig_img.ndim < 2:
        return None
    if orig_img.ndim == 2:
        orig_img = to_rgb(orig_img)
    orig_img = orig_img[:, :, 0:3]

    bounding_boxes, _ = run_detect_face(orig_img, minsize, pnet, rnet, onet, threshold, factor)
    nrof_faces = bounding_boxes.shape[0]
    if nrof_faces > 0:
        det = bounding_boxes[:, 0:4]
        det_arr = []
        img_size = np.asarray(orig_img.shape)[0:2]
        if nrof_faces > 1:
            margin = margin / 1.5
            if detect_multiple_faces:
                for i in range(nrof_faces):
                    det_arr.append(np.squeeze(det[i]))
            else:
                bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
                img_center = img_size / 2
                offsets = np.vstack([(det[:, 0] + det[:, 2]) / 2 - img_center[1],
                                     (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
                offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
                index = np.argmax(bounding_box_size - offset_dist_squared * 2.0)  # some extra weight on the centering
                det_arr.append(det[index, :])
        else:
            det_arr.append(np.squeeze(det))
        cropped_arr = []
        bounding_boxes_arr = []
        for i, det in enumerate(det_arr):
            det = np.squeeze(det)
            bb = np.zeros(4, dtype=np.int32)
            side_1 = int((det[2] - det[0]) * margin)
            side_2 = int((det[3] - det[1]) * margin)

            bb[0] = np.maximum(det[0] - side_1 / 2, 0)
            bb[1] = np.maximum(det[1] - side_1 / 2, 0)
            bb[2] = np.minimum(det[2] + side_2 / 2, img_size[1])
            bb[3] = np.minimum(det[3] + side_2 / 2, img_size[0])
            cropped = orig_img[bb[1]:bb[3], bb[0]:bb[2], :]
            cropped_arr.append(cropped)
            bounding_boxes_arr.append([bb[0], bb[1], bb[2], bb[3]])
        return cropped_arr, bounding_boxes_arr
    else:
        return None
