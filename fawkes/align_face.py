import numpy as np
from mtcnn import MTCNN


def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret


def aligner():
    return MTCNN(min_face_size=30)


def align(orig_img, aligner):
    """ run MTCNN face detector """

    if orig_img.ndim < 2:
        return None
    if orig_img.ndim == 2:
        orig_img = to_rgb(orig_img)
    orig_img = orig_img[:, :, 0:3]

    detect_results = aligner.detect_faces(orig_img)
    cropped_arr = []
    bounding_boxes_arr = []
    for dic in detect_results:
        if dic['confidence'] < 0.9:
            continue
        x, y, width, height = dic['box']
        if width < 30 or height < 30:
            continue
        bb = [y, x, y + height, x + width]
        cropped = orig_img[bb[0]:bb[2], bb[1]:bb[3], :]
        cropped_arr.append(np.copy(cropped))
        bounding_boxes_arr.append(bb)
    return cropped_arr, bounding_boxes_arr

    # if nrof_faces > 0:
    #     det = bounding_boxes[0]['box']
    #     det_arr = []
    #     img_size = np.asarray(orig_img.shape)[0:2]
    #     if nrof_faces > 1:
    #         margin = margin / 1.5
    #         if detect_multiple_faces:
    #             for i in range(nrof_faces):
    #                 det_arr.append(np.squeeze(bounding_boxes[i]['box']))
    #         else:
    #             bounding_box_size = (det[1] + det[3])
    #             img_center = img_size / 2
    #             offsets = np.vstack([(det[0] + det[2]) / 2 - img_center[1],
    #                                  (det[1] + det[3]) / 2 - img_center[0]])
    #             offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
    #             index = np.argmax(bounding_box_size - offset_dist_squared * 2.0)  # some extra weight on the centering
    #             det_arr.append(det[index, :])
    #     else:
    #         det_arr.append(np.squeeze(det))
    #
    #     cropped_arr = []
    #     bounding_boxes_arr = []
    #     for i, det in enumerate(det_arr):
    #         det = np.squeeze(det)
    #         bb = np.zeros(4, dtype=np.int32)
    #         # add in margin
    #         marg1 = int((det[2] - det[0]) * margin)
    #         marg2 = int((det[3] - det[1]) * margin)
    #
    #         bb[0] = max(det[0] - marg1 / 2, 0)
    #         bb[1] = max(det[1] - marg2 / 2, 0)
    #         bb[2] = min(det[0] + det[2] + marg1 / 2, img_size[0])
    #         bb[3] = min(det[1] + det[3] + marg2 / 2, img_size[1])
    #         cropped = orig_img[bb[0]:bb[2], bb[1]: bb[3], :]
    #         cropped_arr.append(cropped)
    #         bounding_boxes_arr.append([bb[0], bb[1], bb[2], bb[3]])
    #     return cropped_arr, bounding_boxes_arr
    # else:
    #     return None
