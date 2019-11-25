from enum import IntEnum
from keras import utils
import numpy as np
from tqdm import tqdm
import os.path as osp
import cv2
import matplotlib.pyplot as plt

'''angle binning'''

def ang_to_bin(ang):
    # return the bin index for the specified angle
    return np.floor_divide(np.mod(float(ang), 180), 180 / GT_INDEX.ANGLE_BINS)


def bin_to_ang(bin_ind):
    # return the angle for the specified bin
    return np.argmax(bin_ind, axis=-1) * 180 / GT_INDEX.ANGLE_BINS


def denormalize_shape(ellipses, shape_mean, shape_std):
    '''
    denormalizing center and sizes to original size
    '''
    ellipses[:, GT_INDEX.SHAPE_BEG:GT_INDEX.SHAPE_END] = ellipses[:, GT_INDEX.SHAPE_BEG:GT_INDEX.SHAPE_END] * \
                                                         shape_std + shape_mean

    ellipses[:, GT_INDEX.SHAPE_BEG:GT_INDEX.SHAPE_END] = np.around(ellipses[:, GT_INDEX.SHAPE_BEG:GT_INDEX.SHAPE_END])
    return ellipses


def normalize_data(gt_train, gt_test):
    '''
    normalize center by reducing the mean and divide by std
    '''
    is_ellipse = np.nonzero(gt_train[:, GT_INDEX.IS_ELLIPSE] == 1)
    shape_mean = np.mean(np.squeeze(gt_train[is_ellipse, GT_INDEX.SHAPE_BEG: GT_INDEX.SHAPE_END]), axis=0)
    shape_std = np.std(np.squeeze(gt_train[is_ellipse, GT_INDEX.SHAPE_BEG: GT_INDEX.SHAPE_END]), axis=0)

    # normalize train and test data
    gt_train[:, GT_INDEX.SHAPE_BEG: GT_INDEX.SHAPE_END] = (gt_train[:, GT_INDEX.SHAPE_BEG: GT_INDEX.SHAPE_END] -
                                                           shape_mean) / shape_std
    gt_test[:, GT_INDEX.SHAPE_BEG: GT_INDEX.SHAPE_END] = (gt_test[:, GT_INDEX.SHAPE_BEG: GT_INDEX.SHAPE_END] -
                                                          shape_mean) / shape_std

    return gt_train, gt_test, shape_mean, shape_std


class GT_INDEX(IntEnum):
    IS_ELLIPSE = 0
    SHAPE_BEG = 1
    X = 1
    Y = 2
    MAJOR_AXIS = 3
    MINOR_AXIS = 4
    SHAPE_END = 5
    ANGLE_BINS = 8
    ANGLE_BIN_BEG = SHAPE_END
    ANGLE_BIN_END = ANGLE_BIN_BEG + ANGLE_BINS
    ANGLE = ANGLE_BIN_END
    ELLIPSE_END = ANGLE + 1


def load_metadata(split):
    with open('./images/%s_data.txt' % split) as f:
        gt_lines = f.readlines()

    # get file names (exclude file header)
    file_name_len = (21 if split == 'train' else 20)
    file_names = [osp.basename(gt_line[:file_name_len])
                  for gt_line in gt_lines[1:] if gt_line.strip()]
    n = len(file_names)

    # map file-lines into ellipses
    gt = np.zeros((n, GT_INDEX.ELLIPSE_END))
    for i, gt_line in tqdm(enumerate(gt_lines[1:n])):
        # note - not very robust but file is assumed to be 100% valid
        gt_fields = [field.strip() for field in gt_line[file_name_len:].split(',')]

        if gt_fields[0] == 'True':
            gt[i, GT_INDEX.IS_ELLIPSE] = 1

            gt[i, GT_INDEX.SHAPE_BEG:GT_INDEX.SHAPE_END] = [float(gt_fields[field_idx])
                                                            for field_idx in [1, 2, 4, 5]]
            gt[i, GT_INDEX.ANGLE] = float(gt_fields[3]) % 180

            # make HA1 the length of the *long* half-axis and HA2 the log of the
            # ratio between the shorter half-axis to the long one
            if gt[i, GT_INDEX.MINOR_AXIS] > gt[i, GT_INDEX.MAJOR_AXIS]:
                gt[i, GT_INDEX.MAJOR_AXIS], gt[i, GT_INDEX.MINOR_AXIS] = gt[i, GT_INDEX.MINOR_AXIS], gt[i, GT_INDEX.MAJOR_AXIS]

                # if we are "swapping" axes, we need to "rotate" by 90 so that the new representation is the same
                # ellipse
                gt[i, GT_INDEX.ANGLE] = (gt[i, GT_INDEX.ANGLE] + 90) % 180

            # encode bin as a one-hot vector
            gt[i, GT_INDEX.ANGLE_BIN_BEG:GT_INDEX.ANGLE_BIN_END] = \
                utils.to_categorical(ang_to_bin(gt[i, GT_INDEX.ANGLE]), num_classes=GT_INDEX.ANGLE_BINS)
        else:
            gt[i, GT_INDEX.ANGLE_BIN_BEG] = 1  # ensure valid dummy probability

    return file_names, gt, n


def draw_ellipse(img, ellipse, shape_mean=None, shape_std=None, denormalize=False, use_ang_bin=False):
    '''
    draw an ellipse on image

      ellipse - parameterization of ellipse, if denormalize=True then this should be our
         representation, if False then this should be the original representation in the
         dataset text file

      use_ang_bin - whether to use the angle encoded in the one-hot vector or the angle
         parameter we carry along for debugging
    '''
    if ellipse[GT_INDEX.IS_ELLIPSE] < 0.5:
        return img
    if use_ang_bin:
        ang = bin_to_ang(ellipse[GT_INDEX.ANGLE_BIN_BEG: GT_INDEX.ANGLE_BIN_END])
    else:
        ang = ellipse[GT_INDEX.ANGLE]

    if denormalize:
        ellipse = np.copy(ellipse)
        ellipse[GT_INDEX.SHAPE_BEG:GT_INDEX.SHAPE_END] = ellipse[GT_INDEX.SHAPE_BEG:GT_INDEX.SHAPE_END] * shape_std \
                                                         + shape_mean
        # ellipse[GT_MAPPING.MINOR_AXIS] = np.exp(ellipse[GT_MAPPING.MINOR_AXIS]) * ellipse[GT_MAPPING.MAJOR_AXIS]
        ellipse[GT_INDEX.SHAPE_BEG:GT_INDEX.SHAPE_END] = np.around(ellipse[GT_INDEX.SHAPE_BEG:GT_INDEX.SHAPE_END])

    cv2.ellipse(img, center=(int(ellipse[GT_INDEX.X]), int(ellipse[GT_INDEX.Y])),
                axes=(int(ellipse[GT_INDEX.MAJOR_AXIS]), int(ellipse[GT_INDEX.MINOR_AXIS])),
                angle=ang, startAngle=0, endAngle=360, color=(1.0, 0, 0), thickness=1)

    return img


def picshow(img):
    num = len(img)
    ax = np.ceil(np.sqrt(num))
    ay = np.rint(np.sqrt(num))
    fig =plt.figure()
    for i in range(1,num+1):
        sub = fig.add_subplot(ax,ay,i)
        # sub.set_title("titre", i)
        sub.axis('off')
        sub.imshow(img[i-1])
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    file_names_train, gt_train, n_train = load_metadata('train')
    file_names_test, gt_test, n_test = load_metadata('test')
    print('Ellipse meta-data loaded')