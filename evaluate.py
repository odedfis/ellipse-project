from keras.models import load_model
from metrics import *
from utils import load_metadata
from data_gen import DataGenerator
from utils import draw_ellipse, normalize_data
import cv2
import numpy as np
from utils import picshow


BATCH = 32

image_path = './images/train/'
target_path = './test_data.txt'


file_names_train, gt_train, n_train = load_metadata('train')
file_names_test, gt_test, n_test = load_metadata('test')
gt_train, gt_test, shape_mean, shape_std = normalize_data(gt_train, gt_test)
model = load_model('models/20191123-211844/saved-model-04-0.69.hdf5',
                   custom_objects={'ellipse_loss': ellipse_loss,
                                   'classifier_accuracy': classifier_accuracy,
                                   'angle_accuracy': angle_accuracy,
                                   'shape_error': shape_error})

out = []
for i in range(121):
    sample = cv2.imread(image_path + file_names_test[i])[np.newaxis, :]
    # sample = cv2.cvtColor(im_sample, cv2.COLOR_BGR2GRAY)
    sample = sample.astype('float32')
    sample = sample / 255

    # sample = sample[np.newaxis, :, :, np.newaxis]
    y = model.predict(sample)
    print(y[0])
    img = draw_ellipse(sample[0, ...], y[0], shape_mean=shape_mean, shape_std=shape_std, denormalize=True, use_ang_bin=True)
    out.append(img)

picshow(out)


