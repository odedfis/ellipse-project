from pathlib import Path
from utils import normalize_data
from keras import callbacks
from keras_tqdm import TQDMNotebookCallback
from datetime import datetime
from keras.models import load_model

from model import FullyConnected
from model import Model
import os
import data_preprocessing
from data_gen import DataGenerator
from metrics import *
from keras.optimizers import Adam, SGD
from model import LIGHT, FullyConnected
train_image_path = './images/train/'
test_image_path = './images/test/'

target_path = './train_data.txt'

time_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
os.makedirs("./models/" + time_stamp)
filepath = "./models/" + time_stamp + \
           "/saved-model-{epoch:02d}-{val_classifier_accuracy:.2f}.hdf5"
checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_classifier_accuracy',
                                       verbose=1, save_best_only=False, mode='max')

BATCH = 32
file_names_train, gt_train, n_train = data_preprocessing.load_metadata('train')
file_names_test, gt_test, n_test = data_preprocessing.load_metadata('test')


gt_train, gt_test, shape_mean, shape_std = normalize_data(gt_train, gt_test)
training_generator = DataGenerator(list(range(len(file_names_train))), file_names_train, train_image_path, gt_train,
                                   dim=(50, 50), batch_size=BATCH, shuffle=True)

validation_generator = DataGenerator(list(range(len(file_names_test))), file_names_test, test_image_path, gt_test,
                                     dim=(50, 50), batch_size=BATCH, shuffle=False)
model = FullyConnected()
epochs = 1000
opt = Adam()

reduce_lr = callbacks.ReduceLROnPlateau(monitor='ellipse_loss', factor=0.2, patience=2, min_lr=0.0001)

# model = load_model('models/20191123-151830/saved-model-22-0.69.hdf5',
#                    custom_objects={'ellipse_loss': ellipse_loss,
#                                    'ellipse_detector_accuracy': ellipse_detector_accuracy,
#                                    'ellipse_angle_accuracy': ellipse_angle_accuracy,
#                                    'ellipse_shape_error': ellipse_shape_error})

model.compile(loss=ellipse_loss,
              optimizer=opt,
              metrics=[classifier_accuracy,
                       angle_accuracy,
                       shape_error,
                       ellipse_loss,
                       ])

tb_callback = callbacks.TensorBoard(log_dir='./logs/%s' % datetime.now().strftime("%Y%m%d-%H%M%S"),
                                    histogram_freq=0, batch_size=BATCH,
                                    write_graph=True, write_grads=False,
                                    write_images=True, embeddings_freq=0,
                                    embeddings_layer_names=None,
                                    embeddings_metadata=None)

model.fit_generator(generator=training_generator,
                    steps_per_epoch=len(training_generator),
                    epochs=epochs,
                    verbose=1,
                    initial_epoch=0,
                    use_multiprocessing=True,
                    validation_data=validation_generator,
                    callbacks=[reduce_lr, checkpoint, tb_callback, TQDMNotebookCallback()])


