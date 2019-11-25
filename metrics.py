from keras import losses
from keras import metrics
from keras import backend as K
from utils import GT_INDEX
import tensorflow as tf
from keras.optimizers import Adam


def ellipse_loss(y_true, y_pred):

    is_ellipse = y_true[..., GT_INDEX.IS_ELLIPSE]

    is_ellipse_loss = focal_loss(is_ellipse, y_pred[..., GT_INDEX.IS_ELLIPSE])

    angle_loss = losses.categorical_crossentropy(y_true[..., GT_INDEX.ANGLE_BIN_BEG:GT_INDEX.ANGLE_BIN_END],
                                                 y_pred[..., GT_INDEX.ANGLE_BIN_BEG:GT_INDEX.ANGLE_BIN_END])

    shape_l1 = K.abs(y_true[:, GT_INDEX.SHAPE_BEG:GT_INDEX.SHAPE_END] -
                     y_pred[:, GT_INDEX.SHAPE_BEG:GT_INDEX.SHAPE_END])

    shape_smooth_l1_loss = K.mean(K.switch(K.less(shape_l1, 1),
                                           0.5 * shape_l1 ** 2,
                                           shape_l1 - 0.5), axis=-1)

    # shape_smooth_l1_loss = K.print_tensor(shape_smooth_l1_loss, "shape_smooth_l1_loss: ")

    is_ellipse_bool = K.equal(is_ellipse, 1)

    zeros = K.zeros_like(is_ellipse)

    shape_loss = K.sum(K.switch(is_ellipse_bool, shape_smooth_l1_loss, zeros)) / (K.sum(is_ellipse) + K.epsilon())

    angle_loss_total = K.sum(K.switch(is_ellipse_bool, angle_loss, zeros)) / (K.sum(is_ellipse) + K.epsilon())

    return 10 * is_ellipse_loss + 5 * shape_loss + angle_loss_total


def classifier_accuracy(y_true, y_pred):
    return metrics.binary_accuracy(y_true[..., GT_INDEX.IS_ELLIPSE], y_pred[..., GT_INDEX.IS_ELLIPSE])


def focal_loss(y_true, y_pred, gamma=2., alpha=.25):
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - \
           K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))


def angle_accuracy(y_true, y_pred):
    is_ellipse = y_true[..., GT_INDEX.IS_ELLIPSE]
    is_ellipse_bool = K.equal(is_ellipse, 1)

    cls_gt = K.argmax(y_true[..., GT_INDEX.ANGLE_BIN_BEG: GT_INDEX.ANGLE_BIN_END], axis=-1)
    cls_pred = K.argmax(y_pred[..., GT_INDEX.ANGLE_BIN_BEG: GT_INDEX.ANGLE_BIN_END], axis=-1)

    same_bin = K.equal(cls_gt, cls_pred)
    same_bin = K.cast(same_bin, K.floatx())
    return K.sum(K.switch(is_ellipse_bool, same_bin, K.zeros_like(same_bin))) / (K.sum(is_ellipse) + K.epsilon())


def shape_error(y_true, y_pred):
    is_ellipse = y_true[..., GT_INDEX.IS_ELLIPSE]
    sad = K.sum(K.abs(y_true[..., GT_INDEX.SHAPE_BEG: GT_INDEX.SHAPE_END] -
                      y_pred[..., GT_INDEX.SHAPE_BEG: GT_INDEX.SHAPE_END]), axis=-1)

    return K.sum(K.switch(is_ellipse, sad, K.zeros_like(sad))) / (K.sum(is_ellipse) + K.epsilon())


if __name__ == '__main__':
    from model import Model
    model = Model()
    model.compile(loss=ellipse_loss,
                  optimizer=Adam(),
                  metrics=[classifier_accuracy, angle_accuracy, shape_error])

