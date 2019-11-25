
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten, Concatenate, Add
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.regularizers import l2
from keras import backend as K
from utils import GT_INDEX


from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten, Concatenate, Add
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.regularizers import l2
from keras import backend as K


IMAGE_SZ = 50
REG = 0.001
def FullyConnected():
    inputs = Input(shape=(IMAGE_SZ, IMAGE_SZ, 3,))
    # 50x50x3 rf=1x1
    hidden1_num_units = 1024
    hidden2_num_units = 1024
    hidden3_num_units = 1024
    hidden4_num_units = 1024
    hidden5_num_units = 1024

    x = Flatten()(inputs)

    x = Dense(output_dim=hidden1_num_units, input_dim=IMAGE_SZ * IMAGE_SZ, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(output_dim=hidden2_num_units, input_dim=hidden1_num_units, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(output_dim=hidden3_num_units, input_dim=hidden2_num_units, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(output_dim=hidden4_num_units, input_dim=hidden3_num_units, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(output_dim=hidden5_num_units, input_dim=hidden4_num_units, activation='relu')(x)

    x = Dense(512)(x)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x_is_ellipse = Dense(64)(x)
    x = Dropout(0.3)(x)
    x_is_ellipse = Dense(1)(x_is_ellipse)
    y_is_ellipse = Activation('sigmoid')(x_is_ellipse)

    x_shape = Dense(512)(x)
    x_shape = Dropout(0.2)(x_shape)
    y_shape = Dense(4)(x_shape)

    x_angle_bin = Dense(GT_INDEX.ANGLE_BINS, kernel_regularizer=l2(REG))(x)
    y_angle_bin = Activation('softmax')(x_angle_bin)

    y = Concatenate(axis=1)([y_is_ellipse, y_shape, y_angle_bin])

    return Model(inputs=inputs, outputs=y)


def LIGHT():
    inputs = Input(shape=(IMAGE_SZ, IMAGE_SZ, 3,))
    x = Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(REG))(inputs)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(REG))(x)
    x = Activation('relu')(x)

    # x = MaxPooling2D(padding='same')(x)

    # x = Conv2D(128, (3, 3), padding='same')(x)
    # # 50x50x32 rf=3x3
    # x = BatchNormalization(axis=-1)(x)
    # x = Activation('relu')(x)
    # x = Conv2D(128, (3, 3), padding='same')(x)
    # # 25x25x64 rf=8x8
    # x = BatchNormalization(axis=-1)(x)
    # x = Activation('relu')(x)
    # x = MaxPooling2D(padding='same')(x)
    #
    # x = Conv2D(256, (3, 3), padding='same')(x)
    # # 50x50x32 rf=3x3
    # x = BatchNormalization(axis=-1)(x)
    # x = Activation('relu')(x)
    # x = Conv2D(256, (3, 3), padding='same')(x)
    # # 50x50x32 rf=3x3
    # x = BatchNormalization(axis=-1)(x)
    # x = Activation('relu')(x)
    # x = Conv2D(256, (3, 3), padding='same')(x)
    # # 25x25x64 rf=8x8
    # x = BatchNormalization(axis=-1)(x)
    # x = Activation('relu')(x)
    # x = MaxPooling2D(padding='same')(x)
    #
    # x = Conv2D(512, (3, 3), padding='same')(x)
    # # 50x50x32 rf=3x3
    # x = BatchNormalization(axis=-1)(x)
    # x = Activation('relu')(x)
    # x = Conv2D(512, (3, 3), padding='same')(x)
    # # 25x25x64 rf=8x8
    # x = BatchNormalization(axis=-1)(x)
    # x = Activation('relu')(x)
    # x = MaxPooling2D(padding='same')(x)
    #
    # x = Conv2D(512, (3, 3), padding='same')(x)
    # x = BatchNormalization(axis=-1)(x)
    # x = Activation('relu')(x)
    #
    x = Flatten()(x)

    x = Dense(512, kernel_regularizer=l2(REG))(x)
    x = Dropout(0.3)(x)
    x = Activation('relu')(x)

    # classification head
    x_is_ellipse = Dense(64)(x)
    x_is_ellipse = Dropout(0.25)(x_is_ellipse)
    x_is_ellipse = Dense(1)(x_is_ellipse)
    y_is_ellipse = Activation('sigmoid')(x_is_ellipse)

    # shape regression head
    x_shape = Dense(512)(x)
    x_shape = Dropout(0.25)(x_shape)
    y_shape = Dense(4)(x_shape)

    # angle classification (regression) head
    x_angle_bin = Dense(GT_INDEX.ANGLE_BINS)(x)
    x_angle_bin = Dropout(0.25)(x_angle_bin)
    y_angle_bin = Activation('softmax')(x_angle_bin)

    y = Concatenate(axis=1)([y_is_ellipse, y_shape, y_angle_bin])


    return Model(inputs=inputs, outputs=y)