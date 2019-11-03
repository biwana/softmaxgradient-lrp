from keras.models import Model
from keras.layers import Input, Dense, multiply, concatenate, Activation, Lambda
from keras.layers import PReLU, LSTM
from keras.layers import Conv2D, BatchNormalization, Dropout
from keras.layers import MaxPooling2D, Flatten
from keras import backend as K


def LeNet(input_shape, nb_class):
    ip = Input(shape=input_shape)

    conv1 = Conv2D(64, (3,3), padding='same', kernel_initializer='he_uniform')(ip)
    conv1 = Activation('relu')(conv1)
    conv1 = MaxPooling2D(pool_size=(2,2))(conv1)

    conv2 = Conv2D(128, (3,3), padding='same', kernel_initializer='he_uniform')(conv1)
    conv2 = Activation('relu')(conv2)
    conv2 = MaxPooling2D(pool_size=(2,2))(conv2)

    conv3 = Conv2D(256, (3,3), padding='same', kernel_initializer='he_uniform')(conv2)
    conv3 = Activation('relu')(conv3)
    conv3 = MaxPooling2D(pool_size=(2,2))(conv3)

    flat = Flatten()(conv3)

    fc1 = Dense(512, activation='relu')(flat)
    fc1 = Dropout(0.5)(fc1)

    fc2 = Dense(512, activation='relu')(fc1)
    fc2 = Dropout(0.5)(fc2)

    out = Dense(nb_class, activation='softmax')(fc2)

    model = Model([ip], [out])

    model.summary()

    return model

def VGGMini(input_shape, nb_class):
    ip = Input(shape=input_shape)

    conv1 = Conv2D(64, (3,3), padding='same', kernel_initializer='he_uniform')(ip)
    conv1 = Activation('relu')(conv1)
    conv1 = Conv2D(64, (3,3), padding='same', kernel_initializer='he_uniform')(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = Conv2D(64, (3,3), padding='same', kernel_initializer='he_uniform')(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = MaxPooling2D(pool_size=(2,2))(conv1)

    conv2 = Conv2D(128, (3,3), padding='same', kernel_initializer='he_uniform')(conv1)
    conv2 = Activation('relu')(conv2)
    conv2 = Conv2D(128, (3,3), padding='same', kernel_initializer='he_uniform')(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = Conv2D(128, (3,3), padding='same', kernel_initializer='he_uniform')(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = MaxPooling2D(pool_size=(2,2))(conv2)

    flat = Flatten()(conv2)

    fc1 = Dense(4096, activation='relu')(flat)
    fc1 = Dropout(0.5)(fc1)

    fc2 = Dense(4096, activation='relu')(fc1)
    fc2 = Dropout(0.5)(fc2)

    out = Dense(nb_class, activation='softmax')(fc2)

    model = Model([ip], [out])

    model.summary()

    return model

def VGGMini2(input_shape, nb_class):
    ip = Input(shape=input_shape)

    conv1 = Conv2D(64, (3,3), padding='same', kernel_initializer='he_uniform')(ip)
    conv1 = Activation('relu')(conv1)
    conv1 = Conv2D(64, (3,3), padding='same', kernel_initializer='he_uniform')(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = MaxPooling2D(pool_size=(2,2))(conv1)

    conv2 = Conv2D(128, (3,3), padding='same', kernel_initializer='he_uniform')(conv1)
    conv2 = Activation('relu')(conv2)
    conv2 = Conv2D(128, (3,3), padding='same', kernel_initializer='he_uniform')(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = MaxPooling2D(pool_size=(2,2))(conv2)

    flat = Flatten()(conv2)

    fc1 = Dense(1024, activation='relu')(flat)
    fc1 = Dropout(0.5)(fc1)

    fc2 = Dense(1024, activation='relu')(fc1)
    fc2 = Dropout(0.5)(fc2)

    out = Dense(nb_class, activation='softmax')(fc2)

    model = Model([ip], [out])

    model.summary()

    return model

def VGGMini2_nodrop(input_shape, nb_class):
    ip = Input(shape=input_shape)

    conv1 = Conv2D(64, (3,3), padding='same', activation='relu', kernel_initializer='he_uniform')(ip)
    conv1 = Conv2D(64, (3,3), padding='same', activation='relu', kernel_initializer='he_uniform')(conv1)
    conv1 = MaxPooling2D(pool_size=(2,2))(conv1)

    conv2 = Conv2D(128, (3,3), padding='same', activation='relu', kernel_initializer='he_uniform')(conv1)
    conv2 = Conv2D(128, (3,3), padding='same', activation='relu', kernel_initializer='he_uniform')(conv2)
    conv2 = MaxPooling2D(pool_size=(2,2))(conv2)

    flat = Flatten()(conv2)

    fc1 = Dense(1024, activation='relu')(flat)

    fc2 = Dense(1024, activation='relu')(fc1)

    out = Dense(nb_class, activation='softmax')(fc2)

    model = Model([ip], [out])

    model.summary()

    return model


def VGG10_nodrop(input_shape, nb_class):
    ip = Input(shape=input_shape)

    conv1 = Conv2D(64, (3,3), padding='same', activation='relu', kernel_initializer='he_uniform')(ip)
    conv1 = Conv2D(64, (3,3), padding='same', activation='relu', kernel_initializer='he_uniform')(conv1)
    conv1 = MaxPooling2D(pool_size=(2,2))(conv1)

    conv2 = Conv2D(128, (3,3), padding='same', activation='relu', kernel_initializer='he_uniform')(conv1)
    conv2 = Conv2D(128, (3,3), padding='same', activation='relu', kernel_initializer='he_uniform')(conv2)
    conv2 = MaxPooling2D(pool_size=(2,2))(conv2)
    
    conv3 = Conv2D(256, (3,3), padding='same', activation='relu', kernel_initializer='he_uniform')(conv2)
    conv3 = Conv2D(256, (3,3), padding='same', activation='relu', kernel_initializer='he_uniform')(conv3)
    conv3 = Conv2D(256, (3,3), padding='same', activation='relu', kernel_initializer='he_uniform')(conv3)
    conv3 = MaxPooling2D(pool_size=(2,2))(conv3)

    flat = Flatten()(conv3)

    fc1 = Dense(1024, activation='relu')(flat)

    fc2 = Dense(1024, activation='relu')(fc1)

    out = Dense(nb_class, activation='softmax')(fc2)

    model = Model([ip], [out])

    model.summary()

    return model

def preprocess_input(input_imgs, img_rows, img_cols, channels):
    if K.image_data_format() == 'channels_first':
        ret = input_imgs.reshape(input_imgs.shape[0], channels, img_rows, img_cols)
    else:
        ret = input_imgs.reshape(input_imgs.shape[0], img_rows, img_cols, channels)

    ret = ret.astype('float32')
    ret /= 127.5
    ret -= 1.
    return ret
    