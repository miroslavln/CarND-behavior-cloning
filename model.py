import PIL
import cv2
import os
import csv
import numpy as np
from PIL import Image
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from tqdm import tqdm
from keras.engine import Input
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, Dropout, Flatten, Dense, MaxPooling2D, ELU, Lambda, BatchNormalization
from keras.utils import np_utils
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
import keras
import tensorflow as tf
tf.python.control_flow_ops = tf

original_w, original_h = 320, 160
w, h = 64, 64
adjustment_angle = 0.08


def load_image(folder, image_file):
    image_path = os.path.join(folder, image_file.strip())
    image = Image.open(image_path)
    image = np.array(image, np.uint8)
    return image


def pre_process_image(image, train=True):
    image = image[60:-10]
    #cv2.imshow('image', image)
    #cv2.waitKey()

    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if train:
        white = np.zeros((h, w, 3), np.uint8) + 255
        image = cv2.addWeighted(image, 1, white, np.random.uniform() - 0.5, 0)

    #cv2.imshow('image', image)
    #cv2.waitKey()
    return image.astype(np.float32) / 255.0



def load_data(folder):
    file = os.path.join(folder, 'driving_log.csv')
    images, angles = [], []

    def add_image_and_angle(image, angle):
        images.append(image)
        angles.append(angle)

    with open(file, 'r') as csvfile:
        data_reader = csv.reader(csvfile, delimiter=',')
        next(data_reader, None)  #skip header
        for row in tqdm(data_reader):
            center_img, left_img, right_img, angle, throttle, brk, speed = row
            angle = float(angle)

            add_image_and_angle(center_img, angle)
            add_image_and_angle(left_img, angle + adjustment_angle)
            add_image_and_angle(right_img, angle - adjustment_angle)

    return images, angles


def get_model():
    model = Sequential()
    model.add(BatchNormalization(input_shape=(h, w, 3)))
    model.add(Conv2D(24, 3, 3, subsample=(1, 1), border_mode='valid'))
    model.add(ELU())
    model.add(MaxPooling2D((2, 2),  strides=(2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(32, 3, 3, subsample=(1, 1), border_mode='valid'))
    model.add(ELU())
    model.add(MaxPooling2D((2, 2),  strides=(2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(64, 3, 3, subsample=(1, 1), border_mode='valid'))
    model.add(ELU())
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(64, 3, 3, subsample=(1, 1), border_mode='valid'))
    model.add(ELU())
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(ELU())
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='tanh'))

    optimizer = Adam(lr=1e-4)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mean_squared_error'])
    return model


def data_generator(folder, images, angles, batch_size=64, change_color=True):
    image_batch = np.zeros((batch_size, h, w, 3))
    angles_batch = np.zeros(batch_size)
    while True:
        for i in range(batch_size):
            idx = np.random.randint(len(images))
            image = load_image(folder, images[idx])
            image = pre_process_image(image, change_color)

            if np.random.random() <= 0.5:
                image_batch[i] = image
                angles_batch[i] = angles[idx]
            else:
                image_batch[i] = cv2.flip(image, 1)
                angles_batch[i] = -angles[idx]

        yield image_batch, angles_batch


if __name__ == '__main__':
    folder = 'data/'
    batch_size = 64
    images, angles = load_data(folder)
    X_train, X_val, y_train, y_val = train_test_split(images, angles, test_size=0.2)

    model = get_model()

    #model.load_weights('model/model.h5')
    keras.backend.get_session().run(tf.global_variables_initializer())

    model.fit_generator(data_generator(folder, X_train, y_train, batch_size, change_color=False), nb_epoch=15,
                        samples_per_epoch=2*len(X_train),
                        validation_data=data_generator(folder, X_val, y_val, batch_size=batch_size, change_color=False),
                        nb_val_samples=len(X_val))

    model.save_weights('model/model.h5', overwrite=True)
    with open('model/model.json', 'w') as json_file:
        model_json = model.to_json()
        json_file.write(model_json)





