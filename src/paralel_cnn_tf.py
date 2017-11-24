import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pims
import scipy
import scipy.misc
import tensorflow as tf
from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.applications.resnet50 import preprocess_input as preprocess_input_resnet50

from tensorflow.python.keras.applications import VGG16
from tensorflow.python.keras.applications.vgg16 import preprocess_input as preprocess_input_vgg16

from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler
from tensorflow.python.keras.layers import Dense, GlobalMaxPooling1D, GlobalAveragePooling1D
from tensorflow.python.keras.layers import GlobalMaxPooling2D, GlobalAveragePooling2D, AveragePooling2D
from tensorflow.python.keras.layers import GlobalMaxPooling3D, GlobalAveragePooling3D
from tensorflow.python.keras.layers import Input, Lambda, Reshape, Flatten, TimeDistributed, Dropout
from tensorflow.python.keras.layers import concatenate
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.regularizers import l1
from tensorflow.python.keras.optimizers import SGD, Adam, RMSprop
from sklearn.model_selection import train_test_split

from multiprocessing.pool import ThreadPool
from collections import namedtuple

import config
import utils

from tensorflow.python.keras import backend as K

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
# tf_config.gpu_options.visible_device_list = "0"
K.set_session(tf.Session(config=tf_config))

INPUT_ROWS = 202
INPUT_COLS = 360
NB_FRAMES = 8
INPUT_SHAPE = (NB_FRAMES, INPUT_ROWS, INPUT_COLS, 3)
VIDEO_FPS = 24

CLASSES = ['bird', 'blank', 'cattle', 'chimpanzee', 'elephant', 'forest buffalo', 'gorilla', 'hippopotamus', 'human',
           'hyena', 'large ungulate', 'leopard', 'lion', 'other (non-primate)', 'other (primate)', 'pangolin',
           'porcupine', 'reptile', 'rodent', 'small antelope', 'small cat', 'wild dog', 'duiker', 'hog']
NB_CLASSES = len(CLASSES)


# def build_model_resnet50(lock_base_model: True):
#     img_input = Input(INPUT_SHAPE, name='data')
#     base_model = ResNet50(input_shape=INPUT_SHAPE[1:], include_top=False, pooling=None)
#     if lock_base_model:
#         for layer in base_model.layers:
#             layer.trainable = False
#
#     base_model_updated = Model(inputs=base_model.inputs, outputs=base_model.layers[-2].output)
#     base_model_updated.summary()
#
#     results = []
#     for i in range(NB_FRAMES):
#         crop = Lambda(lambda x: x[:, i])(img_input)
#         res = base_model_updated(crop)
#         x = AveragePooling2D((3, 3), strides=1)(res)
#         x = GlobalMaxPooling2D()(x)
#         x = Reshape((1, 2048))(x)
#         results.append(x)
#         print(res.get_shape())
#     x = concatenate(results, axis=1)
#     # x = TimeDistributed(updated_base_model, input_shape=INPUT_SHAPE)(img_input)
#     print(x.get_shape())
#     x_avg = GlobalAveragePooling1D()(x)
#     x_max = GlobalMaxPooling1D()(x)
#     combined = concatenate([x_avg, x_max])
#     res = Dense(NB_CLASSES, activation='sigmoid', name='classes', kernel_regularizer=l1(1e-5))(combined)
#
#     model = Model(inputs=img_input, outputs=res)
#     sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
#     model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])
#     model.summary()
#     return model


def build_model_vgg16(lock_base_model: True):
    img_input = Input(INPUT_SHAPE, name='data')
    base_model = VGG16(input_shape=INPUT_SHAPE[1:], include_top=False, pooling=None)
    if lock_base_model:
        for layer in base_model.layers:
            layer.trainable = False

    x = TimeDistributed(base_model, input_shape=INPUT_SHAPE)(img_input)
    print(x.get_shape())

    x_avg = Lambda(lambda x: tf.reduce_mean(x, axis=1))(x)
    x_max = Lambda(lambda x: tf.reduce_max(x, axis=1))(x)
    x_avg = GlobalAveragePooling2D()(x_avg)
    x_max_max = GlobalMaxPooling2D()(x_max)
    x_max = GlobalAveragePooling2D()(x_max)

    combined = concatenate([x_avg, x_max, x_max_max])
    combined = Flatten()(combined)
    x = Dropout(0.25)(combined)
    x = Dense(128, activation='relu', name='last_dense')(x)
    res = Dense(NB_CLASSES, activation='sigmoid', name='classes', kernel_regularizer=l1(1e-5))(x)

    model = Model(inputs=img_input, outputs=res)
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    return model


def build_model_resnet50(lock_base_model: True):
    img_input = Input(INPUT_SHAPE, name='data')
    base_model = ResNet50(input_shape=INPUT_SHAPE[1:], include_top=False, pooling=None)
    if lock_base_model:
        for layer in base_model.layers:
            layer.trainable = False

    x = TimeDistributed(base_model, input_shape=INPUT_SHAPE)(img_input)
    print(x.get_shape())

    x_avg = Lambda(lambda x: tf.reduce_mean(x, axis=1))(x)
    x_max = Lambda(lambda x: tf.reduce_max(x, axis=1))(x)
    x_avg = GlobalAveragePooling2D()(x_avg)
    x_max_max = GlobalMaxPooling2D()(x_max)
    x_max = GlobalAveragePooling2D()(x_max)

    combined = concatenate([x_avg, x_max, x_max_max])
    combined = Flatten()(combined)
    x = Dropout(0.25)(combined)
    x = Dense(64, activation='relu', name='last_dense')(x)
    res = Dense(NB_CLASSES, activation='sigmoid', name='classes', kernel_regularizer=l1(1e-5))(x)

    model = Model(inputs=img_input, outputs=res)
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    return model


ModelInfo = namedtuple('ModelInfo', ['factory', 'preprocess_input', 'batch_size'])

MODELS = {
    'resnet50_w8': ModelInfo(
        factory=build_model_resnet50,
        preprocess_input=preprocess_input_resnet50,
        batch_size=4
    ),
    'vgg16_w8': ModelInfo(
        factory=build_model_vgg16,
        preprocess_input=preprocess_input_vgg16,
        batch_size=4
    )
}


class ParallelCNNDataset:
    def __init__(self, fold, preprocess_input_func, batch_size, validation_batch_size=1):
        self.validation_batch_size = validation_batch_size
        self.batch_size = batch_size
        self.combine_batches = 2  # combine multiple batches in generator for parallel processing
        self.preprocess_input_func = preprocess_input_func
        self.training_set_labels_ds_full = pd.read_csv(config.TRAINING_SET_LABELS)
        # self.loaded_files = set([fn for fn in os.listdir(config.RAW_VIDEO_DIR) if fn.endswith('.mp4')])
        self.loaded_files = set([fn + '.mp4' for fn in os.listdir(config.TRAIN_IMG_DIR)])

        self.training_set_labels_ds = self.training_set_labels_ds_full[
            self.training_set_labels_ds_full.filename.isin(self.loaded_files)]
        self.file_names = list(self.training_set_labels_ds.filename)
        self.training_set_labels_ds = self.training_set_labels_ds.set_index('filename')

        self.folds = pd.read_csv('../input/folds.csv')
        train_clips = set(self.folds[self.folds.fold != fold].filename)
        test_clips = set(self.folds[self.folds.fold == fold].filename)

        self.train_clips = list(self.loaded_files.intersection(train_clips))
        self.test_clips = list(self.loaded_files.intersection(test_clips))

        self.test_clips = self.test_clips[:self.validation_steps() * validation_batch_size]
        self.pool = ThreadPool(16)

        print('train clips:', len(self.train_clips))
        print('test clips:', len(self.test_clips))

        self.train_clips_per_cat = {cls: [] for cls in range(NB_CLASSES)}  # cat id -> list of video_id
        self.y_map = {}
        for cls in range(NB_CLASSES):
            self.train_clips_per_cat[cls] = list(
                self.training_set_labels_ds[self.training_set_labels_ds[CLASSES[cls]] > 0.5].index.values)

        # for video_id, row in self.training_set_labels_ds.iterrows():
        #     classes = self.training_set_labels_ds.loc[[video_id]].as_matrix(columns=CLASSES)[0]
        #     self.y_map[video_id] = classes
        # for cls in range(NB_CLASSES):
        #     if classes[cls] > 0.5:
        #         self.train_clips_per_cat[cls].append(video_id)

        for cls in range(NB_CLASSES):
            print(CLASSES[cls], len(self.train_clips_per_cat[cls]))

    def train_steps_per_epoch(self):
        preprocess_batch_size = self.batch_size * self.combine_batches
        return int(len(self.train_clips) / 2 // preprocess_batch_size * preprocess_batch_size // self.batch_size)

    def validation_steps(self):
        preprocess_batch_size = self.validation_batch_size * self.combine_batches
        return len(self.test_clips) // preprocess_batch_size * preprocess_batch_size // self.validation_batch_size

    def load_train_clip(self, video_id, offset=0, hflip=False):
        base_name = video_id[:-4]
        X = np.zeros(shape=INPUT_SHAPE, dtype=np.float32)
        frame_offsets = [2, 4, 6, 8, 10, 12, 14, 16]

        # def load_img(i):
        #     fn = os.path.join(config.TRAIN_IMG_DIR, base_name, f'{i+offset:04}.jpg')
        #     img = scipy.misc.imread(fn)
        #     return scipy.misc.imresize(img, size=(INPUT_ROWS, INPUT_COLS), interp='bilinear').astype(np.float32)
        #
        # images = self.pool2.map(load_img, frame_offsets)
        # for i in range(NB_FRAMES):
        #     X[i] = images[i]

        for i in range(NB_FRAMES):
            fn = os.path.join(config.TRAIN_IMG_DIR, base_name, f'{frame_offsets[i]+offset:04}.jpg')
            img = scipy.misc.imread(fn).astype(np.float32)
            X[i] = scipy.misc.imresize(img, size=(INPUT_ROWS, INPUT_COLS), interp='bilinear').astype(np.float32)

        if hflip:
            X = X[:, :, ::-1]
        # utils.print_stats('X', X)
        classes = self.training_set_labels_ds.loc[[video_id]].as_matrix(columns=CLASSES)
        return self.preprocess_input_func(X), classes[0]
        #
        # X = np.zeros(shape=INPUT_SHAPE, dtype=np.float32)
        # v = pims.Video(os.path.join(config.SMALL_VIDEO_DIR, video_id))
        # for i in range(NB_FRAMES):
        #     try:
        #         frame = v[i * VIDEO_FPS + offset]
        #         X[i] = frame
        #     except IndexError:
        #         if i > 0:
        #             X[i] = X[i-1]
        #         else:
        #             X[i] = 0.0
        # del v
        #
        # classes = self.training_set_labels_ds.loc[[video_id]].as_matrix(columns=CLASSES)
        # return self.preprocess_input_func(X), classes[0]

    def choose_train_video_id(self):
        while True:
            if np.random.choice([True, False]):
                return np.random.choice(self.train_clips)
            else:
                cls = np.random.randint(0, NB_CLASSES)
                count_threshold = np.random.choice([1000, 100, 10, 1], p=[0.69, 0.2, 0.1, 0.01])
                if len(self.train_clips_per_cat[cls]) >= count_threshold:
                    return np.random.choice(self.train_clips_per_cat[cls])

    def generate(self):
        batch_size = self.batch_size
        X = np.zeros(shape=(batch_size,) + INPUT_SHAPE, dtype=np.float32)
        y = np.zeros(shape=(batch_size, NB_CLASSES), dtype=np.float32)

        def load_clip(video_id):
            return self.load_train_clip(video_id, offset=np.random.choice([0, 1]))

        while True:
            for i in range(int(self.train_steps_per_epoch() // self.combine_batches)):
                values_to_process = batch_size*self.combine_batches
                request_ids = [self.choose_train_video_id() for _ in range(values_to_process)]
                results = self.pool.map(load_clip, request_ids)

                for j in range(values_to_process):
                    X[j % batch_size], y[j % batch_size] = results[j]
                    if (j+1) % batch_size == 0:
                        yield X, y

    def generate_test(self):
        batch_size = self.validation_batch_size
        X = np.zeros(shape=(batch_size,) + INPUT_SHAPE, dtype=np.float32)
        y = np.zeros(shape=(batch_size, NB_CLASSES), dtype=np.float32)

        def load_clip(video_id):
            return self.load_train_clip(video_id, offset=1)

        while True:
            video_ids = self.test_clips

            for i in range(int(self.validation_steps() // self.combine_batches)):
                values_to_process = batch_size*self.combine_batches
                request_ids = video_ids[i*values_to_process: (i+1)*values_to_process]
                results = self.pool.map(load_clip, request_ids)

                for j in range(values_to_process):
                    X[j % batch_size], y[j % batch_size] = results[j]
                    if (j+1) % batch_size == 0:
                        yield X, y


def check_generator():
    dataset = ParallelCNNDataset(preprocess_input_func=preprocess_input_resnet50,
                                 batch_size=2,
                                 validation_batch_size=2,
                                 fold=1)
    batch_id = 0
    startTime = time.time()
    for X, y in dataset.generate():
        batch_id += 1
        elapsedTime = time.time() - startTime
        startTime = time.time()
        utils.print_stats('X', X)
        print(f'{batch_id} {elapsedTime:.3}')
        # for batch_frame in range(dataset.batch_size):
        #     print(y[batch_frame])
        #     for frame in range(NB_FRAMES):
        #         plt.imshow(utils.preprocessed_input_to_img_resnet(X[batch_frame, frame]))
        #         plt.show()


def train_initial(fold, model_name):
    model_info = MODELS[model_name]
    model = model_info.factory(lock_base_model=True)
    model.compile(optimizer=Adam(lr=1e-3), loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    dataset = ParallelCNNDataset(fold=fold,
                                 preprocess_input_func=model_info.preprocess_input,
                                 batch_size=model_info.batch_size, validation_batch_size=1)
    model.fit_generator(
        dataset.generate(),
        steps_per_epoch=dataset.train_steps_per_epoch(),
        epochs=1,
        verbose=1,
        validation_data=dataset.generate_test(),
        validation_steps=dataset.validation_steps()
    )
    model.save_weights(f'../output/{model_name}_initial_fold_{fold}_tf.h5')


def train_continue(fold, model_name):
    model_info = MODELS[model_name]
    model = model_info.factory(lock_base_model=True)
    model.load_weights(f'../output/{model_name}_initial_fold_{fold}_tf.h5')
    w1 = model.layers[-1].get_weights()
    w2 = model.layers[-2].get_weights()
    del model
    K.clear_session()

    # w = np.load(f'../output/{model_name}_initial_fold_{fold}_tf_last_w.npy')
    model = model_info.factory(lock_base_model=False)
    model.layers[-1].set_weights(w1)
    model.layers[-2].set_weights(w2)
    model.compile(optimizer=RMSprop(lr=5e-4), loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    dataset = ParallelCNNDataset(preprocess_input_func=model_info.preprocess_input,
                                 batch_size=model_info.batch_size, validation_batch_size=1, fold=fold)

    checkpoints_dir = f'../output/checkpoints/{model_name}_fold_{fold}'
    tensorboard_dir = f'../output/tensorboard/{model_name}_fold_{fold}'
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)

    def cheduler(epoch):
        if epoch < 5:
            return 1e-4
        if epoch < 9:
            return 5e-5
        if epoch < 15:
            return 3e-5
        return 2e-5

    checkpoint_periodical = ModelCheckpoint(checkpoints_dir + "/checkpoint-{epoch:03d}-{val_loss:.4f}.hdf5",
                                            verbose=1,
                                            save_weights_only=True,
                                            period=1)
    tensorboard = TensorBoard(tensorboard_dir, histogram_freq=0, write_graph=False, write_images=False)
    lr_sched = LearningRateScheduler(schedule=cheduler)

    model.fit_generator(
        dataset.generate(),
        steps_per_epoch=dataset.train_steps_per_epoch(),
        epochs=40,
        verbose=1,
        validation_data=dataset.generate_test(),
        validation_steps=dataset.validation_steps(),
        callbacks=[
            checkpoint_periodical,
            tensorboard,
            lr_sched
        ]
    )
    model.save_weights(f'../output/p_{model_name}_fold_{fold}_tf.h5')


def check_model(weights, model_name):
    model_info = MODELS[model_name]
    model = model_info.factory(lock_base_model=False)
    model.load_weights(weights, by_name=True)

    dataset = ParallelCNNDataset(preprocess_input_func=model_info.preprocess_input, batch_size=1, validation_batch_size=1)
    batch_id = 0
    for X, y in dataset.generate_test():
        pred = model.predict_on_batch(X)
        print()
        for i, cls in enumerate(CLASSES):
            print(f'gt: {y[0, i]}  pred: {pred[0, i]:.03f}  {cls}')
            # print(dataset.test_clips[batch_id])
            # print('gt: ', y[0])
            # print('pred: ', pred[0])

        batch_id += 1
        for batch_frame in range(dataset.batch_size):
            for frame in range(2, 7, 3):
                # print(y[batch_frame])
                plt.imshow(utils.preprocessed_input_to_img_resnet(X[batch_frame, frame]))
                plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parallel cnn')
    parser.add_argument('action', type=str, default='check_model')
    parser.add_argument('--weights', type=str, default='')
    parser.add_argument('--fold', type=int, default=1)
    parser.add_argument('--model', type=str)

    args = parser.parse_args()
    action = args.action
    model_name = args.model

    if action == 'check_generator':
        check_generator()
    elif action == 'check_model':
        check_model(weights=args.weights, model_name=model_name)
    elif action == 'train_initial':
        train_initial(fold=args.fold, model_name=model_name)
    elif action == 'train_continue':
        train_continue(fold=args.fold, model_name=model_name)

