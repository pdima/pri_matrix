import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pims
import tensorflow as tf
from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.applications.resnet50 import preprocess_input as preprocess_input_resnet50
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler
from tensorflow.python.keras.layers import Dense, GlobalMaxPooling1D, GlobalAveragePooling1D
from tensorflow.python.keras.layers import Input, Lambda, Reshape
from tensorflow.python.keras.layers import concatenate
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import SGD
from sklearn.model_selection import train_test_split

from multiprocessing.pool import ThreadPool

import config
import utils

from tensorflow.python.keras import backend as K

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
# tf_config.gpu_options.visible_device_list = "0"
K.set_session(tf.Session(config=tf_config))

INPUT_ROWS = 202
INPUT_COLS = 360
NB_FRAMES = 12
INPUT_SHAPE = (NB_FRAMES, INPUT_ROWS, INPUT_COLS, 3)

CLASSES = ['bird', 'blank', 'cattle', 'chimpanzee', 'elephant', 'forest buffalo', 'gorilla', 'hippopotamus', 'human',
           'hyena', 'large ungulate', 'leopard', 'lion', 'other (non-primate)', 'other (primate)', 'pangolin',
           'porcupine', 'reptile', 'rodent', 'small antelope', 'small cat', 'wild dog', 'duiker', 'hog']
NB_CLASSES = len(CLASSES)


def build_model_resnet50(lock_base_model:True):
    img_input = Input(INPUT_SHAPE, name='data')
    base_model = ResNet50(input_shape=INPUT_SHAPE[1:], include_top=False, pooling=None)
    # utils.lock_layers_until(base_model, lock_layers_until)
    base_model.summary()

    results = []
    for i in range(NB_FRAMES):
        crop = Lambda(lambda x: x[:, i])(img_input)
        res = base_model(crop)
        res = Reshape((1, 2048))(res)
        results.append(res)
        print(res.get_shape())
    x = concatenate(results, axis=1)
    print(x.get_shape())
    x_avg = GlobalAveragePooling1D()(x)
    x_max = GlobalMaxPooling1D()(x)
    combined = concatenate([x_avg, x_max])
    res = Dense(NB_CLASSES, activation='sigmoid', name='classes')(combined)

    model = Model(inputs=img_input, outputs=res)
    # sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    # model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    return model, base_model


class ParallelCNNDataset:
    def __init__(self, preprocess_input_func, batch_size, validation_batch_size=1):
        self.validation_batch_size = validation_batch_size
        self.batch_size = batch_size
        self.combine_batches = 2  # combine multiple batches in generator for parallel processing
        self.preprocess_input_func = preprocess_input_func
        self.training_set_labels_ds_full = pd.read_csv(config.TRAINING_SET_LABELS)
        self.loaded_files = set([fn for fn in os.listdir(config.SMALL_VIDEO_DIR) if fn.endswith('.mp4')])

        self.training_set_labels_ds = self.training_set_labels_ds_full[self.training_set_labels_ds_full.filename.isin(self.loaded_files)]
        self.file_names = list(self.training_set_labels_ds.filename)
        self.training_set_labels_ds = self.training_set_labels_ds.set_index('filename')

        self.train_clips, self.test_clips = train_test_split(sorted(self.file_names),
                                                             test_size=0.1,
                                                             random_state=12)
        self.test_clips = self.test_clips[:self.validation_steps()*validation_batch_size]
        self.pool = ThreadPool(8)

        self.train_clips_per_cat = {cls: [] for cls in range(NB_CLASSES)}  # cat id -> list of video_id
        for video_id in self.train_clips:
            classes = self.training_set_labels_ds.loc[[video_id]].as_matrix(columns=CLASSES)[0]
            for cls in range(NB_CLASSES):
                if classes[cls] > 0.5:
                    self.train_clips_per_cat[cls].append(video_id)

        for cls in range(NB_CLASSES):
            print(CLASSES[cls], len(self.train_clips_per_cat[cls]))

    def train_steps_per_epoch(self):
        preprocess_batch_size = self.batch_size * self.combine_batches
        return int(len(self.train_clips) / 2 // preprocess_batch_size * preprocess_batch_size // self.batch_size)

    def validation_steps(self):
        preprocess_batch_size = self.validation_batch_size * self.combine_batches
        return len(self.test_clips) // preprocess_batch_size * preprocess_batch_size // self.validation_batch_size

    def load_train_clip(self, video_id, offset=0):
        X = np.zeros(shape=INPUT_SHAPE, dtype=np.float32)
        v = pims.Video(os.path.join(config.SMALL_VIDEO_DIR, video_id))
        for i in range(15):
            try:
                frame = v[i * 24 + offset]
                X[i] = frame
            except IndexError:
                if i > 0:
                    X[i] = X[i-1]
                else:
                    X[i] = 0.0
        del v

        classes = self.training_set_labels_ds.loc[[video_id]].as_matrix(columns=CLASSES)
        return self.preprocess_input_func(X), classes[0]

    def generate(self):
        batch_size = self.batch_size
        X = np.zeros(shape=(batch_size,) + INPUT_SHAPE, dtype=np.float32)
        y = np.zeros(shape=(batch_size, NB_CLASSES), dtype=np.float32)

        def load_clip(video_id):
            return self.load_train_clip(video_id, offset=np.random.randint(0, 23))

        while True:
            video_ids = self.train_clips
            np.random.shuffle(video_ids)
            # video_ids = video_ids[:self.train_steps_per_epoch() * batch_size]

            for i in range(int(self.train_steps_per_epoch() // self.combine_batches)):
                values_to_process = batch_size*self.combine_batches
                request_ids = video_ids[i*values_to_process: (i+1)*values_to_process]
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
            return self.load_train_clip(video_id)

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
    dataset = ParallelCNNDataset(preprocess_input_func=lambda x: x, batch_size=2, validation_batch_size=2)
    batch_id = 0
    startTime = time.time()
    for X, y in dataset.generate_test():
        batch_id += 1
        elapsedTime = time.time() - startTime
        startTime = time.time()
        print(f'{batch_id} {elapsedTime:.3}')
        # for batch_frame in range(dataset.batch_size):
        #     for frame in range(15):
        #         print(y[batch_frame])
        #         plt.imshow(X[batch_frame, frame]/255.0)
        #         plt.show()


def train_initial(fold):
    model = build_model_resnet50(lock_layers_until='activation_49')

    model.summary()
    dataset = ParallelCNNDataset(preprocess_input_func=preprocess_input_resnet50, batch_size=4, validation_batch_size=4)

    model_name = 'resnet50_w15'
    tensorboard_dir = f'../output/tensorboard/{model_name}_initial_fold_{fold}'
    os.makedirs(tensorboard_dir, exist_ok=True)

    model.fit_generator(
        dataset.generate(),
        steps_per_epoch=dataset.train_steps_per_epoch(),
        epochs=2,
        verbose=1,
        validation_data=dataset.generate_test(),
        validation_steps=dataset.validation_steps(),
        callbacks=[
            # TensorBoard(tensorboard_dir, histogram_freq=1, write_graph=False, write_images=True)
        ]
    )
    model.save_weights(f'../output/resnet50_w_initial_fold_{fold}_tf.h5')


def train_continue(fold):
    model = build_model_resnet50(lock_layers_until='activation_49')
    model.load_weights(f'../output/resnet50_w_initial_fold_{fold}_tf.h5')
    w = model.layers[-1].get_weights()
    del model
    K.clear_session()

    model = build_model_resnet50(lock_layers_until='input_1')
    model.layers[-1].set_weights(w)
    model.summary()

    dataset = ParallelCNNDataset(preprocess_input_func=preprocess_input_resnet50, batch_size=2, validation_batch_size=2)

    model_name = 'resnet50_w15_tf'
    checkpoints_dir = f'../output/checkpoints/{model_name}_fold_{fold}'
    tensorboard_dir = f'../output/tensorboard/{model_name}_fold_{fold}'
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)

    def cheduler(epoch):
        if epoch < 1:
            return 5e-4
        if epoch < 5:
            return 3e-4
        if epoch < 10:
            return 1e-4
        if epoch < 15:
            return 5e-5
        return 1e-5

    checkpoint_periodical = ModelCheckpoint(checkpoints_dir + "/checkpoint-{epoch:03d}-{val_loss:.4f}.hdf5",
                                            verbose=1,
                                            save_weights_only=True,
                                            period=1)
    tensorboard = TensorBoard(tensorboard_dir, histogram_freq=0, write_graph=False, write_images=False)
    lr_sched = LearningRateScheduler(schedule=cheduler)

    model.fit_generator(
        dataset.generate(),
        steps_per_epoch=dataset.train_steps_per_epoch(),
        epochs=20,
        verbose=1,
        validation_data=dataset.generate_test(),
        validation_steps=dataset.validation_steps(),
        callbacks=[
            checkpoint_periodical,
            tensorboard,
            lr_sched
        ]
    )
    model.save_weights(f'../output/resnet50_w15_fold_{fold}.h5')


def check_model(weights):
    model = build_model_resnet50(lock_layers_until='input_1')
    model.load_weights(weights, by_name=True)

    dataset = ParallelCNNDataset(preprocess_input_func=lambda x: x, batch_size=1, validation_batch_size=1)
    batch_id = 0
    for X, y in dataset.generate_test():
        pred = model.predict_on_batch(X)
        print()
        for i, cls in enumerate(CLASSES):
            print(f'gt: {y[0, i]}  pred: {pred[0, i]:.02f}  {cls}')
            # print(dataset.test_clips[batch_id])
            # print('gt: ', y[0])
            # print('pred: ', pred[0])

        batch_id += 1
        for batch_frame in range(dataset.batch_size):
            for frame in range(4, 15, 8):
                # print(y[batch_frame])
                plt.imshow(X[batch_frame, frame]/255.0)
                plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parallel cnn')
    parser.add_argument('action', type=str, default='check_model')
    parser.add_argument('--weights', type=str, default='')
    parser.add_argument('--fold', type=int, default=1)

    args = parser.parse_args()
    action = args.action

    if action == 'check_generator':
        check_generator()
    elif action == 'check_model':
        check_model(weights=args.weights)
    elif action == 'train_initial':
        train_initial(fold=args.fold)
    elif action == 'train_continue':
        train_continue(fold=args.fold)

