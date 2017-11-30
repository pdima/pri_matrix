import argparse
import os
import time
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pims
import scipy.misc
import tensorflow as tf
from tensorflow.python.keras.applications import ResNet50, InceptionV3, Xception
from tensorflow.python.keras.applications.resnet50 import preprocess_input as preprocess_input_resnet50
from tensorflow.python.keras.applications.xception import preprocess_input as preprocess_input_xception
from tensorflow.python.keras.applications.inception_v3 import preprocess_input as preprocess_input_inception_v3

from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler
from tensorflow.python.keras.layers import Dense, Dropout, GlobalMaxPooling1D, GlobalAveragePooling1D
from tensorflow.python.keras.layers import GlobalMaxPooling2D, GlobalAveragePooling2D, AveragePooling2D
from tensorflow.python.keras.layers import GlobalMaxPooling3D, GlobalAveragePooling3D
from tensorflow.python.keras.layers import Input, Lambda, Reshape, TimeDistributed
from tensorflow.python.keras.layers import concatenate
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.python.keras.regularizers import l1
from sklearn.model_selection import train_test_split

from inception_resnet_v2 import InceptionResNetV2
from inception_resnet_v2 import preprocess_input as preprocess_input_inception_resnet_v2
from cnn_finetune import resnet_152
# import inception_v4
import metrics
from tqdm import tqdm


from multiprocessing.pool import ThreadPool
import concurrent.futures
from queue import Queue
from tqdm import tqdm

import config
import utils
import pickle

from tensorflow.python.keras import backend as K
from PIL import Image

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
# tf_config.gpu_options.visible_device_list = "0"
K.set_session(tf.Session(config=tf_config))

INPUT_ROWS = 404
INPUT_COLS = 720
INPUT_SHAPE = (INPUT_ROWS, INPUT_COLS, 3)
VIDEO_FPS = 24
PREDICT_FRAMES = [2, 8, 12, 18] + [i * VIDEO_FPS // 2 + 24 for i in range(14 * 2)]

CLASSES = ['bird', 'blank', 'cattle', 'chimpanzee', 'elephant', 'forest buffalo', 'gorilla', 'hippopotamus', 'human',
           'hyena', 'large ungulate', 'leopard', 'lion', 'other (non-primate)', 'other (primate)', 'pangolin',
           'porcupine', 'reptile', 'rodent', 'small antelope', 'small cat', 'wild dog', 'duiker', 'hog']
NB_CLASSES = len(CLASSES)


def build_model_resnet50(lock_base_model: bool):
    base_model = ResNet50(input_shape=INPUT_SHAPE, include_top=False, pooling=None)
    if lock_base_model:
        for layer in base_model.layers:
            layer.trainable = False
    x = AveragePooling2D((5, 5), name='avg_pool5', strides=1)(base_model.layers[-2].output)
    x = GlobalMaxPooling2D()(x)
    res = Dense(NB_CLASSES, activation='sigmoid', name='classes', kernel_initializer='zero',
                kernel_regularizer=l1(1e-5))(x)
    model = Model(inputs=base_model.inputs, outputs=res)
    return model


def build_model_resnet50_avg(lock_base_model: bool):
    base_model = ResNet50(input_shape=INPUT_SHAPE, include_top=False, pooling=None)
    if lock_base_model:
        for layer in base_model.layers:
            layer.trainable = False
    x = GlobalAveragePooling2D(name='avg_pool_final')(base_model.layers[-2].output)
    res = Dense(NB_CLASSES, activation='sigmoid', name='classes', kernel_initializer='zero',
                kernel_regularizer=l1(1e-5))(x)
    model = Model(inputs=base_model.inputs, outputs=res)
    return model


def build_model_resnet152(lock_base_model: bool):
    model = resnet_152.resnet152_model(img_shape=INPUT_SHAPE, num_classes=NB_CLASSES)
    if lock_base_model:
        for layer in model.layers[:-1]:
            layer.trainable = False
    # model.summary()
    return model


def build_model_xception(lock_base_model: bool):
    base_model = Xception(input_shape=INPUT_SHAPE, include_top=False, pooling=None)
    if lock_base_model:
        for layer in base_model.layers:
            layer.trainable = False
    x = AveragePooling2D((5, 5), name='avg_pool5', strides=1)(base_model.layers[-2].output)
    x = GlobalMaxPooling2D()(x)
    res = Dense(NB_CLASSES, activation='sigmoid', name='classes', kernel_initializer='zero',
                kernel_regularizer=l1(1e-5))(x)
    model = Model(inputs=base_model.inputs, outputs=res)
    return model


def build_model_xception_avg(lock_base_model: bool):
    base_model = Xception(input_shape=INPUT_SHAPE, include_top=False, pooling=None)
    if lock_base_model:
        for layer in base_model.layers:
            layer.trainable = False
    x = GlobalAveragePooling2D(name='avg_pool_final')(base_model.layers[-1].output)
    res = Dense(NB_CLASSES, activation='sigmoid', name='classes', kernel_initializer='zero',
                kernel_regularizer=l1(1e-5))(x)
    model = Model(inputs=base_model.inputs, outputs=res)
    return model


def build_model_inception_v3_avg(lock_base_model: True):
    base_model = InceptionV3(input_shape=INPUT_SHAPE, include_top=False, pooling=None)
    if lock_base_model:
        for layer in base_model.layers:
            layer.trainable = False
    # base_model.summary()
    x = GlobalAveragePooling2D(name='avg_pool_final')(base_model.layers[-1].output)
    res = Dense(NB_CLASSES, activation='sigmoid', name='classes', kernel_initializer='zero',
                kernel_regularizer=l1(1e-5))(x)
    model = Model(inputs=base_model.inputs, outputs=res)
    # model.summary()
    return model


def build_model_inception_v2_resnet(lock_base_model: True):
    img_input = Input(shape=INPUT_SHAPE)
    base_model = InceptionResNetV2(input_tensor=img_input, include_top=False, pooling='avg')
    if lock_base_model:
        for layer in base_model.layers:
            layer.trainable = False
    # base_model.summary()
    res = Dense(NB_CLASSES, activation='sigmoid', name='classes', kernel_initializer='zero',
                kernel_regularizer=l1(1e-5))(base_model.layers[-1].output)
    model = Model(inputs=img_input, outputs=res)
    # model.summary()
    return model


def build_model_inception_v3_dropout(lock_base_model: True):
    base_model = InceptionV3(input_shape=INPUT_SHAPE, include_top=False, pooling=None)
    if lock_base_model:
        for layer in base_model.layers:
            layer.trainable = False
    # base_model.summary()
    x = GlobalAveragePooling2D(name='avg_pool_final')(base_model.layers[-1].output)
    x = Dropout(0.25)(x)
    res = Dense(NB_CLASSES, activation='sigmoid', name='classes', kernel_initializer='zero',
                kernel_regularizer=l1(1e-5))(x)
    model = Model(inputs=base_model.inputs, outputs=res)
    # model.summary()
    return model


ModelInfo = namedtuple('ModelInfo', ['factory', 'preprocess_input', 'input_shape', 'unlock_layer_name', 'batch_size'])

MODELS = {
    'resnet50': ModelInfo(
        factory=build_model_resnet50,
        preprocess_input=preprocess_input_resnet50,
        input_shape=(404, 720, 3),
        unlock_layer_name='activation_22',
        batch_size=32
    ),
    'resnet50_avg': ModelInfo(
        factory=build_model_resnet50_avg,
        preprocess_input=preprocess_input_resnet50,
        input_shape=(404, 720, 3),
        unlock_layer_name='activation_22',
        batch_size=32
    ),
    'inception_v3_avg': ModelInfo(
        factory=build_model_inception_v3_avg,
        preprocess_input=preprocess_input_inception_v3,
        input_shape=(404, 720, 3),
        unlock_layer_name='mixed9',
        batch_size=32
    ),
    'inception_v3_avg_m8': ModelInfo(
        factory=build_model_inception_v3_avg,
        preprocess_input=preprocess_input_inception_v3,
        input_shape=(404, 720, 3),
        unlock_layer_name='mixed8',
        batch_size=32
    ),
    'inception_v2_resnet': ModelInfo(
        factory=build_model_inception_v2_resnet,
        preprocess_input=preprocess_input_inception_resnet_v2,
        input_shape=(404, 720, 3),
        unlock_layer_name='activation_75',
        batch_size=16
    ),
    'xception_avg': ModelInfo(
        factory=build_model_xception_avg,
        preprocess_input=preprocess_input_xception,
        input_shape=(404, 720, 3),
        unlock_layer_name='block4_pool',
        batch_size=16
    ),
    'resnet152': ModelInfo(
        factory=build_model_resnet152,
        preprocess_input=preprocess_input_resnet50,
        input_shape=(404, 720, 3),
        unlock_layer_name='res3b7_relu',
        batch_size=8
    ),
}

# extra names used for different checkpoints, ideas/etc
MODELS['inception_v3_avg_m8_ch2'] = MODELS['inception_v3_avg_m8']
MODELS['inception_v3_avg_m8_ch5'] = MODELS['inception_v3_avg_m8']
MODELS['inception_v3_avg_m8_ch9'] = MODELS['inception_v3_avg_m8']
MODELS['inception_v3_avg_m8_ch24'] = MODELS['inception_v3_avg_m8']
MODELS['xception_avg_ch10'] = MODELS['xception_avg']


class SingleFrameCNNDataset:
    def __init__(self, fold, preprocess_input_func, batch_size, validation_batch_size=1, use_non_blank_frames=False):
        self.validation_batch_size = validation_batch_size
        self.batch_size = batch_size
        self.combine_batches = 1  # combine multiple batches in generator for parallel processing
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
        self.pool = ThreadPool(8)

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

        self.non_blank_frames = {}
        if use_non_blank_frames:
            for fn in ['resnet50_avg_1_non_blank.pkl',
                       'resnet50_2_non_blank.pkl',
                       'resnet50_avg_3_non_blank.pkl',
                       'resnet50_avg_4_non_blank.pkl']:
                data = pickle.load(open('../output/prediction_train_frames/' + fn, 'rb'))
                self.non_blank_frames.update(data)

    def train_steps_per_epoch(self):
        preprocess_batch_size = self.batch_size * self.combine_batches
        return int(len(self.train_clips) / 2 // preprocess_batch_size * preprocess_batch_size // self.batch_size)

    def validation_steps(self):
        preprocess_batch_size = self.validation_batch_size * self.combine_batches
        return len(self.test_clips) // preprocess_batch_size * preprocess_batch_size // self.validation_batch_size

    def load_train_clip(self, video_id, offset=4, hflip=False):
        # v = pims.Video(os.path.join(config.RAW_VIDEO_DIR, video_id))
        # X = v[offset]
        # if X.shape != INPUT_SHAPE:
        #     X = scipy.misc.imresize(X, size=(INPUT_ROWS, INPUT_COLS), interp='bilinear').astype(np.float32)
        # else:
        #     X = X.astype(np.float32)
        # del v
        base_name = video_id[:-4]
        fn = os.path.join(config.TRAIN_IMG_DIR, base_name, f'{offset+1:04}.jpg')
        X = scipy.misc.imread(fn).astype(np.float32)
        if hflip:
            X = X[:, ::-1]
        # utils.print_stats('X', X)
        classes = self.training_set_labels_ds.loc[[video_id]].as_matrix(columns=CLASSES)
        return self.preprocess_input_func(X), classes[0]

    def choose_train_video_id(self):
        while True:
            if np.random.choice([True, False]):
                return np.random.choice(self.train_clips)
            else:
                cls = np.random.randint(0, NB_CLASSES)
                count_threshold = np.random.choice([1000, 100, 10, 1], p=[0.69, 0.2, 0.1, 0.01])
                if len(self.train_clips_per_cat[cls]) >= count_threshold:
                    return np.random.choice(self.train_clips_per_cat[cls])

    def generate(self, verbose=False):
        batch_size = self.batch_size
        X = np.zeros(shape=(batch_size,) + INPUT_SHAPE, dtype=np.float32)
        y = np.zeros(shape=(batch_size, NB_CLASSES), dtype=np.float32)

        def load_clip(video_id):
            if video_id in self.non_blank_frames:
                weights = self.non_blank_frames[video_id]
                offset = np.random.choice(list(range(1, 17)), p=weights / np.sum(weights))
            else:
                offset = np.random.randint(1, 17)
            return self.load_train_clip(video_id, offset=offset, hflip=np.random.choice([True, False]))

        while True:
            video_ids = self.train_clips
            np.random.shuffle(video_ids)
            # video_ids = video_ids[:self.train_steps_per_epoch() * batch_size]

            for i in range(int(self.train_steps_per_epoch() // self.combine_batches)):
                values_to_process = batch_size * self.combine_batches
                request_ids = [self.choose_train_video_id() for _ in
                               range(values_to_process)]  # video_ids[i*values_to_process: (i+1)*values_to_process]
                if verbose:
                    print(request_ids)
                results = self.pool.map(load_clip, request_ids)

                for j in range(values_to_process):
                    X[j % batch_size], y[j % batch_size] = results[j]
                    if (j + 1) % batch_size == 0:
                        yield X, y

    def generate_test(self, verbose=False, output_video_ids=False):
        batch_size = self.validation_batch_size
        X = np.zeros(shape=(batch_size,) + INPUT_SHAPE, dtype=np.float32)
        y = np.zeros(shape=(batch_size, NB_CLASSES), dtype=np.float32)

        def load_clip(video_id):
            if video_id in self.non_blank_frames:
                weights = self.non_blank_frames[video_id]
                offset = np.argmax(weights) + 1  # first frame is skipped
            else:
                offset = 4
            return self.load_train_clip(video_id, offset=offset)

        while True:
            video_ids = self.test_clips

            for i in range(int(self.validation_steps() // self.combine_batches)):
                values_to_process = batch_size * self.combine_batches
                request_ids = video_ids[i * values_to_process: (i + 1) * values_to_process]
                if verbose:
                    print(request_ids)
                results = self.pool.map(load_clip, request_ids)

                for j in range(values_to_process):
                    X[j % batch_size], y[j % batch_size] = results[j]
                    if (j + 1) % batch_size == 0:
                        yield X, y

    def frames_from_video_clip(self, video_fn):
        X = np.zeros(shape=(len(PREDICT_FRAMES),) + INPUT_SHAPE, dtype=np.float32)
        v = pims.Video(video_fn)
        for i, frame_num in enumerate(PREDICT_FRAMES):
            try:
                frame = v[frame_num]
                if frame.shape != INPUT_SHAPE:
                    frame = scipy.misc.imresize(frame, size=(INPUT_ROWS, INPUT_COLS), interp='bilinear').astype(
                        np.float32)
                else:
                    frame = frame.astype(np.float32)
                X[i] = frame
            except IndexError:
                if i > 0:
                    X[i] = X[i - 1]
                else:
                    X[i] = 0.0
        del v
        return self.preprocess_input_func(X)

    def generate_frames_for_prediction(self):
        for video_id in sorted(self.test_clips):
            X = self.frames_from_video_clip(video_fn=os.path.join(config.RAW_VIDEO_DIR, video_id))
            y = self.training_set_labels_ds.loc[[video_id]].as_matrix(columns=CLASSES)
            yield video_id, X, y

    def generate_test_frames_for_prediction(self):
        test_ds = pd.read_csv(config.SUBMISSION_FORMAT)
        for video_id in test_ds.filename:
            X = self.frames_from_video_clip(video_fn=os.path.join(config.TEST_VIDEO_DIR, video_id))
            yield video_id, X


def check_generator(use_test):
    dataset = SingleFrameCNNDataset(preprocess_input_func=preprocess_input_resnet50,
                                    batch_size=2,
                                    validation_batch_size=2,
                                    fold=1)
    batch_id = 0
    startTime = time.time()

    if use_test:
        gen = dataset.generate_test()
    else:
        gen = dataset.generate()

    for X, y in gen:
        batch_id += 1
        elapsedTime = time.time() - startTime
        startTime = time.time()
        print(f'{batch_id} {elapsedTime:.3}')
        for batch_frame in range(dataset.batch_size):
            print(y[batch_frame])
            plt.imshow(utils.preprocessed_input_to_img_resnet(X[batch_frame]))
            plt.show()


def train_initial(fold, model_name, use_non_blank_frames):
    model_info = MODELS[model_name]
    model = model_info.factory(lock_base_model=True)
    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    # model.summary()
    dataset = SingleFrameCNNDataset(preprocess_input_func=model_info.preprocess_input,
                                    fold=fold,
                                    batch_size=model_info.batch_size,
                                    validation_batch_size=model_info.batch_size,
                                    use_non_blank_frames=use_non_blank_frames)

    tensorboard_dir = f'../output/tensorboard/{model_name}_initial_fold_{fold}'
    os.makedirs(tensorboard_dir, exist_ok=True)

    model.fit_generator(
        dataset.generate(),
        steps_per_epoch=dataset.train_steps_per_epoch(),
        epochs=1,
        verbose=1,
        validation_data=dataset.generate_test(),
        validation_steps=dataset.validation_steps(),
        callbacks=[
            # TensorBoard(tensorboard_dir, histogram_freq=1, write_graph=False, write_images=True)
        ]
    )
    model.save_weights(f'../output/{model_name}_s_initial_fold_{fold}_tf.h5')
    last_w = model.layers[-1].get_weights()
    np.save(f'../output/{model_name}_s_initial_fold_{fold}_tf_last_w.npy', last_w)


def train_continue(fold, model_name, weights, initial_epoch, use_non_blank_frames):
    model_info = MODELS[model_name]
    model = model_info.factory(lock_base_model=True)
    utils.lock_layers_until(model, model_info.unlock_layer_name)
    if weights == '':
        w = np.load(f'../output/{model_name}_s_initial_fold_{fold}_tf_last_w.npy')
        model.layers[-1].set_weights(w)
    else:
        model.load_weights(weights)

    # model.summary()
    model.compile(optimizer=RMSprop(lr=3e-4), loss='binary_crossentropy', metrics=['accuracy'])

    dataset = SingleFrameCNNDataset(preprocess_input_func=model_info.preprocess_input,
                                    fold=fold,
                                    batch_size=model_info.batch_size,
                                    validation_batch_size=model_info.batch_size,
                                    use_non_blank_frames=use_non_blank_frames)

    checkpoints_dir = f'../output/checkpoints/{model_name}_fold_{fold}'
    tensorboard_dir = f'../output/tensorboard/{model_name}_fold_{fold}'
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)

    def cheduler(epoch):
        if epoch < 5:
            return 1e-4
        if epoch < 9:
            return 5e-5
        # if epoch < 15:
        #     return 3e-5
        return 3e-5

    checkpoint_periodical = ModelCheckpoint(checkpoints_dir + "/checkpoint-{epoch:03d}-{val_loss:.4f}.hdf5",
                                            verbose=1,
                                            save_weights_only=True,
                                            period=1)
    tensorboard = TensorBoard(tensorboard_dir, histogram_freq=0, write_graph=False, write_images=False)
    lr_sched = LearningRateScheduler(schedule=cheduler)

    model.fit_generator(
        dataset.generate(),
        steps_per_epoch=dataset.train_steps_per_epoch(),
        epochs=30,
        verbose=1,
        validation_data=dataset.generate_test(),
        validation_steps=dataset.validation_steps(),
        callbacks=[
            checkpoint_periodical,
            tensorboard,
            lr_sched
        ],
        initial_epoch=initial_epoch
    )
    model.save_weights(f'../output/{model_name}_s_fold_{fold}.h5')


def check_model(model_name, weights, fold):
    model = MODELS[model_name].factory(lock_base_model=True)
    model.load_weights(weights, by_name=True)

    dataset = SingleFrameCNNDataset(preprocess_input_func=MODELS[model_name].preprocess_input,
                                    batch_size=1,
                                    validation_batch_size=1,
                                    fold=fold)
    batch_id = 0
    for X, y in dataset.generate_test():
        pred = model.predict_on_batch(X)
        print()
        for i, cls in enumerate(CLASSES):
            print(f'gt: {y[0, i]}  pred: {pred[0, i]:.03f}  {cls}')
        batch_id += 1
        for batch_frame in range(dataset.batch_size):
            plt.imshow(utils.preprocessed_input_to_img_resnet(X[batch_frame]))
            plt.show()


def check_model_score(model_name, weights, fold):
    model = MODELS[model_name].factory(lock_base_model=True)
    model.load_weights(weights, by_name=True)
    model.compile(optimizer=RMSprop(lr=3e-4), loss='binary_crossentropy', metrics=['accuracy'])

    dataset = SingleFrameCNNDataset(preprocess_input_func=MODELS[model_name].preprocess_input,
                                    batch_size=MODELS[model_name].batch_size,
                                    validation_batch_size=4,  # MODELS[model_name].batch_size,
                                    fold=fold,
                                    use_non_blank_frames=True)
    gt = []
    pred = []
    steps = dataset.validation_steps() // 500
    print('steps:', steps)
    step = 0
    for X, y in tqdm(dataset.generate_test(verbose=True)):
        gt.append(y)
        pred.append(model.predict_on_batch(X))
        step += 1
        if step >= steps:
            break

    # print(gt)
    # print(pred)

    gt = np.vstack(gt).astype(np.float64)
    pred = np.vstack(pred).astype(np.float64)
    print(gt.shape, pred.shape)
    # print(gt)
    # print(pred)
    checkpoint_name = os.path.basename(weights)
    out_dir = f'../output/check_model_score/gt{model_name}_{fold}_{checkpoint_name}'
    os.makedirs(out_dir, exist_ok=True)
    print(out_dir)
    np.save(f'{out_dir}/gt.npy', gt)
    np.save(f'{out_dir}/pred.npy', pred)

    for clip in [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
        print(clip, metrics.pri_matrix_loss(gt, np.clip(pred, clip, 1.0 - clip)))


def generate_prediction(model_name, weights, fold):
    model = MODELS[model_name].factory(lock_base_model=True)
    model.load_weights(weights, by_name=True)

    output_dir = f'../output/prediction_train_frames/{model_name}_{fold}/'
    os.makedirs(output_dir, exist_ok=True)

    dataset = SingleFrameCNNDataset(preprocess_input_func=MODELS[model_name].preprocess_input,
                                    batch_size=1,
                                    validation_batch_size=1,
                                    fold=fold)

    # skip processed files
    converted_files = set()
    processed_files = 0
    for video_id in dataset.test_clips:
        res_fn = output_dir + video_id + '.csv'
        if os.path.exists(res_fn):
            processed_files += 1
            converted_files.add(video_id)
    test_clips = sorted(list(set(dataset.test_clips) - converted_files))

    def load_file(video_id):
        X = dataset.frames_from_video_clip(video_fn=os.path.join(config.RAW_VIDEO_DIR, video_id))
        y = dataset.training_set_labels_ds.loc[[video_id]].as_matrix(columns=CLASSES)
        return video_id, X, y

    start_time = time.time()

    pool = ThreadPool(8)
    prev_res = None
    for batch in utils.chunks(test_clips, 16, add_empty=True):
        if prev_res is not None:
            results = prev_res.get()
        else:
            results = []
        prev_res = pool.map_async(load_file, batch)
        for video_id, X, y in results:
            processed_files += 1
            res_fn = output_dir + video_id + '.csv'
            have_data_time = time.time()
            prediction = model.predict(X, batch_size=4)

            ds = pd.DataFrame(index=[-1] + PREDICT_FRAMES,
                              data=np.row_stack([y, prediction]),
                              columns=CLASSES)
            ds.to_csv(res_fn, index_label='frame', float_format='%.5f')

            have_prediction_time = time.time()
            prepare_ms = int((have_data_time - start_time) * 1000)
            predict_ms = int((have_prediction_time - have_data_time) * 1000)
            start_time = time.time()
            print(f'{video_id}  {processed_files} prepared in {prepare_ms} predicted in {predict_ms}')


def generate_prediction_test(model_name, weights, fold):
    model = MODELS[model_name].factory(lock_base_model=True)
    model.load_weights(weights, by_name=True)

    output_dir = f'../output/prediction_test_frames/{model_name}_{fold}/'
    os.makedirs(output_dir, exist_ok=True)

    dataset = SingleFrameCNNDataset(preprocess_input_func=MODELS[model_name].preprocess_input,
                                    batch_size=1,
                                    validation_batch_size=1,
                                    fold=fold)

    # skip processed files
    test_clips = list(pd.read_csv(config.SUBMISSION_FORMAT).filename)

    converted_files = set()
    processed_files = 0
    for video_id in test_clips:
        res_fn = output_dir + video_id + '.csv'
        if os.path.exists(res_fn):
            processed_files += 1
            converted_files.add(video_id)

    test_clips = sorted(list(set(test_clips) - converted_files))

    def load_file(video_id):
        X = dataset.frames_from_video_clip(video_fn=os.path.join(config.TEST_VIDEO_DIR, video_id))
        return video_id, X

    start_time = time.time()

    pool = ThreadPool(8)
    prev_res = None
    for batch in utils.chunks(test_clips, 16, add_empty=True):
        if prev_res is not None:
            results = prev_res.get()
        else:
            results = []
        prev_res = pool.map_async(load_file, batch)
        for video_id, X in results:
            processed_files += 1
            res_fn = output_dir + video_id + '.csv'
            have_data_time = time.time()
            prediction = model.predict(X, batch_size=4)

            ds = pd.DataFrame(index=PREDICT_FRAMES,
                              data=prediction,
                              columns=CLASSES)
            ds.to_csv(res_fn, index_label='frame', float_format='%.5f')
            have_prediction_time = time.time()
            prepare_ms = int((have_data_time - start_time) * 1000)
            predict_ms = int((have_prediction_time - have_data_time) * 1000)
            start_time = time.time()
            print(f'{video_id}  {processed_files} prepared in {prepare_ms} predicted in {predict_ms}')


def find_non_blank_frames(model_name, fold):
    data_dir = f'../output/prediction_train_frames/{model_name}_{fold}/'
    training_labels = pd.read_csv(config.TRAINING_SET_LABELS)
    training_labels = training_labels.set_index('filename')

    res = {}

    for fn in tqdm(sorted(os.listdir(data_dir))):
        if not fn.endswith('csv'):
            continue
        filename = fn[:-len('.csv')]
        if training_labels.loc[filename].blank > 0.9:
            continue

        ds = np.loadtxt(os.path.join(data_dir, fn), delimiter=',', skiprows=1)
        target_frames = np.arange(0, 16) * 12 + 5
        src_frames = ds[1:, 0]

        blank_col = 2
        dst_blank_prob = np.interp(target_frames, src_frames, ds[1:, blank_col])
        res[filename] = 1.0 - dst_blank_prob
    pickle.dump(res, open(f"../output/prediction_train_frames/{model_name}_{fold}_non_blank.pkl", "wb"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parallel cnn')
    parser.add_argument('action', type=str, default='check_model')
    parser.add_argument('--weights', type=str, default='')
    parser.add_argument('--initial_epoch', type=int, default=0)
    parser.add_argument('--fold', type=int, default=1)
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--use_non_blank_frames', action='store_true')

    args = parser.parse_args()
    action = args.action
    model = args.model

    if action == 'check_generator':
        check_generator(use_test=False)
    if action == 'check_generator_test':
        check_generator(use_test=True)
    elif action == 'check_model':
        check_model(model_name=model, weights=args.weights, fold=args.fold)
    elif action == 'check_model_score':
        check_model_score(model_name=model, weights=args.weights, fold=args.fold)
    elif action == 'train_initial':
        train_initial(fold=args.fold, model_name=model, use_non_blank_frames=args.use_non_blank_frames)
    elif action == 'train_continue':
        train_continue(fold=args.fold,
                       model_name=model,
                       weights=args.weights,
                       initial_epoch=args.initial_epoch,
                       use_non_blank_frames=args.use_non_blank_frames)
    elif action == 'generate_prediction':
        generate_prediction(fold=args.fold, model_name=model, weights=args.weights)
    elif action == 'generate_prediction_test':
        generate_prediction_test(fold=args.fold, model_name=model, weights=args.weights)
    elif action == 'find_non_blank_frames':
        find_non_blank_frames(fold=args.fold, model_name=model)
