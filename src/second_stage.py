import numpy as np
import pandas as pd
import os
import pickle
import utils
import sklearn.preprocessing
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import metrics
import config

from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC

NB_CAT = 24


def preprocess_x(data: np.ndarray):
    rows = []

    for row in data:
        items = [
            np.mean(row, axis=0),
            np.median(row, axis=0),
            np.min(row, axis=0),
            np.max(row, axis=0),
            np.percentile(row, q=10, axis=0),
            np.percentile(row, q=90, axis=0),
        ]
        for col in range(row.shape[1]):
            items.append(np.histogram(row[:, col], bins=10, range=(0.0, 1.0), density=True)[0])
        rows.append(np.hstack(items).flatten())

    return np.array(rows)


# def preprocess_x(data: np.ndarray):
#     rows = []
#     downsample = 4
#     for row in data:
#         items = []
#         for col in range(row.shape[1]):
#             sorted = np.sort(row[:, col])
#             items.append(sorted.reshape(-1, downsample).mean(axis=1))
#         rows.append(np.hstack(items))
#     return np.array(rows)


def load_train_data(train_path, load_cache: bool, cache_fn: str, load_raw_cache: bool, raw_cache_fn:str):
    if load_cache and os.path.exists(cache_fn):
        cached = np.load(cache_fn)
        X, y, video_ids = cached['X'], cached['y'], cached['video_ids']
    else:
        if load_raw_cache and os.path.exists(raw_cache_fn):
            cached = np.load(raw_cache_fn)
            X_raw, y, video_ids = cached['X_raw'], cached['y'], cached['video_ids']
        else:
            X_raw = []
            y = []
            video_ids = []
            for fn in tqdm(os.listdir(train_path)):
                if not fn.endswith('csv'):
                    continue
                ds = np.loadtxt(os.path.join(train_path, fn), delimiter=',', skiprows=1)

                # top row is y, top col is frame number
                X_raw.append(ds[1:, 1:])
                y.append(ds[0, 1:])
                video_ids.append(fn[:-4])

            X_raw = np.array(X_raw)
            y = np.array(y)
            np.savez(raw_cache_fn, X_raw=X_raw, y=y, video_ids=video_ids)

        X = preprocess_x(X_raw)
        np.savez(cache_fn, X=X, y=y, video_ids=video_ids)
    return X, y, video_ids


def load_test_data(test_path, video_ids):
    X_raw = []
    for video_id in tqdm(video_ids):
        ds = np.loadtxt(os.path.join(test_path, video_id+'.csv'), delimiter=',', skiprows=1)
        # 0 col is frame number
        X_raw.append(ds[:, 1:])
    X_raw = np.array(X_raw)
    X = preprocess_x(X_raw)
    return X


def avg_probabilities():
    ds = pd.read_csv(config.TRAINING_SET_LABELS)
    data = ds.as_matrix()[:, 1:]
    return data.mean(axis=0)


def try_train_model_xgboost(model_name, fold, load_cache=True):
    with utils.timeit_context('load data'):
        cache_fn = f'../output/prediction_train_frames/{model_name}_{fold}_cache.npz'
        raw_cache_fn = f'../output/prediction_train_frames/{model_name}_{fold}_raw_cache.npz'
        X, y, video_ids = load_train_data(f'../output/prediction_train_frames/{model_name}_{fold}/',
                                          load_cache=load_cache,
                                          cache_fn=cache_fn,
                                          load_raw_cache=True,
                                          raw_cache_fn=raw_cache_fn)

    y_cat = np.argmax(y, axis=1)
    print(X.shape, y.shape)
    print(np.unique(y_cat))

    X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.25, random_state=42)

    model = XGBClassifier(n_estimators=500, objective='multi:softprob', silent=True)
    model.fit(X, y_cat, eval_set=[(X_test, y_test)], early_stopping_rounds=20, verbose=True)

    # model = ExtraTreesClassifier(n_estimators=500, max_features=32)
    # pickle.dump(model, open(f"../output/et_{model_name}.pkl", "wb"))
    # model = pickle.load(open(f"../output/et_{model_name}.pkl", "rb"))

    #
    # print(model)
    # model.fit(X_train, y_train)

    prediction = model.predict_proba(X_test)
    if prediction.shape[1] == 23: # insert mission lion col
        prediction = np.insert(prediction, obj=12, values=0.0, axis=1)
    print(model.score(X_test, y_test))

    y_test_one_hot = np.eye(24)[y_test]
    print(y_test.shape, prediction.shape, y_test_one_hot.shape)
    print(metrics.pri_matrix_loss(y_test_one_hot, prediction))
    print(metrics.pri_matrix_loss(y_test_one_hot, np.clip(prediction, 0.001, 0.999)))
    delta = prediction - y_test_one_hot
    print(np.min(delta), np.max(delta), np.mean(np.abs(delta)), np.sum(np.abs(delta) > 0.5))

    avg_prob = avg_probabilities()
    # print(avg_prob)
    avg_pred = np.repeat([avg_prob], y_test_one_hot.shape[0], axis=0)
    print(metrics.pri_matrix_loss(y_test_one_hot, avg_pred))
    print(metrics.pri_matrix_loss(y_test_one_hot, avg_pred*0.1 + prediction*0.9))


def model_xgboost(model_name, fold, load_cache=True):
    with utils.timeit_context('load data'):
        cache_fn = f'../output/prediction_train_frames/{model_name}_{fold}_cache.npz'
        raw_cache_fn = f'../output/prediction_train_frames/{model_name}_{fold}_raw_cache.npz'
        X, y, video_ids = load_train_data(f'../output/prediction_train_frames/{model_name}_{fold}/',
                                          load_cache=load_cache,
                                          cache_fn=cache_fn,
                                          load_raw_cache=True,
                                          raw_cache_fn=raw_cache_fn)

    y_cat = np.argmax(y, axis=1)
    print(X.shape, y.shape)
    print(np.unique(y_cat))

    model = XGBClassifier(n_estimators=500, objective='multi:softprob', silent=True)
    model.fit(X, y_cat)  # , eval_set=[(X_test, y_test)], early_stopping_rounds=20, verbose=True)
    pickle.dump(model, open(f"../output/xgb_{model_name}_{fold}_full.pkl", "wb"))


def predict_on_test(model_name, fold, use_cache=False):
    model = pickle.load(open(f"../output/xgb_{model_name}_{fold}_full.pkl", "rb"))
    print(model)
    ds = pd.read_csv(config.SUBMISSION_FORMAT)
    classes = list(ds.columns)[1:]
    print(classes)

    data_dir = f'../output/prediction_test_frames/{model_name}_{fold}/'
    with utils.timeit_context('load data'):
        cache_fn = f'../output/prediction_test_frames/{model_name}_{fold}_cache.npy'
        if use_cache:
            X = np.load(cache_fn)
        else:
            X = load_test_data(data_dir, ds.filename)
            np.save(cache_fn, X)
        print(X.shape)
    with utils.timeit_context('predict'):
        prediction = model.predict_proba(X)

    if prediction.shape[1] == 23:
        prediction = np.insert(prediction, obj=12, values=0.0, axis=1)

    for col, cls in enumerate(classes):
        ds[cls] = np.clip(prediction[:, col], 0.001, 0.999)
    os.makedirs('../submissions', exist_ok=True)
    ds.to_csv(f'../submissions/submission_one_model_{model_name}_{fold}.csv', index=False, float_format='%.7f')


def check_corr(sub1, sub2):
    print(sub1, sub2)
    s1 = pd.read_csv('../submissions/' + sub1)
    s2 = pd.read_csv('../submissions/' + sub2)
    for col in s1.columns[1:]:
        print(col, s1[col].corr(s2[col]))

    print('mean ', sub1, sub2, 'sub2-sub1')
    for col in s1.columns[1:]:
        print('{:20}  {:.6} {:.6} {:.6}'.format(col, s1[col].mean(), s2[col].mean(), s2[col].mean() - s1[col].mean()))


def combine_submissions():
    sources = [
        # ('submission_one_model_resnet50_avg_1.csv', 4.0),
        # ('submission_one_model_resnet50_2.csv', 4),
        # ('submission_one_model_resnet50_avg_3.csv', 3),
        # ('submission_one_model_resnet50_avg_4.csv', 4),
        ('submission_one_model_inception_v3_avg_m8_2.csv', 2),
        ('submission_one_model_inception_v3_avg_m8_3.csv', 2),
    ]
    total_weight = sum([s[1] for s in sources])
    ds = pd.read_csv(config.SUBMISSION_FORMAT)
    for src_fn, weight in sources:
        src = pd.read_csv('../submissions/'+src_fn)
        for col in ds.columns[1:]:
            ds[col] += src[col]*weight/total_weight
    ds.to_csv(f'../submissions/submission_11_inception_v3_folds_2_3.csv', index=False, float_format='%.7f')


if __name__ == '__main__':
    with utils.timeit_context('train xgboost model'):
        pass
        # model_xgboost(model_name='resnet50_avg', fold=4, load_cache=False)
        # model_xgboost(model_name='resnet50_avg', fold=1, load_cache=False)
        # model_xgboost(model_name='resnet50', fold=2, load_cache=False)
        # model_xgboost(model_name='inception_v3_avg', fold=3, load_cache=False)
        # model_xgboost(model_name='resnet50_avg', fold=3, load_cache=False)
        # try_train_model_xgboost(model_name='inception_v3_avg', fold=3, load_cache=True)
        # try_train_model_xgboost(model_name='inception_v3_avg_m8', fold=3, load_cache=True)
        # model_xgboost(model_name='inception_v3_avg_m8', fold=3, load_cache=True)
        # model_xgboost(model_name='inception_v3_avg_m8', fold=2, load_cache=False)

    # predict_on_test('resnet50_avg', 1, use_cache=False)
    # predict_on_test('resnet50_avg', 4, use_cache=False)
    # predict_on_test('resnet50', 2, use_cache=False)
    # predict_on_test('inception_v3_avg', 3, use_cache=False)
    # predict_on_test('resnet50_avg', 3, use_cache=False)
    # predict_on_test('inception_v3_avg_m8', 3, use_cache=False)
    # predict_on_test('inception_v3_avg_m8', 2, use_cache=False)

    combine_submissions()
    # check_corr('submission_one_model_resnet50_avg_1.csv', 'submission_one_model_resnet50_avg_4.csv')
    # check_corr('submission_one_model_resnet50_avg_1.csv', 'submission_3_resnet_folds_1_4.csv')
    # check_corr('submission_one_model_resnet50_2.csv', 'submission_3_resnet_folds_1_4.csv')
    # check_corr('submission_one_model_resnet50_2.csv', 'submission_one_model_resnet50_avg_1.csv')
    # combine_submissions()
    # check_corr('submission_5_resnet_folds_1_2_4.csv', 'submission_3_resnet_folds_1_4.csv')
    # check_corr('submission_5_resnet_folds_1_2_4.csv', 'submission_one_model_inception_v3_avg_3.csv')
    # check_corr('submission_5_resnet_folds_1_2_4.csv', 'submission_one_model_resnet50_avg_3.csv')
    # check_corr('submission_one_model_inception_v3_avg_3.csv', 'submission_one_model_resnet50_avg_3.csv')
    # check_corr('submission_5_resnet_folds_1_2_4.csv', 'submission_8_resnet_folds_1_2_3_4.csv')
    check_corr('submission_one_model_inception_v3_avg_m8_2.csv', 'submission_8_resnet_folds_1_2_3_4.csv')



