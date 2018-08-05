# -*- coding: utf-8 -*-
import os
import sys
import cv2 as cv
import random
import numpy as np
from tqdm import tqdm
import pickle

sys.path.append('../')
sys.path.append(os.getcwd())
from pytorch.common.datasets_parsers.av_parser import AVDBParser

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import PCA
from accuracy import Accuracy

import scipy as sc
from time import time
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier


def get_data(dataset_root, file_list, max_num_clips=0, max_num_samples=50):
    dataset_parser = AVDBParser(dataset_root, os.path.join(dataset_root, file_list),
                                max_num_clips=max_num_clips, max_num_samples=max_num_samples,
                                ungroup=False, load_image=True)
    data = dataset_parser.get_data()
    print('clips count:', len(data))
    print('frames count:', dataset_parser.get_dataset_size())
    return data


def calc_features(data):
    radius = 5
    padding = radius * 2
    orb = cv.ORB_create(edgeThreshold=0)

    progresser = tqdm(iterable=range(0, len(data)),
                      desc='calc video features',
                      total=len(data),
                      unit='files')

    feat, targets = [], []
    for i in progresser:
        clip = data[i]
        for sample in clip.data_samples:
            image = np.pad(sample.image, ((padding * 2, padding * 2), (padding * 2, padding * 2), (0, 0)),
                           'constant', constant_values=0)
            image = image.astype(dtype=np.uint8)
            kps = []
            for landmark in sample.landmarks:
                kps.append(cv.KeyPoint(
                    landmark[0] + padding, landmark[1] + padding, radius))
            kp, descr = orb.compute(image, kps)

            pwdist = sc.spatial.distance.pdist(np.asarray(sample.landmarks))
            feat.append(np.hstack((descr.ravel(), pwdist.ravel())))

            targets.append(clip.labels)

    print('feat count:', len(feat))
    return np.asarray(feat, dtype=np.float32), np.asarray(targets, dtype=np.float32)


def report(results, n_top=5):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.8f} (std: {1:.8f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


def classification(X_train, X_test, y_train, y_test, accuracy_fn, pca_dim):
    if pca_dim > 0:
        pass
        # TODO: выполните сокращение размерности признаков с использованием PCA

    # shuffle
    # combined = list(zip(X_train, y_train))
    # random.shuffle(combined)
    # X_train[:], y_train[:] = zip(*combined)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # TODO: используйте классификаторы из sklearn
    # sgd_clf = SGDClassifier(random_state=42, max_iter=5, loss='log')
    # iterations = 20
    # param_dist = {"penalty": ['l2'],
    #               "alpha": np.arange(0.01, 0.03, (0.03 - 0.01) / 100),
    #               "l1_ratio": np.arange(0.15, 0.3, (0.3 - 0.15) / 10),
    #               "class_weight": [None, 'balanced'],
    #               "average": [True],
    #               }

    # search_seed = 42
    # random_search = RandomizedSearchCV(sgd_clf, param_distributions=param_dist,
    #                                    n_iter=iterations, cv=StratifiedKFold(
    #                                        n_splits=4, shuffle=True, random_state=42),
    #                                    random_state=search_seed, verbose=2, n_jobs=4, scoring='f1_macro')

    # start = time()
    # random_search.fit(X_train_scaled, y_train)
    # print("RandomizedSearchCV took %.2f seconds for %d candidates"
    #       " parameter settings." % ((time() - start), iterations))
    # report(random_search.cv_results_)
    # params = random_search.best_params_

    ###### REPORT from RandomizedSearchCV ######
    # Model with rank: 1
    # Mean validation score: 0.79870922 (std: 0.00336902)
    # Parameters: {'penalty': 'l2', 'l1_ratio': 0.2549999999999999,
    #             'class_weight': None, 'average': True, 'alpha': 0.010800000000000002}

    # Model with rank: 2
    # Mean validation score: 0.79861163 (std: 0.00387216)
    # Parameters: {'penalty': 'l2', 'l1_ratio': 0.15, 'class_weight': 'balanced',
    #             'average': True, 'alpha': 0.011200000000000003}

    # Model with rank: 3
    # Mean validation score: 0.79823906 (std: 0.00367094)
    # Parameters: {'penalty': 'l2', 'l1_ratio': 0.16499999999999998,
    #             'class_weight': None, 'average': True, 'alpha': 0.011200000000000003}

    params = {'penalty': 'l2', 'l1_ratio': 0.2549999999999999,
              'class_weight': None, 'average': True, 'alpha': 0.010800000000000002
              }
    sgd_clf = SGDClassifier(**params, random_state=42,
                            max_iter=5, loss='log', n_jobs=4, verbose=10)

    # sgd_clf = SGDClassifier(n_jobs=4, penalty='l2', loss='log', alpha=0.019,
    #                         l1_ratio=0.25, class_weight=None, average=True,
    #                         fit_intercept=True, max_iter=5, verbose=10, random_state=42)

    sgd_clf.fit(X_train_scaled, y_train)
    y_pred = sgd_clf.predict(X_test_scaled)
    accuracy_fn.by_frames(y_pred)
    accuracy_fn.by_clips(y_pred)


if __name__ == "__main__":
    experiment_name = 'ORB_not_sep_rad5_pdist'
    max_num_clips = 0  # загружайте только часть данных для отладки кода
    use_dump = True  # используйте dump для быстрой загрузки рассчитанных фич из файла

    # dataset dir
    base_dir = 'C:/Files/Datasets/SummerSchool STC'
    if 1:
        train_dataset_root = base_dir + '/Ryerson/Video'
        train_file_list = base_dir + '/Ryerson/train_data_with_landmarks.txt'
        test_dataset_root = base_dir + '/Ryerson/Video'
        test_file_list = base_dir + '/Ryerson/test_data_with_landmarks.txt'
    elif 1:
        train_dataset_root = base_dir + \
            '/OMGEmotionChallenge-master/omg_TrainVideos/preproc/frames'
        train_file_list = base_dir + \
            '/OMGEmotionChallenge-master/omg_TrainVideos/preproc/train_data_with_landmarks.txt'
        test_dataset_root = base_dir + \
            '/OMGEmotionChallenge-master/omg_ValidVideos/preproc/frames'
        test_file_list = base_dir + \
            '/OMGEmotionChallenge-master/omg_ValidVideos/preproc/valid_data_with_landmarks.txt'

    if not use_dump:
        # load dataset
        train_data = get_data(train_dataset_root,
                              train_file_list, max_num_clips=max_num_clips)
        test_data = get_data(
            test_dataset_root, test_file_list, max_num_clips=max_num_clips)

        # get features
        train_feat, train_targets = calc_features(train_data)
        test_feat, test_targets = calc_features(test_data)

        accuracy_fn = Accuracy(test_data, experiment_name=experiment_name)

        with open(experiment_name + '.pickle', 'wb') as f:
            pickle.dump([train_feat, train_targets, test_feat,
                         test_targets, accuracy_fn], f, protocol=2)
    else:
        with open(experiment_name + '.pickle', 'rb') as f:
            train_feat, train_targets, test_feat, test_targets, accuracy_fn = pickle.load(
                f)

    # run classifiers
    classification(train_feat, test_feat, train_targets,
                   test_targets, accuracy_fn=accuracy_fn, pca_dim=100)
