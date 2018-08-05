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
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import PCA
from accuracy import Accuracy

import scipy as sc
from time import time


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
            image = np.pad(sample.image, (padding * 2,),
                           'constant', constant_value=0)
            kps = []
            for landmark in sample.landmarks:
                kps.append(cv.KeyPoint(
                    landmark[0] + padding, landmark[1] + padding, radius))
            kp, descr = orb.detect(image, kps)

            pwdist = sc.spatial.distance.pdist(np.asarray(sample.landmarks))
            feat.append(np.hstack((descr.ravel(), pwdist.ravel())))

            targets.append(clip.labels)

    print('feat count:', len(feat))
    return np.asarray(feat, dtype=np.float32), np.asarray(targets, dtype=np.float32)


def classification(X_train, X_test, y_train, y_test, accuracy_fn, pca_dim):
    if pca_dim > 0:
        pass
        # TODO: выполните сокращение размерности признаков с использованием PCA

    # shuffle
    combined = list(zip(X_train, y_train))
    random.shuffle(combined)
    X_train[:], y_train[:] = zip(*combined)

    # TODO: используйте классификаторы из sklearn

    y_pred = []
    accuracy_fn.by_frames(y_pred)
    accuracy_fn.by_clips(y_pred)


if __name__ == "__main__":
    experiment_name = 'ORB_not_sep_rad5_pdist'
    max_num_clips = 0  # загружайте только часть данных для отладки кода
    use_dump = False  # используйте dump для быстрой загрузки рассчитанных фич из файла

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
                              train_file_list, max_num_clips=0)
        test_data = get_data(
            test_dataset_root, test_file_list, max_num_clips=0)

        # get features
        train_feat, train_targets = calc_features(train_data)
        test_feat, test_targets = calc_features(test_data)

        accuracy_fn = Accuracy(test_data, experiment_name=experiment_name)

        # with open(experiment_name + '.pickle', 'wb') as f:
        #    pickle.dump([train_feat, train_targets, test_feat, test_targets, accuracy_fn], f, protocol=2)
    else:
        with open(experiment_name + '.pickle', 'rb') as f:
            train_feat, train_targets, test_feat, test_targets, accuracy_fn = pickle.load(
                f)

    # run classifiers
    classification(train_feat, test_feat, train_targets,
                   test_targets, accuracy_fn=accuracy_fn, pca_dim=100)
