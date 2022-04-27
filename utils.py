# coding=utf-8
# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utils!

These classes are responsible for simulating data and organizing the experiments
outputs. It does not depend on the pipeline choice.

DataSimulation class: assumes one binary treatment, and continuous target.

Experiments class: fits the estimator, return metrics
"""
import math
import os
import time
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.io import gfile

TensorDict = Dict[str, tf.Tensor]
AUTOTUNE = tf.data.AUTOTUNE


# BATCH_SIZE = 16
# IMAGE_SIZE = [587, 587]


class NonImageData:
    """Data Simulation.

  Description: Class to organize, reads/simulate the data, and constructs the
  dataset obj.
  Check the README file for references about the data generation.

  Attr:
    seed: int, random seed
    param_data: dictionary, data parameters

  Returns:
    Creates DataSimulation obj
  """

    def __init__(self, seed, param_data):
        super(DataSimulation, self).__init__()
        self.seed = seed
        self.param_data = param_data
        self.name = param_data['name']
        self.is_Image = param_data.get('is_Image', False)
        self.splitted = False
        if self.name == 'IHDP':
            self._load_ihdp()
        elif self.name == 'ACIC':
            self._load_acic()
        else:
            raise NotImplementedError('Dataset not supported:{}'.format(self.name))
        self._split()

    def _split(self):
        if not self.splitted:
            self._treated_samples()
            self._control_samples()
            self.splitted = True

    def _treated_samples(self):
        if not self.splitted:
            self.outcome_treated = self.outcome[self.treatment == 1].ravel()
            self.covariates_treated = self.covariates[self.treatment == 1, :]
        else:
            return self.outcome_treated, self.covariates_treated

    def _control_samples(self):
        if not self.splitted:
            self.outcome_control = self.outcome[self.treatment == 0].ravel()
            self.covariates_control = self.covariates[self.treatment == 0, :]
        else:
            return self.outcome_control, self.covariates_control

    def print_shapes(self):
        print('Print Shapes')
        print(self.outcome.shape, self.treatment.shape, self.covariates.shape)
        if self.splitted:
            print(self.outcome_control.shape, self.covariates_control.shape)
            print(self.outcome_treated.shape, self.covariates_treated.shape)

    def _load_ihdp(self):
        """Loads semi-synthetic data.

    It updates the object DataSimulation.

    Args:
      self
    Returns:
      None
    """
        self.data_path = self.param_data['data_path'] + 'IHDP/'
        # Reference: https://github.com/AMLab-Amsterdam/CEVAE
        # each iteration, it randomly pick one of the 10 existing repetitions
        np.random.seed(self.seed)

        i = np.random.randint(1, 10, 1)[0]
        path = self.data_path + '/ihdp_npci_' + str(i) + '.csv.txt'
        with gfile.GFile(path, 'r') as f:
            data = np.loadtxt(f, delimiter=',')

        self.outcome, y_cf = data[:, 1][:, np.newaxis], data[:, 2][:, np.newaxis]
        self.outcome = self.outcome.ravel()
        self.treatment = data[:, 0].ravel()
        self.covariates = data[:, 5:]
        scaler = StandardScaler()
        self.covariates = scaler.fit_transform(self.covariates)

        self.sample_size, self.num_covariates = self.covariates.shape
        self.linear, self.noise = False, False
        self.var_covariates = None
        self.treatment_prop = self.treatment.sum() / len(self.treatment)

        # y1, y0 = self.outcome, self.outcome
        y1 = [
            y_cf[j][0] if item == 0 else self.outcome[j]
            for j, item in enumerate(self.treatment)
        ]
        y0 = [
            y_cf[j][0] if item == 1 else self.outcome[j]
            for j, item in enumerate(self.treatment)
        ]
        y1 = np.array(y1)
        y0 = np.array(y0)
        self.tau = (y1 - y0).mean()

    def _load_acic(self):
        """Loads semi-synthetic data.

        It updates the object DataSimulation.

        Args:
          self
        Returns:
          None
        """
        self.data_path = self.param_data['data_path'] + 'ACIC/'
        if self.param_data['data_low_dimension']:
            true_ate_path = self.data_path + 'lowDim_trueATE.csv'
            self.data_path = self.data_path + 'low_dimensional_datasets/'
        else:
            true_ate_path = self.data_path + 'highDim_trueATE.csv'
            self.data_path = self.data_path + 'high_dimensional_datasets/'

        np.random.seed(self.seed)
        i = np.random.randint(0, len(gfile.listdir(self.data_path)), 1)[0]

        path = gfile.listdir(self.data_path)[i]
        with gfile.GFile(self.data_path + path, 'r') as f:
            data = pd.read_csv(f, delimiter=',')

        self.outcome = data['Y'].values
        self.treatment = data['A'].values
        self.covariates = data.drop(['Y', 'A'], axis=1).values
        scaler = StandardScaler()
        self.covariates = scaler.fit_transform(self.covariates)

        self.sample_size, self.num_covariates = self.covariates.shape
        self.linear, self.noise = False, False
        self.var_covariates = None
        self.treatment_prop = self.treatment.sum() / len(self.treatment)

        with gfile.GFile(true_ate_path, 'r') as f:
            true_ate = pd.read_csv(f, delimiter=',')

        path = path[:-4]
        true_ate_row = true_ate[true_ate['filename'] == path]
        self.tau = true_ate_row['trueATE'].values[0]


def repeat_experiment(data, param_method):
    repetitions = param_method.get('repetitions', 1)
    for b in range(repetitions):
        resul, columns = experiments(data=data, seed=b, param_method=param_method)
        if b == 0:
            tab = pd.DataFrame(columns=columns)
            tab = tab.append(resul, ignore_index=True)
        else:
            tab = tab.append(resul, ignore_index=True)
    return tab


def experiments(data, seed, param_method):
    """Function to run experiments.
  Args:
    data: DataSimulation obj.
    seed: currently not used.
    param_method: dictionary with estimator's parameters.
  Returns:
    Dictionary with simulation results.
    col names
  """
    # del seed
    start = time.time()
    estimator = param_method['estimator']
    param_grid = param_method['param_grid']
    tau_, mse, bias, var_tau = estimator(data, param_method, param_grid,
                                         type=param_method['name_estimator'],
                                         seed=seed)
    if data.is_Image:
        tab = {
            't_est': tau_,
            'mse0': mse[0],
            'mse1': mse[1],
            'bias0': bias[0],
            'bias1': bias[1],
            'variance': var_tau,
            'name': data.name,
            'seed': seed,
            'method_estimator': param_method['name_estimator'],
            'method_base_model': param_method['name_base_model'],
            'method_metric': param_method['name_metric'],
            'time': time.time() - start,
        }
    else:
        tab = {
            't_est': tau_,
            't_real': data.tau,
            'mae': np.abs(data.tau - tau_),
            'mse0': mse[0],
            'mse1': mse[1],
            'bias0': bias[0],
            'bias1': bias[1],
            'variance': var_tau,
            'name': data.name,
            'data_n': data.sample_size,
            'data_num_covariates': data.num_covariates,
            'data_noise': data.noise,
            'data_linear': data.linear,
            'data_treatment_prop': np.sum(data.treatment) / data.sample_size,
            'method_estimator': param_method['name_estimator'],
            'method_base_model': param_method['name_base_model'],
            'method_metric': param_method['name_metric'],
            'method_prop_score': param_method['name_prop_score'],
            'time': time.time() - start,
        }

    return tab, list(tab.keys())


class ImageData:
    """Load image dataset.

    The path to the folder determines the type of outcome (clinical or simulated).
    param_data={'name':'kagle_retinal',
              'path_tfrecords':str,
              'prefix_train':str,
              'image_size':[s,s],
              'batch_size':int
     }

     seed = sim_bx_y_val_val_val
    """

    def __init__(self, seed, param_data):
        super(ImageData, self).__init__()
        self.name = param_data['name']
        batch_size = param_data['batch_size']
        features = {'image/encoded': tf.io.FixedLenFeature([], dtype=tf.string),
                    'image/id': tf.io.FixedLenFeature([], dtype=tf.string),
                    f'image/{seed}-pi': tf.io.FixedLenFeature([1], tf.float32),
                    f'image/{seed}-y': tf.io.FixedLenFeature([1], tf.float32),
                    f'image/{seed}-mu0': tf.io.FixedLenFeature([1], tf.float32),
                    f'image/{seed}-mu1': tf.io.FixedLenFeature([1], tf.float32)
                    }

        # path = param_data['data_path']
        filenames = tf.io.gfile.glob(param_data['path_tfrecords'] + param_data['prefix_train'] + '*.tfrec')
        assert len(filenames) > 0, 'No files found! Check path and prefix'
        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.map(_get_parse_example_fn(features), num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(_decode_img, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(lambda x: _filter_treatment(x, seed),
                              num_parallel_calls=tf.data.AUTOTUNE)

        # self.dataset = _get_dataset(dataset, batch_size=batch_size)

        # split treated and non treated and pred (for full conterfactual).
        ds_treated = dataset.filter(lambda x: x['t'])
        ds_control = dataset.filter(lambda x: not x['t'])

        ds_treated = ds_treated.map(lambda x: _filter_cols(x, seed),
                                    num_parallel_calls=tf.data.AUTOTUNE)
        ds_control = ds_control.map(lambda x: _filter_cols(x, seed),
                                    num_parallel_calls=tf.data.AUTOTUNE)

        ds_all = dataset.map(lambda x: _filter_cols_pred(x, seed),
                             num_parallel_calls=tf.data.AUTOTUNE)
        ds_all_ps = dataset.map(lambda x: _filter_cols_ps(x, seed),
                                num_parallel_calls=tf.data.AUTOTUNE)

        self.dataset_treated = _get_dataset(ds_treated, batch_size=batch_size)
        self.dataset_control = _get_dataset(ds_control, batch_size=batch_size)
        self.dataset_all = _get_dataset(ds_all, batch_size=batch_size)
        self.dataset_all_ps = _get_dataset_ps(ds_all_ps, batch_size=batch_size)
        self.is_Image = param_data.get('is_Image', True)
        self.b = param_data.get('b', 1)

    def make_plot(self):
        batch = next(iter(self.dataset_treated))
        plt.imshow(batch[0][0])


def _get_parse_example_fn(features):
    """Returns a function that parses a TFRecord.

  Args:
    features: dict with features for the TFRecord.
  Returns:
    _parse_example
  """

    def _parse_example(example):
        return tf.io.parse_single_example(example, features)

    return _parse_example


def _decode_img(example):
    image_size = [256, 256]
    image = tf.image.decode_jpeg(example['image/encoded'], channels=3)
    image = tf.cast(image, tf.float32) / 255.0  # [0,255] -> [0,1]
    image = tf.image.resize(image, image_size)
    example['image/encoded'] = image
    # image, example['image/target'], example['image/id']
    return example


def _filter_cols_ps(dataset, seed):
    """Mapping function.

  Filter columns for propensity score batch.
  Args:
    dataset: tf.data.Dataset with several columns.
    seed: int
  Returns:
    dataset: tf.data.Dataset with two columns (X,T).
  """
    t_name = f'image/{seed}-pi'
    return dataset['image/encoded'], dataset[t_name]


def _filter_cols_pred(dataset, seed):
    """Mapping function.

  Filter columns for predictions batch.
  Args:
    dataset: tf.data.Dataset with several columns.
    seed: int.
  Returns:
    dataset: tf.data.Dataset with three columns (X,Y,T).
  """
    col_y = f'image/{seed}-y'
    return dataset['image/encoded'], dataset[col_y], dataset['t']


def _filter_treatment(dataset, seed):
    """Mapping function.

  Constructs bool variable (treated = True, control = False)
  Args:
    dataset: tf.data.Dataset
    seed: int
  Returns:
    dataset: tf.data.Dataset
  """
    t = False
    if dataset[f'image/{seed}-pi'] == 1:
        t = True
    dataset['t'] = t
    return dataset


def _filter_cols(dataset, seed):
    """Mapping function.

  Filter columns for batch.
  Args:
    dataset: tf.data.Dataset with several columns.
    seed: int
  Returns:
    dataset: tf.data.Dataset with two columns (X, Y).
  """
    col_y = f'image/{seed}-y'
    return dataset['image/encoded'], dataset[col_y]


def _get_dataset_ps(dataset, batch_size):
    """Prefetch and creates batches of data for the propensity score.

  Args:
    dataset: tf.data.Dataset TFRecord
  Returns:
    dataset: tf.data.Dataset batches
  """

    def _preprocessing_ps(batch0, batch1):
        batch1 = tf.reshape(batch1, [-1])
        batch1 = tf.cast(batch1, tf.int32)
        batch1 = tf.one_hot(batch1, 2)
        return batch0, batch1

    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size).map(_preprocessing_ps)

    return dataset


def _get_dataset(dataset, batch_size):
    """Prefetch and creates batches of data for base models.

  Args:
    dataset: tf.data.Dataset TFRecord
  Returns:
    dataset: tf.data.Dataset batches
  """
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    return dataset
