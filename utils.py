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

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.io import gfile

AUTOTUNE = tf.data.AUTOTUNE


def repeat_experiment(data, param_method, seed):
    repetitions = param_method.get('repetitions', 1)
    for b in range(repetitions):
        resul, columns = experiments(data=data, seed=seed, param_method=param_method)
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
    tau_, mse, bias, var_tau = estimator(data=data,
                                         param_method=param_method,
                                         type=param_method['name_estimator'],
                                         seed=seed)
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
    dataset = dataset.repeat()

    return dataset


def _get_dataset(dataset, batch_size):
    """Prefetch and creates batches of data for base models.

  Args:
    dataset: tf.data.Dataset TFRecord
  Returns:
    dataset: tf.data.Dataset batches
  """
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    return dataset
