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
#
"""Parameters Check
Check the consistency of the parameters and complete some missing fields based on conditions.
"""
import logging
import pandas as pd
import yaml
import os
import itertools
from inspect import getmembers, isfunction, isclass
from typing import Dict
import numpy as np
from sklearn import ensemble
from sklearn import linear_model
from sklearn import metrics as sk_metrics
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

# import estimators
import estimators.base_models as base_models
import estimators.propensity_score_models as ps_models
import estimators.all_estimators as est

# from icetea import estimators

IMAGE_SIZE = [256, 256]
TensorDict = Dict[str, tf.Tensor]

logger = logging.getLogger(__name__)


def consistency_check_feature_extractor(config):
    # Checking if required folders exist.
    check_folders_exist = [config['path_images_png'],
                           config['path_tfrecords'],
                           config['path_features'],
                           ]
    _check_folders(path_root=config['path_root'], check_folders_exist=check_folders_exist)

    # Checking parameters consistency.
    assert 0.1 < config['xfe_proportion_size'] < 0.9, 'xfe_proportion_size should be between 0.1 and 0.9!'
    assert config['epochs'] > 0, 'Number of epochs must be >0.'
    return config


def consistency_check_data_simulation(config):
    # Checking if required folders exist.
    check_folders_exist = [config['path_root'],
                           config['path_tfrecords'],
                           config['path_features'],
                           config['path_tfrecords_new'],
                           config['path_results'],
                           ]
    _check_folders(path_root=config['path_root'], check_folders_exist=check_folders_exist)

    # Checking parameters consistency.
    assert config['b'] > 0, 'Number of repetitions (b) must be >0.'
    assert len(config['knobs']) == 3, 'Lenght of knobs must be 3.'
    return config


def consistency_check_causal_inference(config):
    check_folders_exist = [config['path_root'],
                           config['path_features'],
                           config['path_tfrecords_new'],
                           config['path_results'],
                           ]
    _check_folders(path_root=config['path_root'], check_folders_exist=check_folders_exist)
    check_file = os.path.join(config['path_root'], config['path_features'], 'true_tau.csv')
    assert tf.io.gfile.exists(check_file), 'true_tau missing!'

    return config


def _check_folders(path_root, check_folders_exist):
    check_folders_exist = [os.path.join(path_root, path) for path in check_folders_exist]
    for path in check_folders_exist:
        assert tf.io.gfile.isdir(path), path + ': Folder does not exist!'


def create_configs(config):
    """Creates a list of dictionaries with the parameters.

    Expands all possible combinations between lists inside dictionary.
    Example: dict1 = {a=['data1','data2'], b=['model1','model2']}
    output= [{a='data1',b='model1'}, {a='data1',b='model2'},
    {a='data2',b='model1'}, {a='data2',b='model2'}].

    Args:
    dict_: dictionary with the lists.
    Returns:
    list of config files.
    """
    keys = config.keys()
    vals = config.values()
    configs = []
    for instance in itertools.product(*vals):
        configs.append(dict(zip(keys, instance)))
    return configs


def create_methods_obj(config, use_tpu=False, strategy=None):
    config['estimator'] = _estimator_construction(model_config=config)
    config['base_model'] = _base_image_model_construction_with_strategy(model_config=config,
                                                                        use_tpus=use_tpu,
                                                                        strategy=strategy)
    config['metric'] = _metric_function(config['name_metric'])
    if config['learn_prop_score']:
        config['prop_score'] = _prop_score_function(config['name_prop_score'], config['image_size'])
    return config


def consistency_check_causal_methods(config):
    def _raise_not_implemented(item_implemented, item_called):
        for item in item_called:
            if item not in item_implemented:
                raise NotImplementedError(
                    item + 'is not currently implemented. Check the documentation on how to add it.')

    # Recover implemented methods based on functions inside py files.
    implemented_estimators = getmembers(estimators, isfunction)
    implemented_estimators = [e[0] for e in implemented_estimators]
    filter = np.array([e.startswith('estimator_') for e in implemented_estimators])
    implemented_estimators = np.array(implemented_estimators)
    implemented_estimators = implemented_estimators[filter]
    implemented_estimators = [e.replace('estimator_', '') for e in implemented_estimators]

    implemented_base_models = getmembers(base_models, isfunction)
    implemented_base_models = [bm[0].replace('image_model_', '') for bm in implemented_base_models]

    implemented_prop_score_model = getmembers(ps_models, isclass)
    implemented_prop_score_model = [bm[0].replace('PS_', '') for bm in implemented_prop_score_model]

    implemented_metrics = ['mse']

    _raise_not_implemented(implemented_estimators, config['name_estimator'])
    _raise_not_implemented(implemented_base_models, config['name_base_model'])
    _raise_not_implemented(implemented_prop_score_model, config['name_prop_score'])
    _raise_not_implemented(implemented_metrics, config['name_metric'])


def _estimator_construction(model_config):
    if model_config['name_estimator'] == 'aipw':
        return est.estimator_aipw
    elif model_config['name_estimator'] == 'kob':
        return est.estimator_kob
    else:
        raise NotImplementedError("Please, add your new method in the helper_parameters._estimator_construction")


def _base_image_model_construction_with_strategy(model_config, use_tpus=False, strategy=None):
    if use_tpus:
        with strategy.scope():
            model = _base_image_model_construction(model_config)
    else:
        model = _base_image_model_construction(model_config)
    return model


def _base_image_model_construction(model_config):
    """Constructs the image base model.

  Args:
    model_config: dicstionary with parameters
  Returns:
    model: Model object
  """
    model_config['input_shape'] = (model_config['image_size'], model_config['image_size'], 3)
    initial_learning_rate = model_config.get('initial_learning_rate', 0.001)

    if model_config['name_base_model'] == 'inceptionv3':
        model = base_models.image_model_inceptionv3(model_config)
    if model_config['name_base_model'] == 'resnet50':
        model = base_models.image_model_resnet50(model_config)
    if model_config['name_base_model'] == 'image_regression':
        model = base_models.image_model_image_regression(model_config)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=30, decay_rate=0.9, staircase=True)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss='mean_squared_error',
        metrics=['mse', 'mae'])

    return model


def _metric_function(name_metric):
    if name_metric == 'mse':
        return sk_metrics.mean_squared_error
    else:
        raise NotImplementedError(
            'Estimator not supported:{}'.format(name_metric))


def _prop_score_function(name_prop_score, image_size):
    if name_prop_score == 'LogisticRegression_NN':
        return ps_models.PS_LogisticRegression_NN(image_size)
    else:
        raise NotImplementedError(
            'Estimator not supported:{}'.format(name_prop_score))
