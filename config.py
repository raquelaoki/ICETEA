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

"""config!

config files creation of objects, and organization

"""

import itertools
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

import estimators

#from icetea import estimators

IMAGE_SIZE = [256, 256]
TensorDict = Dict[str, tf.Tensor]


class MakeParameters:
    """Make Parameters Objects.

  Args:
    - configs_data: dictionary with data parameters.
    - configs_methods: dictionary with methods parameters.
  Return:
    parameters: methods parameteres as objects.
  """

    def __init__(self, configs_data, configs_methods, use_tpus=False, strategy=None):
        super(MakeParameters, self).__init__()

        #configs_data['is_Image'] = configs_data.get('is_Image', False)

        self.config_data = _create_configs(configs_data)

        self.config_methods = _read_method_configs(
            config_methods=_create_configs(configs_methods),
            use_tpus=use_tpus,
            strategy=strategy,
        )


def _create_configs(dict_):
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
    keys = dict_.keys()
    vals = dict_.values()
    configs = []
    for instance in itertools.product(*vals):
        configs.append(dict(zip(keys, instance)))
    return configs


def _read_method_configs(config_methods, use_tpus=False, strategy=None):
    """Creates list of dictionaries.

  Each dict is a set of config parameters for the methods.
  Args:
    config_methods: list of dictionaries
    data_name: string with data name
    setting: quick (for testing), samples or covariates analysies
  Returns:
     list with all config parameters, some of them are objects
  """
    parameters_method = []
    for config in config_methods:
        parameters = config

        parameters['estimator'] = estimators.estimator
        parameters['base_model'] = _base_image_model_construction(model_config={'name_base_model':
                                                                                    parameters['name_base_model']},
                                                                  use_tpus=use_tpus,
                                                                  strategy=strategy)
        parameters['metric'] = _metric_function(parameters['name_metric'])
        parameters['prop_score'] = _prop_score_function(parameters['name_prop_score'])

        parameters_method.append(parameters)
    return parameters_method


def _base_image_model_construction(model_config, use_tpus=False, strategy=None):
    """Constructs the image base model.

  Args:
    model_config: dicstionary with parameters
  Returns:
    model: Model object
  """
    name_base_model = model_config.get('name_base_model', 'inceptionv3')
    model_config['weights'] = 'imagenet'
    model_config['input_shape'] = (256, 256, 3)

    if use_tpus:
        with strategy.scope():
            if name_base_model == 'inceptionv3':
                model = image_model_inceptionv3(model_config)
                initial_learning_rate = 0.001
            elif name_base_model == 'resnet50':
                model = image_model_resnet50(model_config)
                initial_learning_rate = 0.001
            elif name_base_model == 'image_regression':
                model = image_model_regression(model_config)
                initial_learning_rate = 0.01
            else:
                raise NotImplementedError(
                    'Estimator not supported:{}'.format(name_base_model))

            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate, decay_steps=30, decay_rate=0.9, staircase=True)

            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                loss='mean_squared_error',
                metrics=['mse', 'mae'])
    else:
        if name_base_model == 'inceptionv3':
            model = image_model_inceptionv3(model_config)
            initial_learning_rate = 0.001
        elif name_base_model == 'resnet50':
            model = image_model_resnet50(model_config)
            initial_learning_rate = 0.001
        elif name_base_model == 'image_regression':
            model = image_model_regression(model_config)
            initial_learning_rate = 0.01
        else:
            raise NotImplementedError(
                'Estimator not supported:{}'.format(name_base_model))

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


def _prop_score_function(name_prop_score):
    if name_prop_score == 'LogisticRegression':
        return linear_model.LogisticRegression()
    elif name_prop_score == 'LogisticRegression_NN':
        return _LogisticRegressionNN()
    else:
        raise NotImplementedError(
            'Estimator not supported:{}'.format(name_prop_score))


class _LogisticRegressionNN:
    """Make NN version of Logistic Regression.

  Used as Propensity Score Model
  """

    def __init__(self):
        super(_LogisticRegressionNN, self).__init__()
        self.model = self._logistic_regression_architecture()

    def fit(self, data):
        """Fits a Classification Model.

    Args:
      data: prefetch batches 16 [B, H, W, C], not repeated, not shuffled.
    """
        self.model.compile(
            optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        epochs = 10

        self.model.summary()
        self.model.fit(data, epochs=epochs, verbose=2, steps_per_epoch=20)

    def predict_proba(self, data, step_lim=50):
        """Predict Probability of each class.

    Args:
      data: tf.data.Dataset
    Returns:
      predict: predictions array
    """
        t_pred = []
        t = []
        for i, (batch_x, batch_t) in enumerate(data):
            t_pred.append(self.model.predict_on_batch(batch_x))
            t.append(batch_t.numpy())
            if step_lim > i:
                break

        t_pred = np.concatenate(t_pred).ravel().reshape(-1, 2)
        return t_pred

    def _logistic_regression_architecture(self):
        """Implements of Propensity Score.

        It takes as input tensors of shape [B, H, W, C] and outputs [B,Y]
        Returns:
          model: NN object
        """
        # A simple logistic regression implemented as NN.
        inputs = tf.keras.Input(shape=[*IMAGE_SIZE, 3], name='LogisticRegression')

        backbone = tf.keras.applications.ResNet50(
            include_top=False,
            weights='imagenet',
            input_shape=(*IMAGE_SIZE, 3),
            pooling= 'avg', )
        backbone_drop_rate = 0.2

        inputs = tf.keras.Input(shape=[*IMAGE_SIZE, 3], name='image')
        hid = backbone(inputs)
        hid = tf.keras.layers.Dropout(backbone_drop_rate)(hid)

        outputs = tf.keras.layers.Dense(2, activation='softmax', use_bias=True,
                                        kernel_regularizer=regularizers.l1_l2(
                                            l1=1e-5,
                                            l2=1e-4),
                                        bias_regularizer=regularizers.l2(1e-4),
                                        activity_regularizer=regularizers.l2(1e-5)
                                        )(hid)
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name='LogisticRegression')

        return model


def image_model_regression(model_config):
    """Implements a one-layer NN that mimics a Linear Regression.

  Args:
    model_config: dictionary with parameters
  Returns:
    model: Model object
  """
    # A simple linear regression implemented as NN.
    last_activation = model_config.get('activation', 'linear')

    inputs = tf.keras.Input(shape=[*IMAGE_SIZE, 3], name='image_regression')
    x = tf.keras.layers.Flatten()(inputs)
    outputs = tf.keras.layers.Dense(1, activation=last_activation, use_bias=True,
                                    kernel_regularizer=regularizers.l1_l2(
                                        l1=1e-5,
                                        l2=1e-4),
                                    bias_regularizer=regularizers.l2(1e-4),
                                    activity_regularizer=regularizers.l2(1e-5))(x)
    model = tf.keras.Model(
        inputs=inputs, outputs=outputs, name='image_regression')
    return model


def image_model_inceptionv3(model_config):
    """Implements inceptionV3 NN model.

  Args:
    model_config: dictionary with parameters
  Returns:
    model: Model object
  """
    last_activation = model_config.get('activation', 'linear')
    backbone = tf.keras.applications.InceptionV3(
        include_top=False,
        weights=model_config.get('weights', 'imagenet'),
        input_shape=(*IMAGE_SIZE, 3),
        pooling=model_config.get('pooling', 'avg'))

    backbone_drop_rate = model_config.get('backbone_drop_rate', 0.2)

    inputs = tf.keras.Input(shape=[*IMAGE_SIZE, 3], name='image')
    hid = backbone(inputs)
    hid = tf.keras.layers.Dropout(backbone_drop_rate)(hid)
    outputs = tf.keras.layers.Dense(1, activation=last_activation,
                                    use_bias=True)(hid)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='inceptionv3')

    return model


def image_model_resnet50(model_config):
    """Implements Resnet NN model.

  Reference:
  https://keras.io/api/applications/#usage-examples-for-image-classification-models

  Args:
    model_config: dictionary with parameters
  Returns:
    model: Model object
  """
    last_activation = model_config.get('activation', 'linear')
    backbone = tf.keras.applications.ResNet50(
        include_top=False,
        weights=model_config.get('weights', 'imagenet'),
        input_shape=(*IMAGE_SIZE, 3),
        pooling=model_config.get('pooling', 'avg'),)

    backbone_drop_rate = model_config.get('backbone_drop_rate', 0.2)

    inputs = tf.keras.Input(shape=[*IMAGE_SIZE, 3], name='image')

    hid = backbone(inputs)
    hid = tf.keras.layers.Dropout(backbone_drop_rate)(hid)
    #x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(1, activation=last_activation,
                                    use_bias=True)(hid)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='resnet50')
    return model


