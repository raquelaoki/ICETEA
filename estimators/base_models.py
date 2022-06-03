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

# """config!
#
# config files creation of objects, and organization
#
# """

import itertools
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


def image_model_image_regression(model_config):
    """Implements a one-layer NN that mimics a Linear Regression.

  Args:
    model_config: dictionary with parameters
  Returns:
    model: Model object
  """
    # A simple linear regression implemented as NN.
    last_activation = model_config.get('activation', 'linear')
    image_size = [model_config['image_size'], model_config['image_size']]
    inputs = tf.keras.Input(shape=[*image_size, 3], name='image_regression')
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
    image_size = [model_config['image_size'], model_config['image_size']]
    backbone = tf.keras.applications.InceptionV3(
        include_top=False,
        weights=model_config.get('weights', 'imagenet'),
        input_shape=(*image_size, 3),
        pooling=model_config.get('pooling', 'avg'))

    backbone_drop_rate = model_config.get('backbone_drop_rate', 0.2)

    inputs = tf.keras.Input(shape=[*image_size, 3], name='image')
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
    image_size = [model_config['image_size'], model_config['image_size']]
    last_activation = model_config.get('activation', 'linear')
    backbone = tf.keras.applications.ResNet50(
        include_top=False,
        weights=model_config.get('weights', 'imagenet'),
        input_shape=(*image_size, 3),
        pooling=model_config.get('pooling', 'avg'),)

    backbone_drop_rate = model_config.get('backbone_drop_rate', 0.2)

    inputs = tf.keras.Input(shape=[*image_size, 3], name='image')

    hid = backbone(inputs)
    hid = tf.keras.layers.Dropout(backbone_drop_rate)(hid)
    #x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(1, activation=last_activation,
                                    use_bias=True)(hid)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='resnet50')
    return model