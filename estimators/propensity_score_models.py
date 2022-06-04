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

# """propensity_score_models!
#
# Implementation of the propensity score models.
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


class PS_LogisticRegression_NN:
    """Make NN version of Logistic Regression.
    """

    def __init__(self, image_size):
        super(PS_LogisticRegression_NN, self).__init__()
        self.model = self._logistic_regression_architecture(image_size)

    def fit(self, data):
        """Fits a Classification Model.

        :param data: prefetch batches 16 [B, H, W, C], not repeated, not shuffled.
        """
        self.model.compile(
            optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        epochs = 10

        self.model.summary()
        self.model.fit(data, epochs=epochs, verbose=2)

    def predict_proba(self, data):
        """Predict Probability of each class.

        :param data: tf.data.Dataset
        :return: predict: predictions array
        """
        t_pred = []
        t = []
        for i, (batch_x, batch_t) in enumerate(data):
            t_pred.append(self.model.predict_on_batch(batch_x))
            t.append(batch_t.numpy())

        t_pred = np.concatenate(t_pred).ravel().reshape(-1, 2)
        return t_pred

    def _logistic_regression_architecture(self, image_size):
        """Implements of Propensity Score.

        It takes as input tensors of shape [B, H, W, C] and outputs [B,Y]
        :param image_size: int
        :return: model: tf.keras.Mode
        """
        # A simple logistic regression implemented as NN.
        image_size = [image_size, image_size]
        inputs = tf.keras.Input(shape=[*image_size, 3], name='LogisticRegression')

        backbone = tf.keras.applications.ResNet50(
            include_top=False,
            weights='imagenet',
            input_shape=(*image_size, 3),
            pooling='avg', )
        backbone_drop_rate = 0.2

        inputs = tf.keras.Input(shape=[*image_size, 3], name='image')
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
