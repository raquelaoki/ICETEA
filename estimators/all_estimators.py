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

# """all_estimators!
#
# It organizes the causal inference estimators.
#
# """

import itertools
import logging
import numpy as np
import numpy as np
from sklearn import linear_model
from sklearn import metrics
from sklearn import model_selection
import tensorflow as tf

logger = logging.getLogger(__name__)

ATE_ESTIMATE = 'treatment_effect_hat'
MSE_CONTROL = 'mse_control'
MSE_TREATED = 'mse_treated'
BIAS_CONTROL = 'bias_control'
BIAS_TREATED = 'bias_treated'
VARIANCE = 'variance'


def estimator_aipw(data, param_method, seed=0, add_kob=True):
    """Estimate treatment effect using AIPW.

    :param data: DataSimulation Class.
    :param param_method: dict with method's parameters.
    :param seed: model seed.

    :return:
    t_estimated: estimated treatment effect using the oaxaca-blinder method,
    metric: list [,], metric on control and treated groups,
    bias: list [,], bias on control and treated group,
    var: float, estimator variance.
    """
    # Fix numpy seed for reproducibility
    np.random.seed(seed)
    logger.debug('Running AIPW.')
    # Fitting two models, one per treatment value.
    logger.debug('Base Model - Control.')
    y_control_hat, mse_control, bias_control, t_obs0, y_obs0 = _fit_base_models(model=param_method['base_model'],
                                                                                data=data.dataset_control,
                                                                                data_all=data.dataset_all,
                                                                                param_method=param_method)
    logger.debug('Base Model - Treated')
    y_treated_hat, mse_treated, bias_treated, t_obs1, y_obs1 = _fit_base_models(model=param_method['base_model'],
                                                                                data=data.dataset_treated,
                                                                                data_all=data.dataset_all,
                                                                                param_method=param_method)

    assert sum(y_obs1) == sum(y_obs0), 'These two values should be identical!'

    logger.debug('Cauculate Propensity Score.')
    y_dif, residual_dif = _calculate_propensity_score(param_method=param_method,
                                                      data=data,
                                                      y_treated_hat=y_treated_hat,
                                                      y_control_hat=y_control_hat,
                                                      t=t_obs1,
                                                      y=y_obs1
                                                      )
    treatment_effect_hat = np.mean(y_dif + residual_dif)
    # Variance.
    variance = (y_dif + residual_dif - treatment_effect_hat) ** 2
    variance = variance.sum() / (len(y_dif) - 1)
    variance = variance / len(y_dif)

    output = {'aipw': {ATE_ESTIMATE: treatment_effect_hat,
                       MSE_CONTROL: mse_control,
                       MSE_TREATED: mse_treated,
                       BIAS_CONTROL: bias_control,
                       BIAS_TREATED: bias_treated,
                       VARIANCE: variance
                       }
              }

    if add_kob:
        # Fill-in unobserved // counterfactual.
        y0_c = y_obs0
        y1_c = y_obs1

        y0_c[t_obs0] = y_control_hat[t_obs0]
        y1_c[~t_obs0] = y_treated_hat[~t_obs0]

        # Variance.
        n0 = len(t_obs0) - t_obs0.sum()
        n1 = t_obs0.sum()
        # Variance and treatment effect estimation
        variance_kob = mse_control / n0 + mse_treated / n1
        treatment_effect_hat_kob = (y1_c - y0_c).mean()

        output['kob'] = {
            ATE_ESTIMATE: treatment_effect_hat_kob,
            MSE_CONTROL: mse_control,
            MSE_TREATED: mse_treated,
            BIAS_CONTROL: bias_control,
            BIAS_TREATED: bias_treated,
            VARIANCE: variance_kob
        }

    return output


def estimator_template(data, param_method, seed=0):
    """Template to add new estimators.

    This function normally returns four values. Some estimators do not return these values.
    In this case, you can either calculate manually (adding lines of code) or return None.
    The only exception is to the treatment_effect_hat (IT SHOULD ALWAYS BE RETURNED).

    IMPORTANT: the prefix 'estimator_' should be kept in order to avoid NotImplementedError errors.

    :param data:
    :param param_method:
    :param seed:
    :return:
    """

    np.random.seed(seed)
    logger.debug('Running YOUR METHOD')

    # 1) Call your estimator.
    # 2) Train/Estimate treatment_effect_hat using data and param_method.

    output = {'new_method': {ATE_ESTIMATE: 999,
                             MSE_CONTROL: None,
                             MSE_TREATED: None,
                             BIAS_CONTROL: None,
                             BIAS_TREATED: None,
                             VARIANCE: None
                             }
              }

    return output


def _fit_base_models(model, data, data_all, param_method):
    """Predicts the outcome on the full data.

    :param data: tf.data.Dataset (control or treated).
    :param model: tf.keras.Model.
    :param dataset_all: tf.data.Dataset (all, for prediction).

    :return: y_pred: [], predictions on data_all.
     mse: array with mse on the control and treated group.
     bias: array with bias on the control and treated group.
     t: [], treatment assigment on data_all.
     y_obs: [], observed outcome on data_all.
    """

    history = model.fit(data, steps_per_epoch=param_method['steps'], epochs=param_method['epochs'], verbose=2)
    try:
        mse = history.history['mean_squared_error'][-1]
    except KeyError:
        mse = history.history['mse'][-1]

    logger.debug('Calculate Predictions')
    y_pred, y_obs, t = _predict_base_models(data=data_all, model=model, steps=param_method['steps_predictions'])
    logger.debug('Done Predictions')
    bias = mean(y_pred) - mean(y_obs)
    return y_pred, mse, bias, t, y_obs


def _predict_base_models(data, model, steps=200):
    """Predicts the outcome on the full data.
    The steps are required as data was created with dataset.repeat().
    Hence, if steps is not given, it enters an infinite loop.

    :param data: tf.data.Dataset.
    :param model: tf.keras.Model.
    :return: arrays with predicted outcome, observed outcome, and treat. assignment.
    """
    y_pred = []
    y_obs = []
    t = []
    for i, (batch_x, batch_y, batch_t) in enumerate(data):
        y_pred.append(model.predict_on_batch(batch_x))
        y_obs.append(batch_y.numpy())
        t.append(batch_t.numpy())
        if i > steps:
            break

    y_pred_flat = np.concatenate(y_pred).ravel()
    y_obs_flat = np.concatenate(y_obs).ravel()
    t_flat = np.concatenate(t).ravel()

    return np.array(y_pred_flat), np.array(y_obs_flat), np.array(t_flat)


def _calculate_propensity_score(param_method, data, y_treated_hat, y_control_hat, t, y):
    """ Calculating Propensity Score and return the treatment effect.

    :param param_method: dictionary.
    :param data: tf.data.Dataset.
    :param y_treated_hat: [], predicted outcome if treated.
    :param y_control_hat: [], predicted outcome if control.
    :param t: [], treatment assignment.
    :param y: [], observed outcome.
    :return: tau (int, treat effect), var (int, variance)
    """
    t = t.ravel()
    y = y.ravel()

    y_dif = (y_treated_hat - y_control_hat)
    residual_treated = (t * (y - y_treated_hat))
    residual_control = ((1 - t) * (y - y_control_hat))

    # Calculating the Propensity Score.
    if param_method['learn_prop_score']:
        data_all = data.dataset_all_ps
        # Propensity score using a logistic regression.
        prop_score = param_method['prop_score']
        prop_score.fit(data_all)
        e = prop_score.predict_proba(data_all)
        # Removing extreme values
        print('before ', e[0:5, :])
        e = _truncate_by_g(propensity_score=e, level=param_method['level'])
        print('after ', e[0:5, :])

        residual_treated = (residual_treated / e[:, 1].ravel())
        residual_control = (residual_control / e[:, 0].ravel())

    else:
        e = sum(t)/len(t)
        residual_treated = (residual_treated / e)
        residual_control = (residual_control / e)

    residual_dif = (residual_treated - residual_control)
    return y_dif, residual_dif


def _truncate_by_g(propensity_score, level=0.005):
    """ Add min value for propensity score for stability.

    :param propensity_score: matrix with values to be truncated.
    :param level: float, limit.
    :return: matrix with truncated values.
    """
    propensity_score[:, 0] = [np.minimum(np.maximum(item, level), 1-level) for item in propensity_score[:, 0]]
    propensity_score[:, 1] = 1 - propensity_score[:,0]
    return e