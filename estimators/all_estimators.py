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
import logging
import numpy as np
import numpy as np
from sklearn import linear_model
from sklearn import metrics
from sklearn import model_selection
import tensorflow as tf

logger = logging.getLogger(__name__)


def estimator_aipw(data, param_method, seed=0):
    """Estimate treatment effect using AIPW.

      Args:
        data: DataSimulation Class
        param_method: dict with method's parameters
        seed: model seed
      Returns:
        t_estimated: estimated treatment effect using the oaxaca-blinder method
        metric: list, metric on control and treated groups
        bias: list, bias on control and treated group
        var: float, estimator variance.

      """
    # Fix numpy seed for reproducibility
    np.random.seed(seed)
    logger.debug('Running AIPW.')
    # Fitting two models, one per treatment value.
    logger.debug('Base Model - Control.')
    y_control_hat, mse_control, e_control, t, y_obs0 = _fit_base_models(model=param_method['base_model'],
                                                                        data=data.dataset_control,
                                                                        data_all=data.dataset_all,
                                                                        param_method=param_method)
    logger.debug('Base Model - Treated')
    y_treated_hat, mse_treated, e_treated, t_, y_obs1 = _fit_base_models(model=param_method['base_model'],
                                                                         data=data.dataset_treated,
                                                                         data_all=data.dataset_all,
                                                                         param_method=param_method)
    assert sum(t) == sum(t_), 'These two values should be identical!'
    assert sum(y_obs1) == sum(y_obs0), 'These two values should be identical!'

    logger.debug('Cauculate Propensity Score.')
    tau_estimated, var = _calculate_propensity_score(param_method=param_method,
                                                     data=data,
                                                     y_treated_hat=y_treated_hat,
                                                     y_control_hat=y_control_hat,
                                                     t=t,
                                                     y=y_obs1
                                                     )

    bias = [e_control, e_treated]

    # Metrics.
    metric = [mse_control, mse_treated]

    return tau_estimated, metric, bias, var


def estimator_kob(data, param_method, seed=0):
    """Estimate treatment effect using KOB.

      Args:
        data: DataSimulation Class
        param_method: dict with method's parameters
        seed: model seed
      Returns:
        t_estimated: estimated treatment effect using the oaxaca-blinder method
        metric: list, metric on control and treated groups
        bias: list, bias on control and treated group
        var: float, estimator variance.

      """
    # Fix numpy seed for reproducibility
    np.random.seed(seed)
    logger.debug('Running KOB')
    # Fitting two models, one per treatment value.
    logger.debug('Base Model - Control.')
    pred_control, mse_control, e_control, t, y_obs0 = _fit_base_models(model=param_method['base_model'],
                                                                       data=data.dataset_control,
                                                                       data_all=data.dataset_all,
                                                                       param_method=param_method)
    logger.debug('Base Model - Treated.')
    pred_treated, mse_treated, e_treated, t_, y_obs1 = _fit_base_models(model=param_method['base_model'],
                                                                        data=data.dataset_treated,
                                                                        data_all=data.dataset_all,
                                                                        param_method=param_method)
    assert sum(t) == sum(t_), 'These two values should be identical!'
    assert sum(y_obs1) == sum(y_obs0), 'These two values should be identical!'
    # Fill-in unobserved // counterfactual.
    y0_c[t] = pred_control[t]
    y1_c[~t] = pred_treated[~t]

    # Variance.
    n0 = len(t) - t.sum()
    n1 = t.sum()
    # Variance and treatment effect estimation
    var = mse_control / n0 + mse_treated / n1
    tau_estimated = (y1_c - y0_c).mean()

    bias = [e_control, e_treated]

    # Metrics.
    metric = [mse_control, mse_treated]

    return tau_estimated, metric, bias, var


def estimator_template(data, param_method, seed=0):
    """Template to add new estimators.

    This function normally returns four values. Some estimators do not return these values.
    In this case, you can either calculate manually (adding lines of code) or return None.
    The only exception is to the tau_estimated (IT SHOULD ALWAYS BE RETURNED).

    IMPORTANT: the prefix 'estimator_' should be kept in order to avoid NotImplementedError errors.

    :param data:
    :param param_method:
    :param seed:
    :return:
    """

    np.random.seed(seed)
    logger.debug('Running YOUR METHOD')

    # 1) Call your estimator.
    # 2) Train/Estimate tau_estimated using data and param_method.

    tau_estimated = 999
    metric = [None, None]
    bias = [None, None]
    var = None

    return tau_estimated, metric, bias, var


def _fit_base_models(model, data, data_all, param_method):
    """Predicts the outcome on the full data.

  Args:
    data: tf.data.Dataset (control or treated).
    model: fitted model.
    dataset_all: tf.data.Dataset (all, for prediction).
  Returns:
    y_pred: predictions on data_all
    mse: array with mse on the control and treated group
    bias: array with bias on the control and treated group
    t: treatment assigment on data_all
    y_obs: observed outcome on data_all
  """

    history = model.fit(data, steps_per_epoch=param_method['steps'], epochs=param_method['epochs'], verbose=2)
    try:
        mse = history.history['mean_squared_error'][-1]
    except KeyError:
        mse = history.history['mse'][-1]

    bias = history.history['mae'][-1]
    logger.debug('Calculate Predictions')
    y_pred, y_obs, t = _predict_base_models(data=data_all, model=model, steps=param_method['steps'])
    logger.debug('Done Predictions')
    return y_pred, mse, bias, t, y_obs


def _predict_base_models(data, model, steps):
    """Predicts the outcome on the full data.

  Args:
    data: tf.data.Dataset.
    model: fitted model.
    #quick: predict in a subset of data.
  Returns:
    arrays with predicted outcome, observed outcome, and treat. assignment.
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
    t = t.ravel()
    y = y.ravel()

    if param_method['learn_prop_score']:
        data_all = data.dataset_all_ps
        # Propensity score using a logistic regression.
        prop_score = param_method['prop_score']
        prop_score.fit(data_all)
        e = prop_score.predict_proba(data_all)
        # Removing extreme values
        # print('y shaoes', y.shape, type(y))
        g = e[:, 1].ravel()
        y = _truncate_by_g(y.reshape(-1, 1), g, 0.005)
        t = _truncate_by_g(t, g, 0.005)
        y_treated_hat = _truncate_by_g(y_treated_hat, g, 0.005)
        y_control_hat = _truncate_by_g(y_control_hat, g, 0.005)
        e = _truncate_by_g(e, g, 0.005)

        # Estimating the treatment effect.
        y_dif = (y_treated_hat - y_control_hat)
        sample_size = len(y)
        residual_treated = (t * (y - y_treated_hat))
        residual_control = ((1 - t) * (y - y_control_hat))

        residual_treated = (residual_treated / e[:, 1].ravel())
        residual_control = (residual_control / e[:, 0].ravel())

    else:
        # Estimating the treatment effect.
        y_dif = (y_treated_hat - y_control_hat)
        sample_size = len(y)
        residual_treated = (t * (y - y_treated_hat))
        residual_control = ((1 - t) * (y - y_control_hat))
        residual_treated = (residual_treated / 0.5)
        residual_control = (residual_control / 0.5)

    residual_dif = (residual_treated - residual_control)
    tau_estimated = np.mean(np.add(y_dif, residual_dif))

    # Variance.
    var = np.add(y_dif, residual_dif)
    var = np.subtract(var, tau_estimated)
    var = var ** 2
    var = var.sum() / (sample_size - 1)
    var = var / sample_size

    return tau_estimated, var


def _truncate_by_g(attribute, g, level=0.005):
    """
    Remove rows with too low or too high g values. attribute and g must have same dimensions.
    :param attribute: column we want to keep after filted
    :param g: filter
    :param level: limites
    :return: filted attribute column
    """
    assert len(attribute) == len(g), 'Dimensions must be the same!' + str(len(attribute)) + ' and ' + str(len(g))
    keep_these = np.logical_and(g >= level, g <= 1. - level)
    return attribute[keep_these]
