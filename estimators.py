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

"""Implementation of treatment effect estimators.

Estimator Functions
tau, mse, bias = estimator_oaxaca()
tau, mse, bias = estimator_aipw()

source https://keras.io/examples/keras_recipes/tfrecord/
"""

import numpy as np
from sklearn import linear_model
from sklearn import metrics
from sklearn import model_selection
import tensorflow as tf


def default_param_method():
    param_method = {}
    param_method['base_model'] = linear_model.LinearRegression()
    param_method['metric'] = metrics.mean_squared_error
    param_method['prop_score'] = linear_model.LogisticRegression()
    param_method['image'] = False
    return param_method


def estimator(data, param_method=None,  type='oahaca', seed=0):
    """Estimate treatment effect.

  Args:
    data: DataSimulation Class
    param_method: dict with method's parameters
    type: oahaca, aipw
  Returns:
    t_estimated: estimated treatment effect using the oaxaca-blinder method
    metric: list, metric on control and treated groups
    bias: list, bias on control and treated group
    var: float, estimator variance.

  """
    # Fix numpy seed for reproducibility
    np.random.seed(seed)

    # Defining basic parameters.
    if param_method is None:
        param_method = default_param_method()

    print('param_method', param_method)
    # Fitting two models, one per treatment value.
    pred_control, mse_control, e_control, treated, y0_c = fit_base_image_models(
        data.dataset_control,
        param_method['base_model'],
        data.dataset_all,
        param_method)
    pred_treated, mse_treated, e_treated, treated, y1_c = fit_base_image_models(
        data.dataset_treated,
        param_method['base_model'],
        data.dataset_all,
        param_method)

    if type == 'oahaca':
        # Fill-in unobserved // counterfactual.
        y0_c[treated] = pred_control[treated]
        y1_c[~treated] = pred_treated[~treated]

        # Variance.
        n0 = len(treated) - treated.sum()
        n1 = treated.sum()
        # Variance and treatment effect estimation
        var = mse_control / n0 + mse_treated / n1
        tau_estimated = (y1_c - y0_c).mean()

    else:
        x = data.dataset_all_ps
        tau_estimated, var = _calculate_propensity_score(param_method, x, data, pred_treated, pred_control,
                                                         t=treated, y0=y0_c, y1=y1_c)

    bias = [e_control, e_treated]

    # Metrics.
    metric = [mse_control, mse_treated]

    return tau_estimated, metric, bias, var


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


def _calculate_propensity_score(param_method, x, data, pred_treated, pred_control, t, y0, y1):

    try:
        z = data.treatment.ravel()
        y = data.outcome.ravel()
    except AttributeError:
        z = t * 1
        y = y0.ravel()


    if param_method['learn_prop_score']:
        # Propensity score using a logistic regression.
        prop_score = param_method['prop_score']
        prop_score.fit(x)
        e = prop_score.predict_proba(x)
        # Removing extreme values
        # print('y shaoes', y.shape, type(y))
        g = e[:, 1].ravel()
        y = _truncate_by_g(y.reshape(-1, 1), g, 0.005)
        z = _truncate_by_g(z, g, 0.005)
        pred_treated = _truncate_by_g(pred_treated, g, 0.005)
        pred_control = _truncate_by_g(pred_control, g, 0.005)
        e = _truncate_by_g(e, g, 0.005)

        # Estimating the treatment effect.
        pred_dif = (pred_treated - pred_control)
        sample_size = len(y)
        residual_treated = (z * (y - pred_treated))
        residual_control = ((1 - z) * (y - pred_control))

        residual_treated = (residual_treated / e[:, 1].ravel())
        residual_control = (residual_control / e[:, 0].ravel())

    else:
        # Estimating the treatment effect.
        pred_dif = (pred_treated - pred_control)
        sample_size = len(y)
        residual_treated = (z * (y - pred_treated))
        residual_control = ((1 - z) * (y - pred_control))
        residual_treated = (residual_treated / 0.5)
        residual_control = (residual_control / 0.5)

    residual_dif = (residual_treated - residual_control)
    tau_estimated = np.mean(np.add(pred_dif, residual_dif))

    # Variance.
    var = np.add(pred_dif, residual_dif)
    var = np.subtract(var, tau_estimated)
    var = var ** 2
    var = var.sum() / (sample_size - 1)
    var = var / sample_size

    return tau_estimated, var


def _prediction_image_models(data, model, step_lim=50):  # quick=False
    """Predicts the outcome on the full data.

  Args:
    data: tf.data.Dataset.
    model: fitted model.
    #quick: predict in a subset of data.
  Returns:
    arrays with predicted outcome, observed outcome, and treat. assignment.
  """
    y_pred = []
    y_sim = []
    t = []
    for i, (batch_x, batch_y, batch_t) in enumerate(data):
        y_pred.append(model.predict_on_batch(batch_x))
        y_sim.append(batch_y.numpy())
        t.append(batch_t.numpy())
        if step_lim>i:
            break

    y_pred_flat = np.concatenate(y_pred).ravel()
    y_sim_flat = np.concatenate(y_sim).ravel()
    t_flat = np.concatenate(t).ravel()

    return np.array(y_pred_flat), np.array(y_sim_flat), np.array(t_flat)


def fit_base_image_models(data, model, dataset_all, model_settings):
    """Predicts the outcome on the full data.

  Args:
    data: tf.data.Dataset (control or treated).
    model: fitted model.
    dataset_all: tf.data.Dataset (all, for prediction).
    quick: predict in a subset of data_all.
  Returns:
    y_pred: predictions on data_all
    mse: array with mse on the control and treated group
    bias: array with bias on the control and treated group
    t: treatment assigment on data_all
    y_sim: observed outcome on data_all
  """
    epochs = model_settings.get('epochs', 10)
    steps = model_settings.get('steps', 10)

    history = model.fit(data, steps_per_epoch=steps, epochs=epochs, verbose=2)
    try:
        mse = history.history['mean_squared_error'][-1]
    except KeyError:
        mse = history.history['mse'][-1]

    bias = history.history['mae'][-1]
    y_pred, y_sim, t = _prediction_image_models(dataset_all, model)  # quick=quick

    return y_pred, mse, bias, t, y_sim
