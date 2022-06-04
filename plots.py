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

import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns

# Local imports.
import helper_data as hd
import utils

sns.set(font_scale=1.5)
sns.set_style("whitegrid")


def exploring_simulated_tau(config_sim):
    """ Create plot to explore simulated data.

    :param config_sim: dictionary
    :return: plot
    """
    tau = pd.read_csv(config_sim['path_features'] + 'true_tau.csv')

    knob_ks = tau[tau['knob'] == 'ks']
    knob_kh = tau[tau['knob'] == 'kh']
    knob_ko = tau[tau['knob'] == 'ko']

    fig, ((ax0, ax1, ax2)) = plt.subplots(ncols=3, nrows=1, figsize=[20, 4])

    sns.set(font_scale=1.5)

    knob_ks.alpha = knob_ks.alpha.astype('str')
    ax0 = sns.lineplot(x='alpha', y='tau', data=knob_ks, ax=ax0)
    ax0.set_ylabel('τ - True Treatment Effect')
    ax0.set_xlabel('α - Treat. Effect Scale Knob')

    ax1 = sns.lineplot(x='gamma', y='tau', data=knob_kh, ax=ax1)
    ax1.set_ylabel('')
    ax1.set_xlabel('γ - Heterogeneity Knob')

    ax2 = sns.lineplot(x='beta', y='tau', data=knob_ko, ax=ax2)
    ax2.set_ylabel('')
    ax2.set_xlabel('β - Overlap Knob')


def checking_tfrecords(config):
    """ Checking if files exist.

    :param config: dictionary.
    :return: plot.
    """
    dataset_train, dataset_val = hd.build_datasets_feature_extractor(dataset_config=config,
                                                                     prefix_train='train',
                                                                     prefix_extract='extractor',
                                                                     type='TrainFeatureExtractor'
                                                                     )
    batch = next(iter(dataset_train))
    plt.imshow(batch[0][0])


def checking_tfrecords_after_join(config, seed=None):
    """ Checking tfrecords images.

    :param config: dictionary.
    :param seed: dataset seed.
    :return: plot.
    """
    if not seed:
        simulations = pd.read_csv(config['path_features'] + 'joined_simulations.csv')
        seeds = simulations.columns[1:]
        seeds = [item.split('-')[0] for item in seeds]
        seed = pd.unique(seeds)[0]

    param_data = {
        'name': 'kagle_retinal',
        'path_tfrecords': os.path.join(config['path_root'], config['path_tfrecords_new']),
        'image_size': [256, 256],
        'batch_size': 2,
        'prefix_train': 'trainNew',
    }
    data = utils.ImageData(seed, param_data)
    data.make_plot()


