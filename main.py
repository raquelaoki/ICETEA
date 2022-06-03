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

"""Main file.
Run a range of items depending on the flags.

"""
from absl import app
from absl import flags
import cv2
from google.cloud import storage
import logging
import matplotlib.pyplot as plt
import os
import pandas as pd
import sys
import tensorflow as tf
import yaml
from tensorflow.io import gfile
from tensorflow.python.client import device_lib

# Local Imports.
import estimators
import data_kaggle as kd
import icetea_feature_extraction as fe
import icetea_data_simulation as ds
import helper_parameters as hp
import utils

logger = logging.getLogger(__name__)

FLAGS = flags.FLAGS
# General Flags.
flags.DEFINE_bool('use_beam', False, 'Use Beam Pipeline.')
flags.DEFINE_bool('load_yaml', False, 'Load yaml files instead of passing some parameters on the FLAGS.')
flags.DEFINE_string('path_root', 'data/', 'Dataset Root Path.')
flags.DEFINE_string('path_config_folder', 'config_yaml/', 'Path to Feature Extrator .yaml file.')
flags.DEFINE_string('path_tfrecords', 'icetea_tfr/', 'tfrecords Folder Name (under path_root).')
flags.DEFINE_string('path_features', 'icetea_features/', 'features Folder Name (under path_root).')
flags.DEFINE_bool('using_gc', False, 'Using google cloud computing.')
flags.DEFINE_string('bucket_name', 'icetea_kaggle_data', 'Google Cloud Bucket Name (if using GCP).')

# Feature Extractor FLAGS.
flags.DEFINE_bool('feature_extraction', False, 'Run Feature Extrator Components.')
flags.DEFINE_bool('feature_extraction_yaml', False, 'Load Feature Extrator .yaml file.')
flags.DEFINE_string('path_images_png', 'icetea_png/train/', 'PNG Images Folder Name (under path_root).')
flags.DEFINE_string('metafile', 'trainLabels.csv', 'metafile name.')
flags.DEFINE_string('feat_extract_optimizer', 'adam', 'Feature Extractor Optimizer')
flags.DEFINE_integer('feat_extract_batch_size', 16, 'Feature Extractor Batch Size.')
flags.DEFINE_integer('feat_extract_input_shape', 128, 'Feature Extractor Image Input shape.')
flags.DEFINE_integer('feat_extract_num_classes', 5, 'Feature Extractor num_classes.')
flags.DEFINE_integer('feat_extract_learning_rate', 0.0002, 'Feature Extractor learning_rate.')
flags.DEFINE_integer('feat_extract_steps_per_epoch', 5, 'Feature Extractor steps_per_epoch.')
flags.DEFINE_integer('feat_extract_hidden_size', 64, 'Feature Extractor hidden_size.')
flags.DEFINE_integer('feat_extract_epochs', 10, 'Feature Extractor epochs.')
flags.DEFINE_integer('feat_extract_backbone_drop_rate', 0.25, 'Feature Extractor backbone_drop_rate.')

# Data Simulation FLAGS.
flags.DEFINE_bool('data_simulation', False, 'Run Data Simulation Components.')
flags.DEFINE_string('path_tfrecords_new', 'icetea_newdata/', 'Folder under root to Save new tfrecords')
flags.DEFINE_integer('data_sim_b', 20, 'Number of dataset repetitions.')
flags.DEFINE_list('data_sim_knobs', [False, False, False], 'Data Sim Knobs: [overlap, heterogeneity, scale].')
flags.DEFINE_list('data_sim_beta_range', [], 'Data Sim Knobs: overlap range.')
flags.DEFINE_list('data_sim_alpha_range', [], 'Data Sim Knobs: scale range.')
flags.DEFINE_list('data_sim_gamma_range', [], 'Data Sim Knobs:  heterogeneity range.')
flags.DEFINE_bool('data_sim_allow_shift', False, 'Data Sim allow_shift: allows scale to 1 in some cases.')

# Causal Inference FLAGs.
flags.DEFINE_bool('causal_inference', False, 'Run Causal Inference.')
flags.DEFINE_list('ci_epochs', [10], 'Base-models DNN epochs.')
flags.DEFINE_list('ci_steps', [20], 'Base-models DNN steps.')
flags.DEFINE_list('ci_batch_size', [64], 'Base-models DNN batch_size.')
flags.DEFINE_list('ci_name_estimator', ['aipw'], 'Causal Inference Estimator.')
flags.DEFINE_list('ci_name_base_model', ['image_regression', 'resnet50', 'inceptionv3'], 'Base Models.')
flags.DEFINE_list('ci_learn_prop_score', [False], 'Should a Prop Score Model be trained?')
flags.DEFINE_list('ci_name_prop_score', ['LogisticRegression_NN'], 'Name Propensity Score (if training).')
flags.DEFINE_list('ci_metric', ['mse'], 'Base models metrics.')
flags.DEFINE_list('ci_model_repetitions', 1, 'Model repetitions.')

flags.DEFINE_string('output_name', 'result_', 'The csv file output_name.')
flags.DEFINE_string('path_results', 'icetea_results/', 'Folder to save Causal Inference results.')
flags.DEFINE_string('name_yaml_index', 'indexes.yaml', 'Causal Inference: index name file.')
flags.DEFINE_list('running_indexes', [0, 1, 2, 3], 'Indexes to run.')
flags.DEFINE_bool('use_tpu', False, 'Train Base-models with TPUs.')
flags.DEFINE_bool('adopt_multiworker', False, 'Adoptt Multiwork strategy.')
flags.DEFINE_string('name_data', 'noname', 'Name of the dataset.')
flags.DEFINE_integer('ci_image_size', 256, 'Image size.')
flags.DEFINE_integer('ci_batch_size', 64, 'Batch Size')
flags.DEFINE_string('ci_prefix_train', 'trainNew', 'TFRecords - Causal Inference Files (after join)')


def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    # IF RUNNING CODE ON GOOGLE CLOUD.
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"
    # The path to your file to upload
    # source_file_name = "local/path/to/file"
    # The ID of your GCS object
    # destination_blob_name = "storage-object-name"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name + source_file_name)
    blob.upload_from_filename(source_file_name)
    logger.info(
        f"File {source_file_name} uploaded to {destination_blob_name}."
    )


def update_experiments(filename, path_root, path_features, row, status):
    list_of_datasets = pd.read_csv(os.path.join(path_root, path_features, filename + '.csv'))
    assert list_of_datasets['running'][row] != 'Done', 'SIMULATION ' + row + ' already done!'
    list_of_datasets['running'][row] = status
    if 'Unnamed: 0' in list_of_datasets.columns:
        list_of_datasets.drop('Unnamed: 0', axis=1, inplace=True)
    list_of_datasets.to_csv(os.path.join(path_root, path_features, filename + '.csv'))


def main(_):
    # params_path, running_indexes_path, use_tpus_str
    if FLAGS.load_yaml:
        path_config = FLAGS.path_config_folder
        assert tf.io.gfile.isdir(path_config), path_config + ': Folder does not exist!'

    if FLAGS.feature_extraction:
        if FLAGS.load_yaml:
            with open(os.path.join(path_config, 'feature_extractor_setup.yaml')) as f:
                config_fe = yaml.safe_load(f)
            config_fe = config_fe['parameters']
        else:
            config_fe = {}
        config_fe['path_root'] = config_fe.get('path_root', FLAGS.path_root)
        config_fe['path_images_png'] = config_fe.get('path_images_png', FLAGS.path_images_png)
        config_fe['path_tfrecords'] = config_fe.get('path_tfrecords', FLAGS.path_tfrecords)
        config_fe['path_features'] = config_fe.get('path_features', FLAGS.path_features)
        config_fe['xfe_proportion_size'] = config_fe.get('xfe_proportion_size', FLAGS.xfe_proportion_size)
        config_fe['metafile'] = config_fe.get('metafile', FLAGS.metafile)
        config_fe['batch_size'] = config_fe.get('batch_size', FLAGS.feat_extract_batch_size)
        config_fe['input_shape'] = config_fe.get('input_shape', [FLAGS.feat_extract_input_shape,
                                                                 FLAGS.feat_extract_input_shape])
        config_fe['num_classes'] = config_fe.get('num_classes', FLAGS.feat_extract_num_classes)
        config_fe['backbone_drop_rate'] = config_fe.get('backbone_drop_rate', FLAGS.feat_extract_backbone_drop_rate)
        config_fe['optimizer'] = config_fe.get('optimizer', FLAGS.feat_extract_optimizer)
        config_fe['learning_rate'] = config_fe.get('learning_rate', FLAGS.feat_extract_learning_rate)
        config_fe['steps_per_epoch'] = config_fe.get('steps_per_epoch', FLAGS.feat_extract_steps_per_epoch)
        config_fe['epochs'] = config_fe.get('epochs', FLAGS.feat_extract_epochs)
        config_fe['hidden_size'] = config_fe.get('hidden_size', FLAGS.feat_extract_hidden_size)

        config_fe = hp.consistency_check_feature_extractor(config_fe)
        fe.feature_extrator_wrapper(config_fe)

    if FLAGS.data_simulation:
        if FLAGS.load_yaml:
            with open(os.path.join(path_config, 'data_simulation_setup.yaml')) as f:
                config_ds = yaml.safe_load(f)
            config_ds = config_ds['parameters']
        else:
            config_ds = {}
        config_ds['path_root'] = config_ds.get('path_root', FLAGS.path_root)
        config_ds['path_tfrecords'] = config_ds.get('path_tfrecords', FLAGS.path_tfrecords)
        config_ds['path_features'] = config_ds.get('path_features', FLAGS.path_features)
        config_ds['b'] = config_ds.get('b', FLAGS.data_sim_b)
        config_ds['knobs'] = config_ds.get('knobs', FLAGS.data_sim_knobs)
        config_ds['beta_range'] = config_ds.get('beta_range', FLAGS.data_sim_beta_range)
        config_ds['alpha_range'] = config_ds.get('alpha_range', FLAGS.data_sim_alpha_range)
        config_ds['gamma_range'] = config_ds.get('gamma_range', FLAGS.data_sim_gamma_range)
        config_ds['allow_shift'] = config_ds.get('allow_shift', FLAGS.data_sim_allow_shift)

        config_ds = hp.consistency_check_data_simulation(config_ds)
        ds.data_simulation_wrapper(config_ds)

    if FLAGS.causal_inference:
        if FLAGS.load_yaml:
            with open(os.path.join(path_config, 'causal_inference_setup.yaml')) as f:
                config_ci = yaml.safe_load(f)
            config_ci = config_ci['parameters']

            name_yaml_index = FLAGS.name_yaml_index
            with open(os.path.join(path_config, name_yaml_index)) as f:
                running_indexes = yaml.safe_load(f)
            running_indexes = running_indexes['parameters']['running_indexes']
        else:
            config_ci = {}
            running_indexes = FLAGS.running_indexes

        use_tpu = config_ci.get('use_tpu', FLAGS.use_tpu)
        adopt_multiworker = config_ci.get('adopt_multiworker', FLAGS.adopt_multiworker)
        param_data = {}
        param_data['name'] = config_ci.get('name_data', FLAGS.name_data)
        path_tfrecords = config_ci.get('path_tfrecords_new', FLAGS.path_tfrecords_new)
        path_root = config_ci.get('path_root', FLAGS.path_root)
        param_data['path_tfrecords'] = os.path.join(path_root, path_tfrecords)
        pixels = config_ci.get('image_size', [FLAGS.ci_image_size,  FLAGS.ci_image_size])
        param_data['image_size'] = pixels
        param_data['batch_size'] = config_ci.get('batch_size', FLAGS.ci_batch_size)
        param_data['prefix_train'] = config_ci.get('prefix_train', FLAGS.ci_prefix_train)

        param_method = {}
        param_method['name_estimator'] = config_ci.get('name_estimator', FLAGS.ci_name_estimator)
        param_method['name_metric'] = config_ci.get('name_metric', FLAGS.ci_metric)
        param_method['name_base_model'] = config_ci.get('name_base_model', FLAGS.ci_name_base_model)
        param_method['learn_prop_score'] = config_ci.get('learn_prop_score', FLAGS.ci_learn_prop_score)
        param_method['name_prop_score'] = config_ci.get('name_prop_score', FLAGS.ci_name_prop_score)
        param_method['epochs'] = config_ci.get('epochs', FLAGS.ci_epochs)
        param_method['steps'] = config_ci.get('steps', FLAGS.ci_steps)
        #param_method['image_size'] = pixels[0]
        model_repetitions = config_ci.get('repetitions', FLAGS.ci_model_repetitions)
        hp.consistency_check_causal_methods(param_method)

        config_ci['output_name'] = config_ci.get('output_name', FLAGS.output_name)
        config_ci['path_results'] = config_ci.get('path_results', FLAGS.path_results)
        using_gc = config_ci.get('using_gc', FLAGS.using_gc)
        config_ci['bucket_name'] = config_ci.get('bucket_name', FLAGS.bucket_name)

        list_of_datasets = pd.read_csv(os.path.join(path_root, path_features, 'true_tau.csv'))
        config_methods = hp.create_configs(param_method)

        hp.consistency_check_causal_methods(config_methods)

        if adopt_multiworker:
            if use_tpu:
                cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
                tf.config.experimental_connect_to_cluster(cluster_resolver)
                tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
                strategy = tf.distribute.experimental.TPUStrategy(cluster_resolver)
                logger.info("Running on TPU ", cluster_resolver.master())
                logger.info("REPLICAS: ", strategy.num_replicas_in_sync)
            else:
                raise NotImplementedError
        else:
            strategy = None

        if FLAGS.use_beam:
            running_indexes = list(range(list_of_datasets.shape[0]))
            # TODO
        else:
            for i, sim_id in enumerate(list_of_datasets['sim_id']):
                if i in running_indexes:
                    logger.debug('running ', i)
                    #  Loads dataset with appropried sim_id.
                    # data = utils.ImageData(seed=sim_id, param_data=param_data)
                    #  Creates a temporary DataFrame to keep the repetitions results under this dataset;
                    #  Meaning: data is loaded once, and we have several models (defined in parameters.config_methods)
                    #  using this dataset.
                    results_one_dataset = pd.DataFrame()
                    for _config in config_methods:
                        _config['image_size'] = pixels[0]
                        results_one_config = utils.repeat_experiment(param_data=param_data,
                                                                     param_method=_config,
                                                                     seed_data=sim_id,
                                                                     seed_method=i,
                                                                     model_repetitions=model_repetitions,
                                                                     use_tpu=use_tpu,
                                                                     strategy=strategy)
                        results_one_config['sim_id'] = sim_id
                        results_one_config = pd.merge(results_one_config, list_of_datasets, how='left')
                        results_one_dataset = pd.concat([results_one_dataset, results_one_config])
                        # Intermediate save.
                        utils.save_results(using_gc=using_gc, params=config_ci, results=results_one_dataset)

                    utils.save_results(using_gc=using_gc, params=config_ci, results=results_one_dataset)
                    update_experiments('true_tau_sorted', path_root, path_features, i, status='done')
    logger.info('DONE')
    return


if __name__ == '__main__':
    app.run(main)
