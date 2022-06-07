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

""" ICETEA Data simulation.

It simulates the treatments and outcomes.
"""
import itertools
import logging
import math
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.io import gfile

logger = logging.getLogger(__name__)


def data_simulation_wrapper(config):
    """ It wrape up together all the data simulation pipeline.
    1) Creating simulations: it reads the features.csv file, and save joined_simulations.csv
    2) Creates new tfrecord files with the new simulated data.

    :param config: dictonary.
    :return: None.
    """
    # 1) Creating simulations: it reads the features.csv file.
    config['path_features'] = os.path.join(config['path_root'], config['path_features'])
    features_file = pd.read_csv(os.path.join(config['path_features'], 'features.csv'))
    logger.info('Features - ' + str(features_file.shape[0]) + ' rows and ' + str(features_file.shape[1]) + ' columns.')

    simulations_files = []
    # Generate the simulations - there are three different settings, one for each knob.
    if sum(config['knobs']) == 0:
        logger.debug('All knobs are false. Using only setting.')
        # All knobs are false -> Fixed knobs (no range explored).
        beta_range = 0.5 if len(config['beta_range']) == 0 else config['beta_range'][0]
        alpha_range = 1 if len(config['alpha_range']) == 0 else config['alpha_range'][0]
        gamma_range = 0.5 if len(config['gamma_range']) == 0 else config['gamma_range'][0]

        sf = generate_simulations(path_root=config['path_features'],
                                  output_name='simulations',
                                  features=features_file,  # pd.DataFrame(),
                                  b=config['b'],
                                  knob_o=False,  # overlap.
                                  knob_h=False,  # heterogeneity.
                                  knob_s=False,  # scale tau.
                                  beta_range=beta_range,
                                  alpha_range=alpha_range,
                                  gamma_range=gamma_range,
                                  allow_shift=config['allow_shift']
                                  )
        simulations_files.append(sf)
    else:
        logger.debug('Running with knobs.')
        if config['knobs'][0]:
            sf = generate_simulations(path_root=config['path_features'],
                                      output_name='simulations',
                                      features=features_file,  # pd.DataFrame(),
                                      b=config['b'],
                                      knob_o=True,  # overlap
                                      knob_h=False,  # heterogeneity
                                      knob_s=False,
                                      allow_shift=config['allow_shift']
                                      )
            simulations_files.append(sf)
        if config['knobs'][1]:
            sf = generate_simulations(path_root=config['path_features'],
                                      output_name='simulations',
                                      features=features_file,  # pd.DataFrame(),
                                      b=config['b'],
                                      knob_o=False,  # overlap
                                      knob_h=True,  # heterogeneity
                                      knob_s=False,
                                      allow_shift=config['allow_shift']
                                      )
            simulations_files.append(sf)

        if config['knobs'][2]:
            sf = generate_simulations(path_root=config['path_features'],
                                      output_name='simulations',
                                      features=features_file,  # pd.DataFrame(),
                                      b=config['b'],
                                      knob_o=False,  # overlap
                                      knob_h=False,  # heterogeneity
                                      knob_s=True,
                                      allow_shift=config['allow_shift']
                                      )
            simulations_files.append(sf)

    logger.debug('Organizing simulations.')
    organizing_simulations(path_features=config['path_features'], simulations_files=simulations_files)
    # 2) Join simulations and tfrecords
    path_simulations = os.path.join(config['path_root'], config['path_features'], 'joined_simulations.csv')
    join_tfrecord_csv(path_simulations=path_simulations,
                      path_input=os.path.join(config['path_root'], config['path_tfrecords']),
                      path_output=os.path.join(config['path_root'], config['path_tfrecords_new']),
                      input_prefix='extract',
                      output_prefix='trainNew')


def generate_simulations(path_root,
                         output_name='simulations',
                         features=pd.DataFrame(),
                         features_name='features.csv',
                         b=30,
                         knob_o=True,  # overlap
                         knob_h=False,  # heterogeneity
                         knob_s=False,  # causal effect scale
                         beta_range=[],
                         alpha_range=[],
                         gamma_range=[],
                         allow_shift=False
                         ):
    """Generate the p(x) and mu(x,y) from h(x).

    It saves the joined_simulations.csv files.

    :param path_root: folder path root.
    :param output_name: str, prefix for outputs results.
    :param features: pd.DataFrame().
    :param features_name: str, features.csv.
    :param b: int, repetitions.
    :param knob_o: bool, True explores overlap knob.
    :param knob_h: bool, True explores h knob.
    :param knob_s: bool, True explores s knob.
    :param beta_range: list with values to be explored if knob_o True (defoult values also provided).
    :param alpha_range: list with values to be explored if knob_h True (defoult values also provided).
    :param gamma_range: list with values to be explored if knob_s True (defoult values also provided).
    :param allow_shift: bool, allows shift so tau = 1.
    :return: None.
    """

    def _generation_weights(n_cols=2048, alpha=10):
        """Generate initial weights.
        :param n_cols: dimension of weights.
        :param alpha: treatment effect knob.
        :return: matrices with weights.
        """
        phi = np.random.uniform(-1, 1, n_cols)
        scale = np.random.uniform(0, alpha, 1)
        eta_1 = np.random.uniform(-1, 1 * scale, n_cols)
        eta_0 = np.random.uniform(-1, 1, n_cols)
        return phi, eta_1, eta_0

    def _t_function(features, gam, beta=0.5):
        """ Calcualting the treatment assigment.
        :param features: pd.DataFrame().
        :param gam: matrix.
        :param beta: int, overlap knob.
        :return: list [1,...,0], treatment assigment.
        """

        def sigmoid(x):
            return 1 / (1 + math.exp(-x))

        assert 0 <= beta <= 1, 'Beta out of range [0,1]'
        # Component extracted from features.
        Z = np.matmul(features, gam).reshape(-1, 1)
        scaler = MinMaxScaler((-2, 2))
        Z = scaler.fit_transform(Z).ravel()
        Z = [sigmoid(item) for item in Z]
        # Component from knob.
        ones = [1 if item > 0.5 else 0 for item in Z]
        pt = np.multiply(beta, Z)
        ones = np.multiply(1 - beta, ones)
        pt = pt + ones
        # Treatment assignment.
        T = [np.random.binomial(1, item) for item in pt]
        return T

    def _y_function(features, eta_1, eta_0, gamma=0.5, shift=False, alpha=1):
        """ Calculating the outcome.
        :param features: pd.DataFrame().
        :param eta_0: matrix, weights if untreated.
        :param eta_1: matrix, weights if treated.
        :param gamma: int, knob_s value.
        :param shift: bool, it sets tau to be 1.
        :return:  y_1, y_0: outcome if treated, outcome if control.
        """
        # Component from features.
        y_1 = np.array(np.matmul(features, eta_1))
        y_0 = np.array(np.matmul(features, eta_0))
        mu = np.array(np.concatenate([y_1, y_0]))
        scaler = MinMaxScaler((0, alpha))
        scaler.fit(mu.reshape(-1, 1))
        y_1 = scaler.transform(y_1.reshape(-1, 1))
        y_0 = scaler.transform(y_0.reshape(-1, 1))
        y_1 = np.multiply(gamma, y_1)
        y_0 = np.multiply(gamma, y_0)

        # Component from knobs.
        ones = np.ones(features.shape)
        y_1_ones = scaler.transform(np.array(np.matmul(ones, eta_1)).reshape(-1, 1))
        y_0_ones = scaler.transform(np.array(np.matmul(ones, eta_0)).reshape(-1, 1))
        y_1_ones = np.multiply(1 - gamma, y_1_ones)
        y_0_ones = np.multiply(1 - gamma, y_0_ones)
        y_1, y_0 = y_1 + y_1_ones, y_0 + y_0_ones

        if shift:
            # adding a constant so tau is always 1
            dif = 1 - (y_1.mean() - y_0.mean())
            y_1 = y_1 + dif

        return y_1, y_0

    # Organizing knobs.
    shift = False
    if knob_o:
        logger.debug('Knob - Overlap.')
        # Make a range of values for beta
        alpha = [10]
        beta = beta_range if len(beta_range) > 0 else [0, 0.5, 1]
        gamma = [0.5]
        if allow_shift:
            shift = True
        name_id = '_ko'
    elif knob_h:
        logger.debug('Knob - heterogeneity.')
        # Make a range of values for gamma
        alpha = [10]
        beta = [0.5]
        gamma = gamma_range if len(gamma_range) > 0 else [0, 0.5, 1]
        if allow_shift:
            shift = True
        name_id = '_kh'
    elif knob_s:
        logger.debug('Knob - Treat Effect Scale.')
        alpha = alpha_range if len(alpha_range) > 0 else [0.1, 1, 10]
        beta = [0.5]
        gamma = [0.5]
        name_id = '_ks'
    else:
        logger.debug('No knobs.')
        alpha = [1]
        beta = [0.5]
        gamma = [0.5]
        name_id = '_no'
        if allow_shift:
            shift = True

    # Loading features if necessary.
    if features.empty:
        features = pd.read_csv(os.path.join(path_root, features_name))

    output = pd.DataFrame()
    output['images_id'] = features['images_id']
    features = features.drop(['images_id'], axis=1)

    # b: Dataset repetition.
    for i in range(b):
        output_ = pd.DataFrame()
        for j, (alpha_, beta_, gamma_) in enumerate(list(itertools.product(alpha, beta, gamma))):
            np.random.seed(i * 100 + j)
            # Creating weights.
            phi, eta1, eta0 = _generation_weights(features.shape[1], alpha_)
            # Creating treatment assigment.
            t = _t_function(features.values, phi, beta_)
            # Creating outcomes.
            y_treat, y_control = _y_function(features.values, eta1, eta0, gamma_, shift=shift, alpha=alpha_)
            # Creatiing observed outcome.
            y = [y_control[i][0] if p == 0 else y_treat[i][0] for i, p in enumerate(t)]

            # Saving.
            knob = str(alpha_) + '_' + str(beta_) + '_' + str(gamma_)
            prefix = 'sim' + name_id + str(j) + '_b' + str(i) + '_' + knob
            output_[prefix + '-t'] = t
            output_[prefix + '-y1'] = y_treat
            output_[prefix + '-y0'] = y_control
            output_[prefix + '-y'] = y
        output = pd.concat([output, output_], axis=1)

    logger.debug('Simulation is done!')
    with gfile.GFile(os.path.join(path_root, output_name + name_id + '.csv'), 'w') as out:
        out.write(output.to_csv(index=False))
    return output_name + name_id + '.csv'


def join_tfrecord_csv(path_simulations,
                      path_input,
                      path_output,
                      input_prefix,
                      output_prefix='trainNew',
                      ):
    """Joining TFRecord files and joined_simulations.csv.

    :param path_simulations: str, path to joined_simulations.csv file.
    :param path_input: str, path to tfrecords.
    :param path_output: str, path to save new tfrecords.
    :param input_prefix: str, prefix tfrecords.
    :param output_prefix: str, prefix new tfrecords.
    :return: list []. Names of failed tfrecords.
    """

    def _make_filepaths(path_input, path_output, input_prefix, output_prefix):
        """ Creates lists with tfrecords files of a given prefix inside a folder.

        :param path_input: str, tfrecords path.
        :param path_output: str, new tfrecords path.
        :param input_prefix: str, tfrecords prefix.
        :param output_prefix: str, new tfrecors prefix.
        :return: filenames list input_prefix[], filenames list output_prefix[]
        """
        # Fetch TFRecord shards.
        tfrecord_input_filenames = []
        tfrecord_output_filenames = []

        for filename in os.listdir(str(path_input)):
            if filename.startswith(input_prefix):
                tfrecord_input_filenames.append(os.path.join(path_input, filename))
                fileout = filename.replace(input_prefix, output_prefix)
                tfrecord_output_filenames.append(os.path.join(path_output, fileout))

        # print('_make_filepaths',tfrecord_output_filenames)
        return tfrecord_input_filenames, tfrecord_output_filenames

    def _add_labels_to_tfrecords(path_input, path_output, label_records, overwrite=True):
        """Adding the labels to tfrecords.

        Runs the pipeline, constructing new labeled UKB TFRecords
        :param path_input: str, path to tfrecords.
        :param path_output: str, path to new tfrecords.
        :param label_records: dict, id: label.
        :param overwrite: bool, overwrites files.
        :return:
        """
        # Only regenerate existing TFRecords if `overwrite==True`.
        if os.path.exists(str(path_output)) and not overwrite:
            _print_status('SKIPPED (`output_path` already exists)', tfrecord_path,
                          output_path)
            return
        try:
            dict_id_to_encoded_images = _map_id_to_encoded_images(path_input)
            dict_label_tfrecords = _build_label_tensor_dicts(dict_id_to_encoded_images, label_records)
            tf_examples = _convert_tensor_dicts_to_examples(dict_id_to_encoded_images, dict_label_tfrecords)
            _write_tf_examples(tf_examples, path_output)
            _print_status('completed successfully', path_input, path_output)
            return None
        except ValueError:
            print_status('FAILED', tfrecord_path, output_path)
            return tfrecord_path

    def _map_id_to_encoded_images(path_input):
        """ Mapping ids to images.

        Returns a dictionary of encoded image tensors keyed on image id.
        :param path_input: str, tfrecords path.
        :return: dictionary id: image.
        """
        # Load and parse the TFRecords located at `ukb_tfrecord_path`.
        tfrecord_ds = tf.data.TFRecordDataset(filenames=path_input)
        features = _build_features()
        parsed_tfrecord_ds = tfrecord_ds.map(_get_parse_example_fn(features),
                                             num_parallel_calls=tf.data.AUTOTUNE)
        # Build a map (dict) of image ids to encoded image tensors.
        id_to_encoded_images = _map_id_to_encoded_image(parsed_tfrecord_ds)  # dict[id]=image
        return id_to_encoded_images

    def _build_features():
        """Returns a feature dictionary used to parse TFRecord examples.

        We assume that the UKB TFRecords are defined using the following schema:

          1. An encoded image with key `IMAGE_ENCODED_TFR_KEY` that can be decoded
             using `tf.image.decode_png`.
          2. A unique identifier for each image with key `IMAGE_ID_TFR_KEY`.

        The `tf.io.parse_single_example` function uses the resulting feature
        dictionary to parse each TFRecord.

        Returns:
          A feature dictionary for parsing TFRecords.
        """
        features = {
            'image/encoded': tf.io.FixedLenFeature([], dtype=tf.string),
            'image/id': tf.io.FixedLenFeature([], dtype=tf.string),
            'image/side': tf.io.FixedLenFeature([], dtype=tf.int64),
            'image/target': tf.io.FixedLenFeature([], dtype=tf.int64)
        }
        return features

    def _get_parse_example_fn(features):
        """Returns a function that parses a TFRecord example using `features`."""

        def _parse_example(example):
            return tf.io.parse_single_example(example, features)

        return _parse_example

    def _map_id_to_encoded_image(tfrecord_ds):
        """Convert a `tf.data.Dataset` containing images and ids to a tensor dict."""
        ids_to_encoded = {}
        for example in tfrecord_ds:
            # for example in next(iter(tfrecord_ds)):
            image_id = example['image/id'].numpy().decode('utf-8')
            image_encoded = example['image/encoded']
            ids_to_encoded[image_id] = image_encoded
        return ids_to_encoded

    def _build_label_tensor_dicts(dict_id_to_encoded_images, label_records):
        """Converts label records w/ids in `id_to_encoded_images` to tensor dicts.

        :param dict_id_to_encoded_images: dict, id: images.
        :param label_records:  dict, id: labels.
        :return: dict, id: labels (subset of labels present at dict_id_to_encoded_images).
        """
        dict_labels = {}
        for first_name in dict_id_to_encoded_images:
            if first_name not in label_records.keys():
                print("MISSING", first_name)
                continue
            dict_labels[first_name] = label_records[first_name]
        return dict_labels

    def _convert_tensor_dicts_to_examples(dict_id_to_encoded_images, dict_label_tfrecords):
        """Converts a list of tensor dict records to `tf.train.Example`s

        :param dict_id_to_encoded_images: dict, id: images.
        :param dict_label_tfrecords: dict, id:labels.
        :return: dict, id: {images, labels}.
        """
        examples = []

        for image_id in dict_label_tfrecords.keys():
            encoded_image = dict_id_to_encoded_images[image_id].numpy()
            examples.append(_build_tf_record_features(encoded_image,
                                                      image_id.encode('utf-8'),
                                                      dict_label_tfrecords[image_id]))
        return examples

    def _build_tf_record_features(encoded_image, image_id, simulations):
        """ Building tfrecords features.

        image + id + simulations

        :param encoded_image: image.
        :param image_id: str.
        :param simulations: dict, id:simulations
        :return: tf.train.Example.
        """
        # Specific for our dataset
        feature = {
            'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded_image])),
            'image/id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_id])),
        }
        for key in simulations.keys():
            if key != 'images_id':
                feature['image/' + key] = tf.train.Feature(float_list=tf.train.FloatList(value=[simulations[key]]))

        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto  # .SerializeToString()

    def _write_tf_examples(tf_examples, output_path):
        """ Writes a list of `tf.train.Example`s as TFRecords the `output_path`.

        :param tf_examples: dict, id:example.
        :param output_path: str, path to new tfrecords.
        :return: None
        """
        with tf.io.TFRecordWriter(str(output_path)) as writer:
            for example in tf_examples:
                writer.write(example.SerializeToString())

    def _print_status(status, tfrecord_path, output_path, error=None):
        """Prints the update status for the given paths.

        :param status: str, status message.
        :param tfrecord_path: str.
        :param output_path: str, new tfrecords path.
        :param error: None
        :return:
        """
        lines = [
            f'\nTFRecord update {status}:'
            f'\n\ttfrecord_path="{tfrecord_path}"',
            f'\n\toutput_path="{output_path}"',
        ]
        if error:
            lines.append(f'\n\terror=\n{error}')

    # Loading simulations.
    simulations = pd.read_csv(path_simulations)
    print('simulations', len(pd.unique(simulations['images_id'])), simulations.shape)
    if path_input == path_output:
        raise ValueError('Input and Output path should be different!')

    label_records = {
        record['images_id']: record
        for record in simulations.to_dict('records')
    }
    tfrecord_input_filenames, tfrecord_output_filenames = _make_filepaths(path_input=path_input,
                                                                          path_output=path_output,
                                                                          input_prefix=input_prefix,
                                                                          output_prefix=output_prefix)
    print('tfrecord_input_filenames', tfrecord_input_filenames)

    failed_paths_with_none = []
    for input_path, output_path in zip(tfrecord_input_filenames,
                                       tfrecord_output_filenames):
        failed = _add_labels_to_tfrecords(path_input=input_path,
                                          path_output=output_path,
                                          label_records=label_records)
        if failed:
            failed_paths_with_none.append(failed)

    # Print the filepaths of any failed runs for reprocessing.
    failed_paths = [str(path) for path in failed_paths_with_none if path]
    if failed_paths:
        failed_path_str = '\n\t'.join([''] + failed_paths)
        print(f'The following TFRecord updates failed:{failed_path_str}')
    print('DONE with JOIN features')


def organizing_simulations(path_features, simulations_files=[]):
    """ Organize all simulations in a single file.

    One might choose to create several simulation files (one file per knob, or a knob with different parameters).
    This function will combine all simulation files in the same folder into a single simulation file and also
    creates a true_tau.csv file, which contains the true value per simulation index.

    :param path_features: str, features folder.
    :param simulations_files: [], if given, it loads only these simulations.csv files.
    :return: None
    """
    #  1.Find all files in folder starting with simulations_.
    if len(simulations_files) == 0:
        simulations_files = []
        for item in os.listdir(path_features):
            if item.startswith('simulations_'):
                simulations_files.append(item)

    #  2. Join all files simulations_*.
    join_all_simulations = pd.DataFrame()
    for file in simulations_files:
        sim = pd.read_csv(os.path.join(path_features, file))
        if join_all_simulations.empty:
            join_all_simulations = sim
        else:
            join_all_simulations = pd.merge(join_all_simulations, sim, how='outer', on='images_id')

    #  3. Write joined simulations.
    with gfile.GFile(os.path.join(path_features, 'joined_simulations' + '.csv'), 'w') as out:
        out.write(join_all_simulations.to_csv(index=False))

    #  4. Getting unique simulation id (sim_id).
    sim_id = list(join_all_simulations.columns)
    sim_id.remove('images_id')
    sim_id = [item.split('-')[0] for item in sim_id]
    sim_id = pd.unique(sim_id)

    # 5. Calculating true tau and writting into a csv file.
    tau = pd.DataFrame(
        columns={'sim_id', 'tau', 'setting_id', 'knob', 'setting', 'repetition', 'alpha', 'beta', 'gamma'})
    for item in sim_id:
        y1 = join_all_simulations[[item + '-y1']].values
        y0 = join_all_simulations[[item + '-y0']].values
        ite = y1 - y0
        items = item.split('_')

        tau_ = {'sim_id': item,
                'tau': np.mean(ite),
                'knob': items[1][0:2],
                'setting': items[1][-1],
                'setting_id': items[1],
                'repetition': items[2],
                'alpha': items[3],
                'beta': items[4],
                'gamma': items[5]
                }
        tau = tau.append(tau_, ignore_index=True)
    with gfile.GFile(os.path.join(path_features, 'true_tau' + '.csv'), 'w') as out:
        out.write(tau.to_csv(index=False))

    tau.sort_values('repetition', inplace=True)
    tau.reset_index(inplace=True)
    tau.drop('index', axis=1, inplace=True)
    with gfile.GFile(os.path.join(path_features, 'true_tau_sorted' + '.csv'), 'w') as out:
        out.write(tau.to_csv(index=False))
