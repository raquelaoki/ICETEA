""" ICETEA Data simulation.

"""

import concurrent.futures
import itertools
import math
import numpy as np
import os
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from tensorflow.io import gfile

# Local imports
import data_kaggle as kd


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
                         gamma_range=[]
                         ):
    """Generate the p(x) and mu(x,y) from h(x).

    Args:
    path_root: path to save output file (same where features are located)
    output_name: string with filename
    features: pd.DataFrame istead of reading (time consuming)
    path_features: path for the features_name.csv file.
    features_name: features filename
    b: repetitions
    knob_o: bool, True explores overlap knob
    knob_h: bool, True explores h knob
    knob_s: bool, True explores s knob
    beta_range: list with values to be explored if knob_o True (defoult values also provided)
    alpha_range: list with values to be explored if knob_h True (defoult values also provided)
    gamma_range: list with values to be explored if knob_s True (defoult values also provided)
    """

    def _generation_weights(n_cols=2048, alpha=2):
        """Generate initial weights.

        :param n_cols: dimension of weights.
        :param alpha: treatment effect knob.
        :return: matrices with weights.
        """
        phi = np.random.uniform(-1, 1, n_cols)
        # Multiply by 1.5 and 1.8 so it's not centered in 0.
        eta_1 = np.random.uniform(alpha, alpha * 1.5, n_cols)
        eta_0 = np.random.uniform(alpha, alpha * 1.8, n_cols)
        return phi, eta_1, eta_0

    def _pi_x_function(features, gam, beta=0.5):
        """ Calcualte the treatment assigment.
        :param features: input features, extracted on phase 1 of the framework;
        :param gam: weights
        :param beta: overlap knob
        :return: treatment assigment
        """

        def sigmoid(x):
            return 1 / (1 + math.exp(-x))

        assert 0 <= beta <= 1, 'Beta out of range [0,1]'
        pi = np.matmul(features, gam).reshape(-1, 1)
        scaler = MinMaxScaler((-2, 2))
        pi = scaler.fit_transform(pi)
        pi = pi.ravel()
        pi = [sigmoid(item) for item in pi]
        zeros = [1 if item > 0.5 else 0 for item in pi]
        pi = np.multiply(beta, pi)
        zeros = np.multiply(1 - beta, zeros)
        pi = pi + zeros
        t = [np.random.binomial(1, item) for item in pi]
        return t

    def _mu_x_function(features, eta1, eta0, gamma=1):
        """ Calcualte the outcome.
        :param features: input features, extracted on phase 1 of the framework;
        :param eta0: weights if untreated
        :param eta1: weights if treated
        :param gamma: knob_s values (treat effect)
        :return:
        """
        mu1 = np.array(np.matmul(features, eta1))
        mu0 = np.array(np.matmul(features, eta0))
        full = np.array(np.concatenate([mu1, mu0]))
        scaler = MinMaxScaler()
        scaler.fit(full.reshape(-1, 1))
        mu1 = scaler.transform(mu1.reshape(-1, 1))
        mu0 = scaler.transform(mu0.reshape(-1, 1))

        ones = np.ones(features.shape)
        mu1_ones = scaler.transform(np.array(np.matmul(ones, eta1)).reshape(-1, 1))
        mu0_ones = scaler.transform(np.array(np.matmul(ones, eta0)).reshape(-1, 1))

        mu1 = np.multiply(gamma, mu1)
        mu0 = np.multiply(gamma, mu0)
        mu1_ones = np.multiply(1 - gamma, mu1_ones)
        mu0_ones = np.multiply(1 - gamma, mu0_ones)

        return mu1 + mu1_ones, mu0 + mu0_ones

    assert knob_h + knob_s + knob_o == 1, 'Only one knob can be True!'

    if knob_o:
        # Make a range of values for beta
        alpha = [1]
        if len(beta_range) > 0:
            beta = beta_range
        else:
            beta = [0, 0.25, 0.5, 0.75, 1]
        gamma = [0.5]
        name_id = '_ko'
        output_name = output_name + name_id
    elif knob_h:
        # Make a range of values for gamma
        alpha = [1]
        beta = [0.5]
        if len(gamma_range) > 0:
            gamma = gamma_range
        else:
            gamma = [0, 0.25, 0.5, 0.75, 1]
        name_id = '_kh'
        output_name = output_name + name_id
    elif knob_s:
        # knob_s
        if len(alpha_range) > 0:
            alpha = alpha_range
        else:
            alpha = [0.1, 0.5, 1, 2, 8]
        beta = [0.5]
        gamma = [0.5]
        name_id = '_ks'
        output_name = output_name + name_id
    else:
        alpha = [1]
        beta = [0.5]
        gamma = [0.5]

    if features.empty:
        features = pd.read_csv(os.path.join(path_root, features_name))

    output = pd.DataFrame()
    output['images_id'] = features['images_id']
    features = features.drop(['images_id'], axis=1)
    for i in range(b):
        for j, (alpha_, beta_, gamma_) in enumerate(list(itertools.product(alpha, beta, gamma))):
            np.random.seed(i * 1000 + j)
            phi, eta1, eta0 = _generation_weights(features.shape[1], alpha_)
            pi = _pi_x_function(features.values, phi, beta_)
            mu1, mu0 = _mu_x_function(features.values, eta1, eta0, gamma_)
            y = mu1
            y[pi == 0] = mu0[pi == 0]

            knob = str(alpha_) + '_' + str(beta_) + '_' + str(gamma_)
            output['sim' + name_id + str(j) + '_b' + str(i) + '_' + knob + '-pi'] = pi
            output['sim' + name_id + str(j) + '_b' + str(i) + '_' + knob + '-mu1'] = mu1
            output['sim' + name_id + str(j) + '_b' + str(i) + '_' + knob + '-mu0'] = mu0
            output['sim' + name_id + str(j) + '_b' + str(i) + '_' + knob + '-y'] = y

    with gfile.GFile(os.path.join(path_root, output_name + '.csv'), 'w') as out:
        out.write(output.to_csv(index=False))

    return output


def join_tfrecord_csv(path_simulations,
                      path_input,
                      path_output,
                      input_prefix,
                      output_prefix='train_features'):
    """Join TFRecord files and a csv file with a common id.
    Args:
    path_simulations: path for csv file
    input_prefix: prefix of TFRecords
    path_input: path to TFRecords
    path_output: path to save TFRecors (should be differet to avoid overwrite)
    output_prefix: prefix of joined TFRecords
    """

    def _make_filepaths(path_input, path_output, input_prefix, output_prefix):
        # Fetch TFRecord shards.
        tfrecord_input_filenames = []
        tfrecord_output_filenames = []

        for filename in os.listdir(str(path_input)):
            if filename.startswith(input_prefix):
                tfrecord_input_filenames.append(os.path.join(path_input, filename))
                fileout = filename.replace(input_prefix, output_prefix)
                tfrecord_output_filenames.append(os.path.join(path_output, fileout))

        return tfrecord_input_filenames, tfrecord_output_filenames

    def _add_labels_to_tfrecords(path_input, path_output, label_records, overwrite=True):
        """Runs the pipeline, constructing new labeled UKB TFRecords."""
        # print('_add_labels_to_tfrecords',label_records)
        # Only regenerate existing TFRecords if `overwrite==True`.
        if os.path.exists(str(path_output)) and not overwrite:
            _print_status('SKIPPED (`output_path` already exists)', tfrecord_path,
                          output_path)
            return
        dict_id_to_encoded_images = _map_id_to_encoded_images(path_input)
        dict_label_tfrecords = _build_label_tensor_dicts(dict_id_to_encoded_images,
                                                         label_records)
        tf_examples = _convert_tensor_dicts_to_examples(dict_id_to_encoded_images, dict_label_tfrecords)
        _write_tf_examples(tf_examples, path_output)
        _print_status('completed successfully', path_input, path_output)

    def _map_id_to_encoded_images(path_input):
        """Returns a dictionary of encoded image tensors keyed on image id."""
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
        # image_prefix = 'image/'
        for example in tfrecord_ds:
            # print('type(tfrecord_ds', type(tfrecord_ds))
            # for example in next(iter(tfrecord_ds)):
            image_id = example['image/id'].numpy().decode('utf-8')
            image_encoded = example['image/encoded']
            ids_to_encoded[image_id] = image_encoded
        return ids_to_encoded

    def _build_label_tensor_dicts(dict_id_to_encoded_images, label_records):
        """Converts label records w/ids in `id_to_encoded_images` to tensor dicts."""
        dict_labels = {}
        for first_name in dict_id_to_encoded_images:
            if first_name not in label_records.keys():
                continue
            dict_labels[first_name] = label_records[first_name]
        return dict_labels

    def _convert_tensor_dicts_to_examples(dict_id_to_encoded_images, dict_label_tfrecords):
        """Converts a list of tensor dict records to `tf.train.Example`s."""
        examples = []

        for image_id in dict_label_tfrecords.keys():
            encoded_image = dict_id_to_encoded_images[image_id].numpy()
            # simulations = dict_label_tfrecords[image_id]
            examples.append(_build_tf_record_features(encoded_image,
                                                      image_id.encode('utf-8'),
                                                      dict_label_tfrecords[image_id]))
        return examples

    def _build_tf_record_features(encoded_image, image_id, simulations):
        """Returns a string of the example."""
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
        """Writes a list of `tf.train.Example`s as TFRecords the `output_path`."""
        with tf.io.TFRecordWriter(str(output_path)) as writer:
            for example in tf_examples:
                writer.write(example.SerializeToString())

    def _print_status(status, tfrecord_path, output_path, error=None):
        """Prints the update status for the given paths."""
        lines = [
            f'\nTFRecord update {status}:'
            f'\n\ttfrecord_path="{tfrecord_path}"',
            f'\n\toutput_path="{output_path}"',
        ]
        if error:
            lines.append(f'\n\terror=\n{error}')

    # Loading simulations.
    simulations = pd.read_csv(path_simulations)
    if path_input == path_output:
        raise ValueError('Input and Output path should be different!')

    label_records = [{
        record['images_id']: record
        for record in simulations.to_dict('records')
    }]
    tfrecord_input_filenames, tfrecord_output_filenames = _make_filepaths(path_input, path_output, input_prefix,
                                                                          output_prefix)

    # Process each shard in parallel, pairing images and labels.
    with concurrent.futures.ThreadPoolExecutor(max_workers=40) as executor:
        failed_paths_with_none = list(
            executor.map(_add_labels_to_tfrecords,
                         tfrecord_input_filenames,
                         tfrecord_output_filenames,
                         label_records))

    # Print the filepaths of any failed runs for reprocessing.
    failed_paths = [str(path) for path in failed_paths_with_none if path]
    if failed_paths:
        failed_path_str = '\n\t'.join([''] + failed_paths)
        print(f'The following TFRecord updates failed:{failed_path_str}')
    print('DONE with JOIN features')


def organizing_simulations(path_features):
    """ Organize all simulations in a single file.

    One might choose to create several simulation files (one file per knob, or a knob with different parameters).
    This function will combine all simulation files in the same folder into a single simulation file and also
    creates a true_tau.csv file, which contains the true value per simulation index.

    :param path_features:
    :return: sim_id (simulation unique id)
    """

    #  1.Find all files in folder starting with simulations_.
    simulations = []
    for item in os.listdir(path_features):
        if item.startswith('simulations_'):
            simulations.append(item)

    #  2. Join all files simulations_*.
    print('Joining simulations:')
    join_all_simulations = pd.DataFrame()
    for file in simulations:
        sim = pd.read_csv(os.path.join(path_features, file))
        if join_all_simulations.empty:
            join_all_simulations = sim
        else:
            #print(join_all_simulations.shape)
            join_all_simulations = pd.merge(join_all_simulations, sim, how='outer', on='images_id')
    #print(join_all_simulations.shape)

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
        mu1 = join_all_simulations[[item + '-mu1']].values
        mu0 = join_all_simulations[[item + '-mu0']].values
        ite = mu1 - mu0
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

    return sim_id
