""" ICETEA Feature Extractor.

Reference:
github.com/Google-Health/genomics-research/blob/main/ml-based-vcdr/learning/model_utils.py
"""

import cv2
import logging
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from tensorflow.io import gfile

# Local imports
import data_kaggle as kd
import icetea_data_simulation as ds

logger = logging.getLogger(__name__)


def feature_extrator_wrapper(config, prefix_train = 'train',prefix_extract = 'extract',prefix_trainNew='trainNew'):

    hp._consistency_check_feature_extractor(config)
    prefix_trainNew = prefix_output = prefix_trainNew

    paths = {
        'images': config['path_images_png'],  # folder
        'meta': config['meta'],  # file
        'write_in': config['path_tfrecords'],  # folder
        'root': config['path_root'],
    }
    # 1. From .PNG to TFRecord
    # Kaggle datasets contain the individual images and a csv file with targets.
    # This first section will combine the images with the csv and save as TFRecord.
    logger.info('1. Convert PNG to RFRecord.')
    kd.write_images_as_tfrecord(paths=paths,
                                xfe_proportion_size=config['xfe_proportion_size']
                                )

    # 2)Extracting features: it creates a new csv file called features.csv.
    logger.info('2. Train Feature Extractor.')
    config['path_tfrecords'] = os.path.join(config['path_root'], config['path_tfrecords'])
    config['path_features'] = os.path.join(config['path_root'], config['path_features'])

    model = fe.extract_hx(model_config)


def compile_model(model_config):
    """Builds a graph and compiles a tf.keras.Model based on the configuration."""

    model = _inceptionv3(model_config)
    losses = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    metrics = [tf.keras.metrics.CategoricalAccuracy(name='cat_accuracy')]
    optimizer = _build_optimizer(model_config)
    model.compile(
        optimizer=optimizer,
        loss=losses,
        metrics=metrics)

    model.summary()
    print(f'Number of l2 regularizers: {len(model.losses)}.')

    return model


def _inceptionv3(model_config, image_shape=[256, 256]):
    """Returns an InceptionV3 architecture as defined by the configuration.
    See https://tensorflow.org/api_docs/python/tf/keras/applications/InceptionV3.
    Args:
    model_config: A dict containing model hyperparamters.
    image_shape
    Returns:
    An InceptionV3-based model.
    """

    backbone = tf.keras.applications.InceptionV3(
        include_top=False,
        weights=model_config.get('weights', 'imagenet'),
        input_shape=[*image_shape, 3],
    )
    l2 = model_config.get('l2', None)
    kernel_regularizer = tf.keras.regularizers.L2(l2)
    model = tf.keras.Sequential([
        backbone,
        tf.keras.layers.Dense(model_config.get('hidden_size', 1024),
                              activation='relu',
                              kernel_regularizer=kernel_regularizer),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(model_config.get('num_classes', 5),
                              activation='softmax',
                              kernel_regularizer=kernel_regularizer)
    ])

    return model


def _build_optimizer(model_config):
    """Builds optimizer based on config file.

    :param model_config: dict with parameters
    :return: tf.keras.optimizers obj
    """
    initial_learning_rate = model_config.get('learning_rate', 0.001)
    steps_per_epoch = model_config.get('steps_per_epoch', 10)
    schedule_config = model_config.get('schedule', {})
    schedule_type = schedule_config.get('schedule', None)

    if schedule_type == 'exponential':
        decay_steps = int(schedule_config.epochs_per_decay * steps_per_epoch)
        learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=decay_steps,
            decay_rate=schedule_config.decay_rate,
            staircase=schedule_config.staircase)
    elif schedule_type is None:
        learning_rate = initial_learning_rate
        print(f'No LR schedule provided. Using a fixed LR of "{learning_rate}".')
    else:
        raise ValueError(f'Unknown scheduler name: {schedule_type}')

    opt_type = model_config.get('optimizer', None)
    if opt_type == 'adam':
        opt = tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=model_config.get('beta_1', 0.9),
            beta_2=model_config.get('beta_2', 0.999),
            epsilon=model_config.get('epsilon', 1e-07),
            amsgrad=model_config.get('amsgrad', False))
        return opt
    else:
        raise ValueError(f'Unknown optimizer name: {opt_type}')


def extract_hx(model_config):
    """ Function that extracts the features;

    :param model_config: dictionary with model settings. Required keys: {'epochs', 'steps_per_epoch', 'batch_size',
     'path_tfrecords', 'path_features','weights', 'hidden_size', 'num_classes', 'learning_rate', 'optimizer' }

    :return: model, and saves a .csv file with extracted features
    """

    #  1. Build tfrecord datasets - important to use type='TrainFeatureExtractor'
    dataset_train, dataset_extract = kd.build_datasets_feature_extractor(model_config,
                                                                         prefix_extract='extract',
                                                                         prefix_train='train',
                                                                         type='TrainFeatureExtractor')
    #  2. Creates model
    model = compile_model(model_config)
    model_config['epochs'] = model_config.get('epochs', 10)
    model_config['steps_per_epoch'] = model_config.get('steps_per_epoch', 2)

    #  3. Fits model on train* images, using dataset_extract only as validation
    model.fit(
        dataset_train,  # portion of the dataset used to train feature extractor only
        validation_data=dataset_extract,  # portion of the dataset used as covariates
        epochs=model_config['epochs'],
        initial_epoch=0,
        steps_per_epoch=model_config['steps_per_epoch'],
    )

    #  4. Creates a model with only the last layer of the image based model
    extract = tf.keras.Model(model.inputs, model.layers[-2].output)

    #  5. Rebuild the dataset using type 'ExtractFeatures' - it will return batch with IDs (used on join)
    _, dataset_extract = kd.build_datasets_feature_extractor(model_config,
                                                             prefix_extract='extract',
                                                             prefix_train='train',
                                                             type='ExtractFeatures')

    #  6. Extract features, and save them as .csv file on path_features.
    features = _extraction(dataset_extract, model_config['path_features'], extract)

    return model


def _extraction(data, path, model):
    """Extract features using the model (last layer of an image model - before prediction of classes) on the data.

  Args:
    data: built dataset.
    path: folder to save csv file.
    model: fitted model.
  """
    progbar = tf.keras.utils.Progbar(
        None,
        width=30,
        verbose=1,
        interval=0.05,
        stateful_metrics=None,
        unit_name='step')

    # features = []
    image_id = []
    features = pd.DataFrame()

    for i, (batch_images, batch_labels, batch_id) in enumerate(data):
        batch_predict = model.predict_on_batch(batch_images)
        # features.append(batch_predict)
        # print('_extraction',batch_predict.shape,batch_predict[0:3,0:3])
        features = pd.concat([features, pd.DataFrame(batch_predict)], axis=0)
        ##print('fetures', features.tail())
        image_id.append(batch_id)
        progbar.add(1)
        # if i > 10:
        #    break

    # features = np.array(features)
    # s = features.shape
    # print(
    #    '\nshapes pd ', featuresnew.shape
    # )
    # print('\nshapes ', s, s[0].shape)
    # features = features.reshape(s[0] * s[1], s[2])

    columns = [f'f{i}' for i in range(features.shape[1])]
    # features = pd.DataFrame(data=features,
    #                       columns=columns)
    features.columns = columns
    image_id = np.concatenate(image_id).ravel()
    image_id = [item.decode('utf-8') for item in image_id]
    # print('shape', features.shape)
    features['images_id'] = image_id
    with gfile.GFile(path + '/features.csv', 'w') as table_names:
        table_names.write(features.to_csv(index=False))
    return features
