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

""" Working with diabetic-retinopathy-detection data from Kaggle.
This file organizes and prepere the data from Kaggle to be used by the models.

1. We will use only train.zip files as we want to have the true targets for our samples;
2. There are 35126 training samples - splited into 5 zips:
    a. train.zip.001 and train.zip.002 will be use to train feature extractor.
    b. train.zip.003 and train.zip.004 will be used as train set.
    c. train.zip.005 will be test.
3. We will download the images and re-write them as TFRecords. There will be two prefix:
    - train_prefix: images used to train the feature extractor.
    - prefix_extract: images from where features will be extract, and used for causal inference analysis.

Reference: https://www.kaggle.com/cdeotte/how-to-create-tfrecords#Verify-TFRecords
"""

import cv2
import math
import numpy as np
import os
import pandas as pd
import sys
import tensorflow as tf

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split


AUTOTUNE = tf.data.AUTOTUNE
IMAGE_SIZE = [256, 256]


def write_images_as_tfrecord(paths, tfrecord_size=1024, xfe_proportion_size=0.33):
    """ Read images -> Write as TFRecord
    Writes two types of TFrecords in paths['write in']:
    1) extract* files: (1-xfe_proportion_size)% of images;
    2) train* files: xfe_proportion_size% of images;

    :param paths: dictionary, keys: path_meta,path_root, path_images_png, path_tfrecords.
    :param tfrecord_size: int, quant. of images group together.
    :param xfe_proportion_size: float, value between 0.1 and 0.9.
    :return: None (files are written in folder).
    """
    def _writer_loop(image_files_path,
                     tfrecord_size,
                     paths,
                     meta_data,
                     set_name,
                     serialize_example):
        """ Writer loop that loads/write TFrecords.

        :param image_files_path: list with path to all images;
        :param tfrecord_size: int, num of images inside a TFRecord;
        :param paths: dictionary, keys: path_meta,path_root, path_images_png, path_tfrecords;
        :param meta_data: pd.Dataframe file with meta data;
        :param set_name: string, train/val/test;
        :param serialize_example: how to serialize a sample;
        :return: None;
        """
        n_tfrecords = len(image_files_path) // tfrecord_size + int(len(image_files_path) % tfrecord_size != 0)
        base_name = os.path.join(paths['path_root'], paths['path_tfrecords'])
        for j in range(n_tfrecords):
            print('Writing TFRecord - %s: %i of %i...' % (base_name, j, n_tfrecords))
            nper_tfrecord = min(tfrecord_size, len(image_files_path) - j * tfrecord_size)
            with tf.io.TFRecordWriter('%s%s%.2i-%i.tfrec' % (base_name, set_name, j, nper_tfrecord)) as writer:
                for k in range(nper_tfrecord):
                    path = os.path.join(paths['path_root'], paths['path_images_png'], image_files_path[tfrecord_size * j + k])
                    img = cv2.imread(path)
                    img = cv2.imencode('.jpg', img, (cv2.IMWRITE_JPEG_QUALITY, 94))[1].tostring()
                    name = image_files_path[tfrecord_size * j + k].split('.')[0]
                    row = meta_data.loc[meta_data.image_name == name]
                    example = serialize_example(img, name.encode(), row)
                    writer.write(example)

    def _build_tf_record_features(img, name, row):
        """Returns a string of the example.
        :param img, matrix.
        :param name, str.
        :param row.
        :return: tf.train.Example()
        """
        # Specific for our dataset
        feature = {
            'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img])),
            'image/id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[name])),
            'image/side': tf.train.Feature(int64_list=tf.train.Int64List(value=[row.side.values])),
            'image/target': tf.train.Feature(int64_list=tf.train.Int64List(value=[row.target.values])),
        }
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()

    # 0. Setting prefixes
    prefix_train = 'train'
    prefix_extract = 'extract'

    #  1. Loading metadata:
    meta_data = pd.read_csv(os.path.join(paths['path_root'], paths['path_meta']))
    meta_data.rename({'image': 'image_name', 'level': 'target'}, axis=1, inplace=True)
    meta_data['side'] = meta_data.apply(lambda x: x['image_name'].split('_')[1], axis=1)
    meta_data['side'] = meta_data.apply(lambda x: 1 if x['side'] == 'right' else 0, axis=1)  # Right is 1, Left is 0

    # 2. Recover the paths to all images
    image_files = os.listdir(os.path.join(paths['path_root'], paths['path_images_png']))

    #  3. Split images between train and extract (fixed seed for reproducibility)
    image_files_extract, image_files_train = train_test_split(image_files,
                                                              test_size=xfe_proportion_size,
                                                              random_state=0)
    #  4. Write the tfrecord files with the appropriate prefix.
    _writer_loop(
        image_files_path=image_files_train,
        tfrecord_size=tfrecord_size,
        paths=paths,
        meta_data=meta_data,
        serialize_example=_build_tf_record_features,
        set_name=prefix_train,
    )
    _writer_loop(
        image_files_path=image_files_extract,
        tfrecord_size=tfrecord_size,
        paths=paths,
        meta_data=meta_data,
        serialize_example=_build_tf_record_features,
        set_name=prefix_extract,
    )


def get_batched_dataset(filenames, type='TrainFeatureExtractor', train=True, batch_size=16):
    """
    Two different types of image_decoder:
    1) TrainFeatureExtractor - uses _decode_img_train_feature_extractor
        features: {'image/encoded','image/id', 'image/side', 'image/target'}
        batch returns image, target
    2) ExtractFeatures - uses _decode_img_extrac_features
        features: {'image/encoded','image/id', 'image/side', 'image/target'}
        batch returns image, target, image_id

    Dataset performance guide: https://www.tensorflow.org/guide/performance/datasets

    :param filenames: array with filenames.
    :param type: str, options are TrainFeatureExtractor and ExtractFeatures.
    :param train: bool, indicate if datasets is used for training or validation.
    :param batch_size: int.
    :return: dataset tf.data.Dataset
    """

    def _decode_img(example):
        """ Decode images from TFRecord (USED ONLY TO TRAIN FEATURE EXTRACTOR).

        The batches are used to train the feature extractor model.

        :param example: file/sample.
        :return: batch image, target.
        """
        image_size = IMAGE_SIZE
        features = {
            'image/encoded': tf.io.FixedLenFeature([], dtype=tf.string),
            'image/id': tf.io.FixedLenFeature([], dtype=tf.string),
            'image/side': tf.io.FixedLenFeature([], dtype=tf.int64),
            'image/target': tf.io.FixedLenFeature([], dtype=tf.int64)
        }
        example = tf.io.parse_single_example(example, features)
        image = tf.image.decode_jpeg(example['image/encoded'], channels=3)
        image = tf.cast(image, tf.float32) / 255.0
        image = tf.image.resize(image, image_size)

        return image, example['image/target']

    def _decode_img_and_id(example):
        """ Decode images from TFRecord (USED ONLY TO IN FEATURE EXTRACTION).

        The batches are used to extract features (later used in the join and causal inf.).

        :param example: file/sample.
        :return: batch image, target, id.
        """
        image_size = IMAGE_SIZE
        features = {
            'image/encoded': tf.io.FixedLenFeature([], dtype=tf.string),
            'image/id': tf.io.FixedLenFeature([], dtype=tf.string),
            'image/side': tf.io.FixedLenFeature([], dtype=tf.int64),
            'image/target': tf.io.FixedLenFeature([], dtype=tf.int64)
        }
        example = tf.io.parse_single_example(example, features)
        image = tf.image.decode_jpeg(example['image/encoded'], channels=3)
        image = tf.cast(image, tf.float32) / 255.0  # [0,255] -> [0,1]
        image = tf.image.resize(image, image_size)
        return image, example['image/target'], example['image/id']

    def _load_dataset(filenames, type):
        """  From filenames array, pick which _decode_img to use based on type.
        Options:
            ExtractFeatures: batch contains image, target, id.
            TrainFeatureExtractor: batch contains image, target.

        :param filenames: array, [].
        :param type: str, TrainFeatureExtractor or ExtractFeatures.
        :return: dataset  tf.data.Dataset.
        """
        print(filenames)
        option_no_order = tf.data.Options()
        option_no_order.experimental_deterministic = False
        dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE)
        # image_size is new
        if type == 'TrainFeatureExtractor':
            dataset = dataset.map(_decode_img, num_parallel_calls=AUTOTUNE)
        elif type == 'ExtractFeatures':
            dataset = dataset.map(_decode_img_and_id, num_parallel_calls=AUTOTUNE)
        else:
            raise NotImplementedError('Option not defined')
        return dataset

    #  1. Load dataset based on filename and type.
    dataset = _load_dataset(filenames=filenames, type=type)
    #  2. Training dataset: repeat then batch (Best practices for Keras).
    if train:
        # Evaluation dataset: do not repeat.
        dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(AUTOTUNE)  # prefetch next batch while training (autotune prefetch buffer size).
    return dataset


def build_datasets_feature_extractor(dataset_config, prefix_train, prefix_extract, type='ExtractFeatures'):
    """Returns train and evaluation datasets.

    Datasets are decoded from png images using RGB, [0,1].
    Args:
    :param dataset_config: keys={'path', 'batch_size', 'image_size'}.
    :param prefix_train: str, name prefix used to write TFRecods of images used to train the feature extractor.
    :param prefix_extract: str, name prefix used to write TFRecords of images from where we extract the feactures, and used
    for causal inference analysis.
    :param type: str, ExtractFeatures or ExtractFeatures.

    :return train_ds: tf.data.Dataset, keys={'image/encoded', image/target'}.
    :return extract_ds: tf.data.Dataset, keys={'image/id', 'image/encoded', image/target'}.
    """
    path = os.path.join(dataset_config['path_root'], dataset_config['path_tfrecords'])
    dataset_config['batch_size'] = dataset_config.get('batch_size', 16)
    filenames_train_feat_extractor = tf.io.gfile.glob(path + prefix_train + '*.tfrec')
    filenames_extract_features_from = tf.io.gfile.glob(path + prefix_extract + '*.tfrec')

    train_ds = get_batched_dataset(filenames_train_feat_extractor, type=type, train=True,
                                   batch_size=dataset_config['batch_size'])
    extract_ds = get_batched_dataset(filenames_extract_features_from, type=type, train=False,
                                     batch_size=dataset_config['batch_size'])

    return train_ds, extract_ds
