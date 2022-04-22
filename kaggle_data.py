"""
1. We will use only train.zip files as we want to have the true targets for our samples;
2. There are 35126 training samples - splited into 5 zips:
    a. train.zip.001 and train.zip.002 will be use to train feature extractor
    b. train.zip.003 and train.zip.004 will be used as train set
    c. train.zip.005 will be test
3. Working with TFRecord files


References:
    https://www.kaggle.com/cdeotte/how-to-create-tfrecords#Verify-TFRecords

"""

import os, sys, math
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import cv2

# print("Tensorflow version " + tf.__version__)
AUTOTUNE = tf.data.AUTOTUNE


def write_images_as_tfrecord(paths, tfrecord_size=1024):
    """ Read images -> Write as TFRecord
    Writes two types of TFrecords in paths['write in']:
    1) train* files: 67% of images
    2) val* files: 33% of images

    :param paths: dictionary, keys: meta (.csv file), root, images, write in
    :param tfrecord_size:
    :param use_test:
    :return: None
    """

    def _writer_loop(image_files_path, tfrecord_size, paths, meta_data,
                     set_name, serialize_example):
        """ Writer loop that loads/write TFrecords.

        :param image_files_path: list with path to all images;
        :param tfrecord_size: int, num of images inside a TFRecord;
        :param paths: dictionary, keys: meta (.csv file), root, images, write in
        :param meta_data: pd.Dataframe file with meta data
        :param set_name: string, train/val/test
        :param serialize_example: how to serialize a sample
        :return: -
        """
        n_tfrecords = len(image_files_path) // tfrecord_size + int(len(image_files_path) % tfrecord_size != 0)
        base_name = os.path.join(paths['root'], paths['write_in'])
        for j in range(n_tfrecords):
            print()
            print('Writing TFRecord - %s: %i of %i...' % (base_name, j, n_tfrecords))
            nper_tfrecord = min(tfrecord_size, len(image_files_path) - j * tfrecord_size)
            with tf.io.TFRecordWriter('%s%s%.2i-%i.tfrec' % (base_name, set_name, j, nper_tfrecord)) as writer:
                for k in range(nper_tfrecord):
                    path = os.path.join(paths['root'], paths['images'], image_files_path[tfrecord_size * j + k])
                    img = cv2.imread(path)
                    img = cv2.imencode('.jpg', img, (cv2.IMWRITE_JPEG_QUALITY, 94))[1].tostring()
                    name = image_files_path[tfrecord_size * j + k].split('.')[0]
                    row = meta_data.loc[meta_data.image_name == name]
                    example = serialize_example(img, name.encode(), row)
                    writer.write(example)

    def _build_tf_record_features(img, name, row):
        """Returns a string of the example."""
        # Specific for our dataset
        feature = {
            'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img])),
            'image/id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[name])),
            'image/side': tf.train.Feature(int64_list=tf.train.Int64List(value=[row.side.values])),
            'image/target': tf.train.Feature(int64_list=tf.train.Int64List(value=[row.target.values])),
        }
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()
    # Loading metadata:
    meta_data = pd.read_csv(paths['meta'])
    meta_data.rename({'image': 'image_name', 'level': 'target'}, axis=1, inplace=True)
    meta_data['side'] = meta_data.apply(lambda x: x['image_name'].split('_')[1], axis=1)
    meta_data['side'] = meta_data.apply(lambda x: 1 if x['side'] == 'right' else 0, axis=1)  # Right is 1, Left is 0

    # Paths to all images available
    image_files = os.listdir(os.path.join(paths['root'], paths['images']))

    image_files_train, image_files_extract = train_test_split(image_files, test_size=0.33)

    _writer_loop(
        image_files_path=image_files_train,
        tfrecord_size=tfrecord_size,
        paths=paths,
        meta_data=meta_data,
        serialize_example=_build_tf_record_features,
        set_name='train',
    )
    _writer_loop(
        image_files_path=image_files_extract,
        tfrecord_size=tfrecord_size,
        paths=paths,
        meta_data=meta_data,
        serialize_example=_build_tf_record_features,
        set_name='extract',
    )


def get_batched_dataset(filenames, type='TrainFeatureExtractor', train=True, batch_size=16,
                        image_size=[587,587]):
    """
    Two different types of image_decoder:
    1) TrainFeatureExtractor - uses _decode_img_train_feature_extractor
        features: {'image/encoded','image/id', 'image/side', 'image/target'}
        batch returns image, target
    2) ExtractFeatures - uses _decode_img_extrac_features
        features: {'image/encoded','image/id', 'image/side', 'image/target'}
        batch returns image, target, image_id

    :param filenames:
    :param type: str
    :param train: indicate if datasets is used for training or validation
    :param batch_size: int
    :param image_size: list [size,size]
    :return:
    """

    def _decode_img(example):
        image_size = [256, 256]
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

        return image, example['image/target']

    def _decode_img_and_id(example):
        image_size = [256, 256]
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
        print(filenames)
        option_no_order = tf.data.Options()
        option_no_order.experimental_deterministic = False
        dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE)
        #dataset = dataset.with_options(option_no_order)
        # image_size is new
        if type == 'TrainFeatureExtractor':
            dataset = dataset.map(_decode_img, num_parallel_calls=AUTOTUNE)
        elif type == 'ExtractFeatures':
            dataset = dataset.map(_decode_img_and_id, num_parallel_calls=AUTOTUNE)
        else:
            raise NotImplementedError('Option not defined')
        return dataset

    dataset = _load_dataset(filenames=filenames, type=type)
    # dataset = dataset.cache()  # This dataset fits in RAM
    if train:
        # Best practices for Keras:
        # Training dataset: repeat then batch
        # Evaluation dataset: do not repeat
        dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(AUTOTUNE)  # prefetch next batch while training (autotune prefetch buffer size)
    # should shuffle too but this dataset was well shuffled on disk already
    return dataset
    # source: Dataset performance guide: https://www.tensorflow.org/guide/performance/datasets


def build_datasets_feature_extractor(dataset_config, train_suffix, val_suffix, type='ExtractFeatures'):
    """Returns train and evaluation datasets.

  Datasets are decoded from png images using RGB, [0,1].
  Args:
    dataset_config: keys={'path', 'batch_size', 'image_size'}
    extract: return ids
  Returns:
    train_ds: tf.data.Dataset, keys={'image/id', 'image/encoded',
    'image/outcome_name/value','image/outcome_name/weight'}
    pred_ds: tf.data.Dataset, keys={'image/id', 'image/encoded',
    'image/outcome_name/value','image/outcome_name/weight'}
  """
    filenames_train = tf.io.gfile.glob(dataset_config['path_tfrecords'] + train_suffix + '*.tfrec')
    filenames_val = tf.io.gfile.glob(dataset_config['path_tfrecords'] + val_suffix + '*.tfrec')

    train_ds = get_batched_dataset(filenames_train, type=type, train=True,
                                   batch_size=dataset_config['batch_size'])
    extract_ds = get_batched_dataset(filenames_val, type=type, train=False,
                                     batch_size=dataset_config['batch_size'])

    return train_ds, extract_ds
