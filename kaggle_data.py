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

from zipfile import ZipFile
import tensorflow as tf, re, math
import pandas as pd
import pathlib
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt, cv2


def write_wraper(paths, tfrecord_size=1024, use_test=False):
    """ Read images -> Write as TFRecord
    Writes two types of TFrecords in paths['write in']:
    1) train* files: 67% of images
    2) val* files: 33% of images

    :param paths: dictionary, keys: meta (.csv file), root, images, write in
    :param tfrecord_size:
    :param use_test:
    :return: None
    """
    # Loading metadata:
    meta_data = pd.read_csv(paths['meta'])
    meta_data.rename({'image': 'image_name', 'level': 'target'}, axis=1, inplace=True)
    meta_data['side'] = meta_data.apply(lambda x: x['image_name'].split('_')[1], axis=1)
    meta_data['side'] = meta_data.apply(lambda x: 1 if x['side'] == 'right' else 0, axis=1)  # Right is 1, Left is 0

    # Paths to all images available
    image_files = os.listdir(os.path.join(paths['root'], paths['images']))

    if use_test:
        print('TODO')
    else:
        image_files_train, image_files_val = train_test_split(image_files, test_size=0.33)

    _writer_loop(
        image_files_path=image_files_train,
        tfrecord_size=tfrecord_size,
        paths=paths,
        meta_data=meta_data,
        serialize_example=_build_tf_record_features,
        set_name='train',
    )
    _writer_loop(
        image_files_path=image_files_val,
        tfrecord_size=tfrecord_size,
        paths=paths,
        meta_data=meta_data,
        serialize_example=_build_tf_record_features,
        set_name='val',
    )


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


def _decode_img(record_bytes, image_size=(587, 587)):
    labels = {
        'image/encoded': tf.io.FixedLenFeature([], dtype=tf.string),
        'image/id': tf.io.FixedLenFeature([], dtype=tf.string),
        'image/side': tf.io.FixedLenFeature([], dtype=tf.int64),
        'image/target': tf.io.FixedLenFeature([], dtype=tf.int64)
    }
    example = tf.io.parse_single_example(record_bytes, labels)
    image = tf.image.decode_jpeg(example['image/encoded'], channels=3)
    image = (tf.cast(image, tf.float32)) / 255  # [0,255] -> [0,1]
    image = tf.image.resize(image, image_size)
    # image = tf.math.multiply(image, 2.0)
    # image = tf.math.subtract(image, 0.1 )
    # return image, example
    labels = example['image/target']
    return image, labels


def _load_data(path, set_name, batch_size=16, ):
    filenames = tf.io.gfile.glob(path + set_name + '*.tfrec')
    tf_record_ds = tf.data.TFRecordDataset(filenames)
    dataset = tf_record_ds.map(_decode_img)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


def build_datasets(dataset_config):
    """Returns train and evaluation datasets.

  Datasets are decoded from png images using RGB, [0,1].
  Args:
    dataset_config: keys={'path', 'batch_size', 'image_size'}
  Returns:
    train_ds: tf.data.Dataset, keys={'image/id', 'image/encoded',
    'image/outcome_name/value','image/outcome_name/weight'}
    pred_ds: tf.data.Dataset, keys={'image/id', 'image/encoded',
    'image/outcome_name/value','image/outcome_name/weight'}
  """
    train_ds = _load_data(path=dataset_config['path'], set_name='train',
                          batch_size=dataset_config['batch_size'])
    pred_ds = _load_data(path=dataset_config['path'], set_name='val',
                         batch_size=dataset_config['batch_size'])

    return train_ds, pred_ds
