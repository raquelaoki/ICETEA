import cv2
import matplotlib.pyplot as plt
import os
import pandas as pd
import tensorflow as tf
from tensorflow.io import gfile
import sys
import yaml
from google.cloud import storage


import config
import estimators
import data_kaggle as kd
import icetea_feature_extraction as fe
import icetea_data_simulation as ds
import utils
from tensorflow.python.client import device_lib

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"
    # The path to your file to upload
    # source_file_name = "local/path/to/file"
    # The ID of your GCS object
    # destination_blob_name = "storage-object-name"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name+source_file_name)

    blob.upload_from_filename(source_file_name)

    print(
        f"File {source_file_name} uploaded to {destination_blob_name}."
    )


def update_experiments(filename, path_root, path_features, row, status):
    list_of_datasets = pd.read_csv(os.path.join(path_root, path_features, filename + '.csv'))
    assert list_of_datasets['running'][row] != 'Done', 'SIMULATION ' + row + ' already done!'
    list_of_datasets['running'][row] = status
    if 'Unnamed: 0' in list_of_datasets.columns:
        list_of_datasets.drop('Unnamed: 0', axis=1, inplace=True)
    list_of_datasets.to_csv(os.path.join(path_root, path_features, filename + '.csv'))


def main(params_path, running_indexes_path, use_tpus_str):

    with open(params_path) as f:
        params = yaml.safe_load(f)
    params = params['parameters']


    with open(running_indexes_path) as f:
        running_indexes = yaml.safe_load(f)
    running_indexes = running_indexes['parameters']['running_indexes']


    if use_tpus_str=='True':
        use_tpus = True
    else:
        use_tpus = False

    if use_tpus:
        print('USING TUPS')
        try:
            cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
            #if tpu_address not in ("", "local"):
            tf.config.experimental_connect_to_cluster(cluster_resolver)
            tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
            strategy = tf.distribute.experimental.TPUStrategy(cluster_resolver)
            print("Running on TPU ", cluster_resolver.master())
            print("REPLICAS: ", strategy.num_replicas_in_sync)
            #strategy = tf.distribute.experimental.TPUStrategy(cluster_resolver)
            #return cluster_resolver, strategy
        except ValueError:
            raise BaseException(
                'ERROR: Not connected to a TPU runtime; please see the previous cell in this notebook for instructions!')
    else:
        print("DEVICES - ", device_lib.list_local_devices())
        strategy = None

    path_root = params['path_root']
    path_tfrecords_new = params['path_tfrecords_new']
    path_features = params['path_features']
    path_results = params['path_results']

    # Prefix of images after join (images + simulated t and y )
    prefix_trainNew = prefix_output = 'trainNew'

    paths_list = [path_tfrecords_new, path_features, path_results]
    paths_list = [os.path.join(path_root, path) for path in paths_list]

    for path in paths_list:
        assert tf.io.gfile.isdir(path), path + ': Folder does not exist!'

    list_of_datasets = pd.read_csv(os.path.join(path_root, path_features, 'true_tau_sorted.csv'))
    # print(list_of_datasets.shape, path_tfrecords_new)
    print(list_of_datasets[0:10])
    sim_id = list_of_datasets['sim_id']

    param_data = {
        'name': ['kagle_retinal'],
        'path_tfrecords': [os.path.join(path_root, path_tfrecords_new)],  # path_tfrecords
        'prefix_train': [prefix_trainNew],
        'image_size': [[256, 256]],
        'batch_size': params['batch_size'],
    }

    param_method = {
        'name_estimator': params['name_estimator'],
        'name_metric': ['mse'],
        'name_base_model': params['name_base_model'],
        'name_prop_score': ['LogisticRegression_NN'],
        'learn_prop_score': [False],
        'epochs': params['epochs'],
        'steps': params['steps'],
        'repetitions': params['repetitions']
    }

    parameters = config.MakeParameters(param_data, param_method, use_tpus=use_tpus, strategy=strategy)

    #running_indexes = params['running_indexes']

    for i in running_indexes:
        update_experiments('true_tau_sorted', path_root, path_features, i, status='yes')

    print('path',os.path.join(path_root, path_features, 'true_tau_sorted.csv'))
    list_of_datasets = pd.read_csv(os.path.join(path_root, path_features, 'true_tau_sorted.csv'))

    for i, sim_id in enumerate(sim_id):
        if i in running_indexes:
            print('running ', i)
            #  Loads dataset with appropried sim_id.
            data = utils.ImageData(seed=sim_id, param_data=parameters.config_data[0])
            #  Creates a temporary DataFrame to keep the repetitions results under this dataset;
            #  Meaning: data is loaded once, and we have several models (defined in parameters.config_methods)
            #  using this dataset.
            results_one_dataset = pd.DataFrame()
            for _config in parameters.config_methods:
                # utils.repead_experiment: (data, setting) x param_method.repetitions
                results_one_config = utils.repeat_experiment(data, _config, seed=i)
                results_one_config = pd.merge(results_one_config, list_of_datasets, how='left')
                results_one_dataset = pd.concat([results_one_dataset, results_one_config])
                results_one_dataset['sim_id'] = sim_id
                #results_one_dataset = pd.merge(results_one_dataset, list_of_datasets, how='left')
                results_one_dataset.to_csv(params['output_name'] + str(i) + '.csv')
                upload_blob(bucket_name=params['bucket_name'],
                            source_file_name=params['output_name'] + str(i) + '.csv',
                            destination_blob_name=path_results,
                            )

            results_one_dataset['sim_id'] = sim_id
            #results_one_dataset = pd.merge(results_one_dataset, list_of_datasets, how='left')
            results_one_dataset.to_csv(params['output_name'] + str(i) + '.csv')

            upload_blob(bucket_name=params['bucket_name'],
                        source_file_name=params['output_name'] + str(i) + '.csv',
                        destination_blob_name=path_results,
                        )
            update_experiments('true_tau_sorted', path_root, path_features, i, status='done')

    print('DONE')
    return


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
