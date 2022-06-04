# Images as Covariates in Estimation of Treatment Effects Analysis (ICETEA)

Last Update: 2022-06-02

Semi-synthetic benchmark based on images for Causal Inference evaluation. The observed images 
are used as covariates (X), the treatment assigment (T) is simulated, and the continuous outcome 
(Y) can be either simulated or modified from a observed covariate. For mode details on the 
creation of the dataset, please check ADD LINK TO PAPER. 

Ideal use cases: 
* Evaluation of causal inference methods.

The framework can be divided into two parts: 
1. Feature Extraction: We split the images available in two groups: X (default 70%) 
and W (defeault 100-70%). The images in W are used to train an image model that 
predicts an outcome associated with the original images. After training the model,
we remove the last layer (responsible for predicting the original outcome), and 
define this new model H().

2. Data Simulation: We extract features, defined as h = H(X). Note that X was never seen by the
image model H(). We use h to simulate the treatment assignment T, and the oucome Y 
(see the original publication for details on how to modify a observed clinical variable).

Causal inference analysis receive as input X (covariates/images), T(binary treat.)
and Y (continuous outcome). 

## Use-case:

We provide a use case of our framework with retinal
fundus images. Recent evidence suggests retinal fundus images are
correlated with general vascular conditions, which affects several medical
conditions. Therefore, it is reasonable to consider include retinal
images to make a confounder adjustment to study causal effects. 
Kaggle hosted a [Diabetic Retinopathy Detection competition](https://www.kaggle.com/c/diabetic-retinopathy-detection), 
and its datasets 
contain ~35k retinal fundus images [[1]](###References). 

## Dependencies

Please check [requirements.txt](requirements.txt) for a list of basic dependencies for most of the tools in this directory.
To install the dependencies in the Python virtual environment run

```shell
pip3 install -r requirements
```

## Summary

This repo contains:

- `README.md`: This file, repository description and usage.

Files directly related to the framework:
- [`icetea_feature_extraction.py`](icetea_feature_extraction.py): Feature Extraction (example using EyePACS). 
It trains the h() model (image model).
- [`icetea_data_simulation.py`](icetea_data_simulation.py): Data Generation, and creates new tfrecords with the simulated 
data (T and Y). The knobs are added in this phase.
- [`data_kaggle.py`](helper_data.py): Specific for the EyePACS dataset adopted to illustrate
our framework. Contain functions to trasnform the png images into tfrecords, and to load tf.data.Dataset.

Causal Inference experiments to explore the framework:
- [`estimators.py`](estimators.py): Treatment Effect Estimators (oahaca and aipw). There are
three base models available: resnet50, inceptionV3, linear regression;
- [`utils.py`](utils.py) : Creates ImageData obj used by the estimators, experiments loop based on parameters;
- [`config.py`](config.py): Based on parameters, it creates objectts used by the experiments.


# Usage:
This repository contains three options to create the benchmark dataset: using Colabs (Google Drive dependency), 
submitting a shell script , or a beam pipeline. These options differ on complexity and compute power. All these 
options assume at least one GPU available.

## 1. Colab (or Jupyter Notebook)

This option is the easiest to start exploring the dataset. Recommended to independent researchers that might not have 
access to a lot of computer power, or just want to run a small number of repetitions with the dataset. 

We created two notebooks that illustrates how to generate the dataset and run a
causal inference model using the datasets. 

- [icetea_feature_extraction_data_simulation.ipynb](notebooks/icetea_feature_extraction_data_simulation.ipynb): 
framework implementation - from loading png gfiles, to extracting features and simulating the data; 
- [icetea_causal_inference.ipynb](notebooks/icetea_causal_inference.ipynb): load the tfrecords with the simulated data, and 
estimate treatment effects

Obs: If running on Colab, the dataset should be storaged on Google Drive (~24GB).

## 2. Shell script

This option is ideal for researchers with access to cloud computing (clusters, Google Cloud, and similar services).

### Preparing the data
You will need following folders on your cloud computer: 
- ```root```: contains the train_labels.csv file, and the folders below.
  - ```icetea_data```: contains the png images (large dataset);
  - ```icetea_tfr```: contains the tfrecords of the png images (files used by the Feature Extractor)
  - ```icetea_features```: the features learned will be saved on this folder, along with the true_tau.csv and 
simulated Y and T.
  - ```icetea_newdata```: contains the tfrecods from ```icetea_tfr``` with the synthetic treatments and outcomes created
  by the data simulation process. 
  - ```icetea_results```: causal inference results are saved in this folder.

Feature Extractor:
```shell
# Loading yaml files:
python main.py --feature_extraction=True --load_yaml=True

# Using Flags:
```

Data Simulation:
```shell
# Loading yaml files:
python main.py --data_simulation=True --load_yaml=True

# Using Flags:

```

Causal Inference:
```shell
# Loading yaml files:
python main.py --causal_inference=True --load_yaml=True

# Using Flags:

```


Running causal inference estimators in the semi-synthetic dataset.
```shell
#!/bin/bash
#SBATCH --gres=gpu:1

echo 'Starting ....'
module load --ignore-cache python/3.7 cuda cudnn
SOURCEDIR= ~/projects/

python3 -m venv env
source env/bin/activate
git clone --branch adding_kaggle https://github.com/raquelaoki/icetea
pip install -r icetea/requirements.txt

python icetea/main.py --causal_inference=True --load_yaml=True
echo 'DONE!'
```

The amount of time and memory for this type of job depends on type of GPU, number of epochs, among other paramaters. 
Check setup_aipw.yaml and indexes_groupa.yaml for examples of these configuration files. 


## 3. Beam Pipeline (Complexity: high)
This option is recommened for who wants to run large scale experiments and has access to a lot of computing power. 

TODO 



###References
[1] Cuadros J, Bresnick G. EyePACS: An Adaptable Telemedicine System for Diabetic Retinopathy Screening. Journal of 
diabetes science and technology (Online). 2009;3(3):509-516.



This is not an officially supported Google product.
