# Images as Covariates in Estimation of Treatment Effects Analysis (ICETEA)

Last Update: 

Semi-synthetic benchmark based on images. The observed images are used as covariates (X), 
the treatment assigment (T) is simulated, and the continuous outcome (Y) can be either 
simulated or modified from a observed covariate. For mode details on the 
creation of the dataset, please check ADD LINK TO PAPER. 

Ideal use cases: 
* Evaluation of methods that need to perform confounder adjustment. 
* Evaluation of causal inference methods with ML components;

We provide a use case of our framework with retinal
fundus images. Recent evidence suggests retinal fundus images are
correlated with general vascular conditions, which affects several medical
conditions. Therefore, it is reasonable to consider include retinal
images to make a confounder adjustment to study causal effects. 
Kaggle hosted a [Diabetic Retinopathy Detection competition](https://www.kaggle.com/c/diabetic-retinopathy-detection), and its datasets 
contain ~35k retinal fundus images [[1]](###References). 


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

## Summary

This repo contains:

- `README.md`: This file, repository description and usage.

Files directly related to the framework:
- [`icetea_feature_extraction.py`](icetea_feature_extraction.py): Feature Extraction (example using Kaggle Retinal Images).
It train H() model, and extract features h=H(X).
- [`icetea_data_simulation.py`](icetea_data_simulation.py): Data Generation. Join existing tfrecords to new simulated data (T and Y).
It also contain knobs to better control simulated data.
- [`data_kaggle.py`](data_kaggle.py): Specific for the Kaggle dataset adopted to illustrate
our framework. Contain functions to trasnform the png images into tfrecords, and to load tf.data.Dataset obj.

Causal Inference experiments to explore the framework:
- [`estimators.py`](estimators.py): Treatment Effect Estimators (oahaca and aipw). There are
three base models available: resnet50, inceptionV3, linear regression;
- [`utils.py`](utils.py) : Creates ImageData obj used by the estimators, experiments loop based on parameters;
- [`config.py`](config.py): Based on parameters, it creates objectts used by the experiments.

Files related to UKB: (REMOVE?)
- `beam_utils.py`: remove?
- `ukb.py`: remove?
- `ukb_utils.py`: remove?
- `train.py` remove?

### Usage:

TODO ADD COLAB 


###References
[1] Cuadros J, Bresnick G. EyePACS: An Adaptable Telemedicine System for Diabetic Retinopathy Screening. Journal of diabetes science and technology (Online). 2009;3(3):509-516.


###TODOS: 
- Should I remove stuff related to IHDP and ACIC from previous experiments?
- Should I remove the files related to ukb.py?


This is not an officially supported Google product.
