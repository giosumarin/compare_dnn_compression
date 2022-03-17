# On the Choice of Deep Neural Networks Compression: a Comparative Study
This repository contains the code allowing to reproduce the results described in G. Marinò et al.,
_On the Choice of Deep Neural Networks Compression: a Comparative Study_, 
extending the conference paper presented at [ICPR2020](https://www.micc.unifi.it/icpr2020/), whose package and paper are available at [this repository](https://github.com/giosumarin/ICPR2020_sHAM).

This package improves the previous contribution by adding new quantization strategies, and  
convolutional layers are now supported in addition to fully-connected layers.

We also included the method Sparse Low-Rank Factorization (SLR) to the comparison.

The `experiments` folder contains the basic scripts to replicate the tests carried out in the
our paper.


## Getting Started

### Prerequisites

* Install `python3`, `python3-pip` and `python3-venv` (Debian 10.6).
* Make sure that `python --version` starts by 3 or execute `alias python='pyhton3'` in the shell.
* For CUDA configuration (if a GPU is available) follow https://www.tensorflow.org/install/gpu.
* Create a virtual environment and install the required depedencies: `pip install tensorflow==2.2.0 click==8.0.1 matplotlib==3.4.3 sklearn numpy==1.20 numba==0.54.0 pympler==0.9 pytest`. The set of suggested dependencies have been tested with Python 3.7. We also ran the code with later versions of python and tensorflow more recent than the ones mentioned above.
* Finally, from the root of this repository, install the Python package through pip: `pip install ./sHAM_package`.

### Additional data
The trained models -- as well as data required by DeepDTA -- are rather big, so they are not versioned. Rather, 
they are available for download. You need to [download](https://www.mediafire.com/file/fcd92fohngitd0k/experiments.zip/file) it. 
Extract the downloaded zip and merge it into the `experiments` directory.

## Usage
Compression and compact storage of the network are separately executed in two stages.
1. To apply pruning and/or quantization to a model, we provide the `compression.py` script in the
`experiments/performance_eval/X` directory, where `X` is one of `VGG19` and `DeepDTA`. 
These scripts are customized for VGG19 and DeepDTA networks, and a minimal runner script is contained 
in each network sub-directory. The script to apply SLR, which does not require retraining, is `experiments/space_performance/uws_slrf_space.py`. This script can be used on a model with quantized convolutional layers or directly on an uncompressed model.
2. To store and evaluate the trained network with either HAM or sHAM we provide the `uws_testing_space.py`
example script in the `experiments/space_performance` directory, as well with a sample runner script. Please note that this 
script must be executed after those at point 1, since they generate the compressed models to be evaluated
here and the corresponding folders. If at point 1 only partial tests are executed, modify this script accordingly to
evaluate just  the model generated.

### Time estimation for examples
Running the examples we provide, [experiments/performance_eval/VGG19/runner_example.sh](https://github.com/giosumarin/compare_dnn_compression/blob/main/experiments/performance_eval/VGG19/runner_example.sh) and [experiments/performance_eval/DeepDTA/runner_example.sh](https://github.com/giosumarin/compare_dnn_compression/blob/main/experiments/performance_eval/DeepDTA/runner_example.sh), takes about an hour on a laptop with RTX2060 mobile GPU.

## Usage on other neural network/datasets

To perform a compression on a new model follow the following steps:
1. Train the model and save it via `model.save('model_name.h5')`.
2. Go to [experiments/performance_eval/VGG19](https://github.com/giosumarin/compare_dnn_compression/tree/main/experiments/performance_eval/VGG19)
3. Add a new function related to your dataset to the `datasets.py` script
4. Open the `compression.py` script, add the import of the function created at the point above (line 7), add a new branch to the `if` for choosing the dataset (line 89), modify the optimizer if necessary and the loss for retraining after compression (lines 123 and 124, for optimization we noticed better performance when using the same optimizer as the model you want to compress with a slightly lower learning rate, as happens with tranfer learning).
5. You are now ready to run the compression.py script, to see the various necessary settings you can run `python compression.py --help`, the re-training when using patience is based on the first metric you compiled the original model


### License
This software is distributed under the [Apache-2.0 License](https://github.com/giosumarin/compare_dnn_compression/blob/main/LICENSE).
