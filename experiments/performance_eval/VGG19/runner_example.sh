#!/bin/bash

# This example applies pruning and uCWS quantization to the VGG19 model trained on the MNIST dataset.
# Pruning and clustering is only applied to fully connected layers (90% of pruning, 64 clusters).
# After compression we calculate space ratio compression (psi in paper) as compressed_dense_space/uncompressed_dense_space,
# we use HAM and sHAM for dense layers.
# Results of compression ratio will be printed on terminal and will be saved in txt format 
# at path experiments/space_performance/results/.
python compression.py --compression pruCWS --net original_nets/VGG19-MNIST.h5 --dataset MNIST --clusterfc 64 --prfc 90 --epochs 5
cd ../../space_performance
python uws_testing_space.py -t all -d ../performance_eval/VGG19/VGG19-MNIST/pruCWS/ -m ../performance_eval/VGG19/original_nets/VGG19-MNIST.h5 -s mnist -q 0

cd ../performance_eval/VGG19/

# This example applies uCWS quantization to the VGG19 model trained on the CIFAR10 dataset.
# Compression is only applied to convolutional layers, with clustering fixed to 32 clusters.
# After that we apply Sparse Low Rank (SLR) with q=64 to dense layers, and finally calculate compression ratio.
# Convolutional layers are stored via Index Map format.
python compression.py --compression uCWS --net original_nets/VGG19-CIFAR10.h5 --dataset CIFAR10 --clustercnn 32 --epochs 5
cd ../../space_performance
python uws_slrf_space.py --compression also_quant --net ../performance_eval/VGG19/original_nets/VGG19-CIFAR10.h5 --dataset cifar10 --directory ../performance_eval/VGG19/VGG19-CIFAR10/uCWS/ --keep 64 --sr 0.5 --rr 0.5

cd ../performance_eval/VGG19


# This example applies SLR to the VGG19 model trained on the CIFAR100 dataset.
# Compression is only applied to dense layers, with q fixed to 64 clusters.
# After that we calculate the final occupancy ratio.
cd ../../space_performance
python uws_slrf_space.py --compression only_dense --net ../performance_eval/VGG19/original_nets/VGG19-CIFAR100.h5 --dataset cifar100  --keep 64 --sr 0.5 --rr 0.5

cd ../../performance_eval/VGG19
