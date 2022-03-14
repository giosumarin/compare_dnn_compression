#!/bin/bash

# This example applies uECQS quantization to the DeepDTA model trained on the Davis dataset.
# Clustering is applied to CNN (64 clusters) and FC (64 clusters) layer. 
# After compression we calculate space ratio compression (psi in paper) as compressed_space/uncompressed_space,
# we use iMap for convolutional layer and HAM for dense layer.
# Results of compression ratio will be printed on terminal and will be saved in txt format 
# at path experiments/space_performance/results/.
python compression.py --compression uECSQ --net original_nets/deepDTA_davis.h5 --dataset DAVIS --clusterfc 64 --clustercnn 64
cd ../space_performance
python uws_testing_space.py -t also_cnn -d ../performance_eval/DeepDTA/deepDTA_davis/uECQS/ -m ../performance_eval/DeepDTA/original_nets/deepDTA_davis.h5 -s davis -q 0

cd ../performance_eval


# This example applies pruning and uCWS quantization to the DeepDTA model trained on the Kiba dataset.
# Pruning is only applied to fully connected layers (percentage 60%), while clustering is applied to 
# fully connected (32 clusters) and convolutional (128 clusters) layers.
# After compression we calculate space ratio compression (psi in paper) as compressed_space/uncompressed_space,
# we use iMap for convolutional layer, for dense layer we calculate space with HAM and sHAM.
# Results of compression ratio will be printed on terminal and will be saved in txt format 
# at path experiments/space_performance/results/.

python compression.py --compression pruCWS --net original_nets/deepDTA_kiba.h5 --dataset KIBA --clustercnn 64 --clusterfc 64 --prfc 60
cd ../space_performance
python uws_testing_space.py -t also_cnn -d ../performance_eval/DeepDTA/deepDTA_kiba/pruCWS/ -m ../performance_eval/DeepDTA/original_nets/deepDTA_kiba.h5 -s kiba -q 0

cd ../performance_eval
