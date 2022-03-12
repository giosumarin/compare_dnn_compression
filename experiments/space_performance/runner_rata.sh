#!/bin/bash

# python uws_testing_space.py -t all -d /home/gioscatmarino/exp_balrog/pruPWS/ \
# -m /home/gioscatmarino/exp_balrog/original_nets/vgg19cifar100.h5 -s cifar100 -q 0

# python uws_testing_space.py -t all -d /home/gioscatmarino/exp_balrog/pruCWS/ \
# -m /home/gioscatmarino/exp_balrog/original_nets/vgg19cifar100.h5 -s cifar100 -q 0

python uws_testing_space.py -t also_cnn -d ~/backup_balrog/sHAM/experiments/performance_eval/VGG19/vgg19cifar100/pruCWS/ -m ~/backup_balrog/sHAM/experiments/performance_eval/VGG19/original_nets/vgg19cifar100.h5 -s cifar100 -q 0
python uws_testing_space.py -t also_cnn -d ~/backup_balrog/sHAM/experiments/performance_eval/VGG19/vgg19cifar100/pruPWS-all/ -m ~/backup_balrog/sHAM/experiments/performance_eval/VGG19/original_nets/vgg19cifar100.h5 -s cifar100 -q 0
python uws_testing_space.py -t also_cnn -d ~/backup_balrog/sHAM/experiments/performance_eval/VGG19/vgg19cifar100/pruUQ/ -m ~/backup_balrog/sHAM/experiments/performance_eval/VGG19/original_nets/vgg19cifar100.h5 -s cifar100 -q 0