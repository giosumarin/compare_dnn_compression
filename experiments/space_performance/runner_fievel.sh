
#python uws_testing_space.py -t all -d ../performance_eval/VGG19/vgg19cifar100/pruECSQ/ -m ../performance_eval/VGG19/original_nets/vgg19cifar100.h5 -s cifar100 -q 0




# python uws_slrf_space.py --compression also_quant --net ../performance_eval/VGG19/original_nets/VGG19-MNIST.h5 --dataset mnist --directory ../performance_eval/VGG19/VGG19-MNIST/uECSQ/ --keep 8 --sr 0.5 --rr 0.5
# python uws_slrf_space.py --compression also_quant --net ../performance_eval/VGG19/original_nets/VGG19-MNIST.h5 --dataset mnist --directory ../performance_eval/VGG19/VGG19-MNIST/uECSQ/ --keep 16 --sr 0.5 --rr 0.5
# python uws_slrf_space.py --compression also_quant --net ../performance_eval/VGG19/original_nets/VGG19-MNIST.h5 --dataset mnist --directory ../performance_eval/VGG19/VGG19-MNIST/uECSQ/ --keep 32 --sr 0.5 --rr 0.5
# python uws_slrf_space.py --compression also_quant --net ../performance_eval/VGG19/original_nets/VGG19-MNIST.h5 --dataset mnist --directory ../performance_eval/VGG19/VGG19-MNIST/uECSQ/ --keep 64 --sr 0.5 --rr 0.5


# python uws_slrf_space.py --compression also_quant --net ../performance_eval/VGG19/original_nets/VGG19-CIFAR10.h5 --dataset cifar10 --directory ../performance_eval/VGG19/VGG19-CIFAR10/uECSQ/ --keep 8 --sr 0.5 --rr 0.5
# python uws_slrf_space.py --compression also_quant --net ../performance_eval/VGG19/original_nets/VGG19-CIFAR10.h5 --dataset cifar10 --directory ../performance_eval/VGG19/VGG19-CIFAR10/uECSQ/ --keep 16 --sr 0.5 --rr 0.5
# python uws_slrf_space.py --compression also_quant --net ../performance_eval/VGG19/original_nets/VGG19-CIFAR10.h5 --dataset cifar10 --directory ../performance_eval/VGG19/VGG19-CIFAR10/uECSQ/ --keep 32 --sr 0.5 --rr 0.5
# python uws_slrf_space.py --compression also_quant --net ../performance_eval/VGG19/original_nets/VGG19-CIFAR10.h5 --dataset cifar10 --directory ../performance_eval/VGG19/VGG19-CIFAR10/uECSQ/ --keep 64 --sr 0.5 --rr 0.5


# python uws_slrf_space.py --compression also_quant --net ../performance_eval/VGG19/original_nets/vgg19cifar100.h5 --dataset cifar100 --directory ../performance_eval/VGG19/vgg19cifar100/uECSQ/ --keep 8 --sr 0.5 --rr 0.5
# python uws_slrf_space.py --compression also_quant --net ../performance_eval/VGG19/original_nets/vgg19cifar100.h5 --dataset cifar100 --directory ../performance_eval/VGG19/vgg19cifar100/uECSQ/ --keep 16 --sr 0.5 --rr 0.5
# python uws_slrf_space.py --compression also_quant --net ../performance_eval/VGG19/original_nets/vgg19cifar100.h5 --dataset cifar100 --directory ../performance_eval/VGG19/vgg19cifar100/uECSQ/ --keep 32 --sr 0.5 --rr 0.5
# python uws_slrf_space.py --compression also_quant --net ../performance_eval/VGG19/original_nets/vgg19cifar100.h5 --dataset cifar100 --directory ../performance_eval/VGG19/vgg19cifar100/uECSQ/ --keep 64 --sr 0.5 --rr 0.5





# python uws_slrf_space.py --compression also_quant --net ../performance_eval/DeepDTA/original_nets/deepDTA_davis.h5 --dataset davis --directory ../performance_eval/DeepDTA/deepDTA_davis/uECSQ/ --keep 8 --sr 0.5 --rr 0.5
# python uws_slrf_space.py --compression also_quant --net ../performance_eval/DeepDTA/original_nets/deepDTA_davis.h5 --dataset davis --directory ../performance_eval/DeepDTA/deepDTA_davis/uECSQ/ --keep 16 --sr 0.5 --rr 0.5
# python uws_slrf_space.py --compression also_quant --net ../performance_eval/DeepDTA/original_nets/deepDTA_davis.h5 --dataset davis --directory ../performance_eval/DeepDTA/deepDTA_davis/uECSQ/ --keep 32 --sr 0.5 --rr 0.5
# python uws_slrf_space.py --compression also_quant --net ../performance_eval/DeepDTA/original_nets/deepDTA_davis.h5 --dataset davis --directory ../performance_eval/DeepDTA/deepDTA_davis/uECSQ/ --keep 64 --sr 0.5 --rr 0.5


# python uws_slrf_space.py --compression also_quant --net ../performance_eval/DeepDTA/original_nets/deepDTA_kiba.h5 --dataset kiba --directory ../performance_eval/DeepDTA/deepDTA_kiba/uECSQ/ --keep 8 --sr 0.5 --rr 0.5
# python uws_slrf_space.py --compression also_quant --net ../performance_eval/DeepDTA/original_nets/deepDTA_kiba.h5 --dataset kiba --directory ../performance_eval/DeepDTA/deepDTA_kiba/uECSQ/ --keep 16 --sr 0.5 --rr 0.5
# python uws_slrf_space.py --compression also_quant --net ../performance_eval/DeepDTA/original_nets/deepDTA_kiba.h5 --dataset kiba --directory ../performance_eval/DeepDTA/deepDTA_kiba/uECSQ/ --keep 32 --sr 0.5 --rr 0.5
# python uws_slrf_space.py --compression also_quant --net ../performance_eval/DeepDTA/original_nets/deepDTA_kiba.h5 --dataset kiba --directory ../performance_eval/DeepDTA/deepDTA_kiba/uECSQ/ --keep 64 --sr 0.5 --rr 0.5

# cd ../performance_eval/VGG19/
# for p in 90 92 95 96 97 99
# do
# 	for k in 32 64 128 256
# 	do
# 		python compression.py --compression pruECSQ --net original_nets/vgg19cifar100.h5 --dataset CIFAR100 --clusterfc $k --clustercnn $k --prfc $p
# 	done
# done 

python uws_testing_space.py -t also_cnn -d ../performance_eval/VGG19/vgg19cifar100/pruECSQ_all/ -m ../performance_eval/VGG19/original_nets/vgg19cifar100.h5 -s cifar100 -q 0
#python uws_testing_space.py -t all -d ../performance_eval/VGG19/vgg19cifar100/uECSQ/ -m ../performance_eval/VGG19/original_nets/vgg19cifar100.h5 -s cifar100 -q 0
