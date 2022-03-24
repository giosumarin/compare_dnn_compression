import os
from os import listdir
from os.path import isfile, join
import gc
import sys
from sys import getsizeof
from math import floor
import pickle
import timeit
import getopt
from functools import reduce
from itertools import repeat
from multiprocessing import Pool

from numba import njit
import numpy as np
import tensorflow as tf
from datahelper_noflag import *
from tensorflow.keras.datasets import mnist, cifar10, cifar100

from sHAM import huffman
from sHAM import sparse_huffman
from sHAM import sparse_huffman_only_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

exec(open("../performance_eval/GPU.py").read())

def make_word_with_info(matr, encoded, bit_words_machine=64):
    bit = bit_words_machine
    list_string_col =[]
    string = ''
    row = 0
    info = []

    for x in np.nditer(encoded, order='F'):
        if row == 0:
            info += [(len(list_string_col), len(string))]
        string += (np.array2string(x)[1:-1])
        if len(string) > bit:
            bit_overflow = len(string)%bit
            list_string_col += [string[:-bit_overflow]]
            string = string[-bit_overflow:]
        elif len(string) == bit:
            list_string_col += [string]
            string = ''
        row += 1
        if row >= matr.shape[0]:
            row = 0

    bit_remain = len(string)
    info += [(len(list_string_col), bit_remain)]
    if bit_remain > 0:
        string += "0"*(bit-bit_remain) #padding di 0 per renderla lunga bit
        list_string_col += [string]

    return list_string_col, info

def prepare_interaction_pairs(XD, XT,  Y, rows, cols):
    drugs = []
    targets = []
    targetscls = []
    affinity=[]

    for pair_ind in range(len(rows)):
        drug = XD[rows[pair_ind]]
        drugs.append(drug)

        target=XT[cols[pair_ind]]
        targets.append(target)

        affinity.append(Y[rows[pair_ind],cols[pair_ind]])

    drug_data = np.stack(drugs)
    target_data = np.stack(targets)

    return drug_data,target_data,  affinity

def dict_space(dict_):
    space_byte = 0
    for key in dict_:
        space_byte += getsizeof(key)-getsizeof("") + 1 #byte fine stringa
        space_byte += 4 #byte per float32    #dict_[key]
        space_byte += 8 #byte per struttura dict
    return space_byte

def dense_space(npmatr2d):
    if npmatr2d.dtype == np.float64:
        byte = 64/8
    elif npmatr2d.dtype == np.float32:
        byte = 32/8
    else:
        return npmatr2d.shape[0]*npmatr2d.shape[1]*32/8
    return npmatr2d.shape[0]*npmatr2d.shape[1]*byte

def cnn_space(npmatr2d):
    if npmatr2d.dtype == np.float64:
        byte = 64/8
    elif npmatr2d.dtype == np.float32:
        byte = 32/8
    return npmatr2d.size * byte

def list_of_dense_indexes(model):
    index_denses = []
    i = 2
    for layer in model.layers[2:]:
        if type(layer) is tf.keras.layers.Dense:
            index_denses.append(i)
        i += 1
    return index_denses

def make_model_pre_post_dense(model, index_dense):
    submodel = tf.keras.Model(inputs=model.input,
                            outputs=model.layers[index_dense-1].output)
    return submodel

def list_of_dense_weights_indexes(list_weights):
    indexes_denses_weights = []
    i = 2
    for w in list_weights[2:]:
        if len(w.shape) == 2:
            indexes_denses_weights.append(i)
        i += 1
    return indexes_denses_weights

def list_of_cnn_and_dense_weights_indexes(list_weights):
    # Deep DTA
    if len(list_weights) == 22:
        return [2, 4, 6, 8, 10, 12], [14, 16, 18, 20]
    # VGG19
    elif len(list_weights) == 114:
        return [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90], [96, 102, 108]

def make_huffman(model, lodi, lodwi, lw):
    bit_words_machine = 64

    vect_weights = [np.hstack(lw[i]).reshape(-1,1) for i in lodwi]
    all_vect_weights = np.concatenate(vect_weights, axis=None).reshape(-1,1)

    symb2freq = huffman.dict_elem_freq(all_vect_weights)
    e = huffman.encode(symb2freq)

    d_rev = huffman.reverse_elements_list_to_dict(e)
    d = dict(e)

    encodeds = [huffman.matrix_with_code(lw[l], d, d_rev) for l in lodwi]
    list_bins = [huffman.make_words_list_to_int(encoded, bit_words_machine) for encoded in encodeds]
    int_from_strings = [huffman.convert_bin_to_int(list_bin) for list_bin in list_bins]
    
    space_dense = sum([dense_space(lw[i]) for i in lodwi])
    space_huffman = dict_space(d_rev) 
    space_huffman += sum([bit_words_machine/8 * (len(int_from_string)) for int_from_string in int_from_strings]) 

    return space_dense, space_huffman




def space_for_row_cum(matr, list_):
    len_ = matr.shape[0]
    if len_ < 2**8:
        return 1 * len(list_)
    elif len_ < 2**16:
        return 2 * len(list_)
    elif len_ < 2**32:
        return 4 * len(list_)
    return 8 * len(list_)

def make_huffman_sparse_par(model, lodi, lodwi, lw):
    bit_words_machine = 64

    vect_weights = [np.hstack(lw[i]).reshape(-1,1) for i in lodwi]
    all_vect_weights = np.concatenate(vect_weights, axis=None).reshape(-1,1)
        
    symb2freq = huffman.dict_elem_freq(all_vect_weights[all_vect_weights != 0])
    e = huffman.encode(symb2freq)

    d_rev = huffman.reverse_elements_list_to_dict(e)
    d = dict(e)

    dense_inputs = []

    space_dense = 0
    space_sparse_huffman = 0
    space_sparse_huffman += dict_space(d_rev)
    for l in lodwi:
        matr_shape, int_data, d_rev_data, row_index, cum, expected_c, min_length_encoded = sparse_huffman_only_data.do_all_for_me(lw[l], bit_words_machine)
        # data, row_index, cum = sparse_huffman.convert_dense_to_csc(lw[l])
        # print(len(cum), len(row_index))
        # d_data, d_rev_data  = sparse_huffman_only_data.huffman_sparse_encoded_dict(data)
        # data_encoded = sparse_huffman_only_data.encoded_matrix(data, d_data, d_rev_data)
        # int_from_strings = huffman.convert_bin_to_int(huffman.make_words_list_to_int(data_encoded, bit_words_machine))
        space_dense += dense_space(lw[l])  
        space_sparse_huffman += bit_words_machine/8 * len(int_data) 
        space_sparse_huffman += space_for_row_cum(lw[l], cum) + space_for_row_cum(lw[l], row_index)
    return space_dense, space_sparse_huffman    


def split_filename(fn):
    c = fn.split(sep="-")
    if len(c) == 2:
        return 0, c[0], c[1]
    elif len(c) == 3:
        return c[0], c[1], c[2]
    elif len(c) == 5:
        return c[0], c[2], c[4]


# Get the arguments from the command-line except the filename
argv = sys.argv[1:]

try:
    string_error = 'usage: testing_time_space.py -t <type of compression> -d <directory of compressed weights> -m <file original keras model>'
    # Define the getopt parameters
    opts, args = getopt.getopt(argv, 't:d:m:s:q:', ['type', 'directory', 'model', 'dataset', 'quant'])
    if len(opts) != 5:
      print (string_error)
      # Iterate the options and get the corresponding values
    else:

        for opt, arg in opts:
            if opt == "-t":
                #print("type: ", arg)
                type_compr = arg
            elif opt == "-d":
                #print("directory: ", arg)
                directory = arg
            elif opt == "-m":
                #print("model_file: ", arg)
                model_file=arg
            elif opt == "-q":
                pq = True if arg==1 else False
            elif opt == "-s":
                if arg == "kiba":
                    #print(arg)
                    # data loading
                    dataset_path = '../performance_eval/DeepDTA/data_utils/kiba/'
                    dataset = DataSet( fpath = dataset_path, ### BUNU ARGS DA GUNCELLE
                                          setting_no = 1, ##BUNU ARGS A EKLE
                                          seqlen = 1000,
                                          smilen = 100,
                                          need_shuffle = False )

                    XD, XT, Y = dataset.parse_data(dataset_path, 0)

                    XD = np.asarray(XD)
                    XT = np.asarray(XT)
                    Y = np.asarray(Y)

                    test_set, outer_train_sets = dataset.read_sets(dataset_path, 1)

                    flat_list = [item for sublist in outer_train_sets for item in sublist]

                    label_row_inds, label_col_inds = np.where(np.isnan(Y)==False)
                    trrows = label_row_inds[flat_list]
                    trcol = label_col_inds[flat_list]
                    drug, targ, aff = prepare_interaction_pairs(XD, XT,  Y, trrows, trcol)
                    trrows = label_row_inds[test_set]
                    trcol = label_col_inds[test_set]
                    drug_test, targ_test, aff_test = prepare_interaction_pairs(XD, XT,  Y, trrows, trcol)

                    x_train=[np.array(drug), np.array(targ)]
                    y_train=np.array(aff)
                    x_test=[np.array(drug_test), np.array(targ_test)]
                    y_test=np.array(aff_test)

                elif arg == "davis":
                    # data loading
                    dataset_path = '../performance_eval/DeepDTA/data_utils/davis/'
                    dataset = DataSet( fpath = dataset_path, ### BUNU ARGS DA GUNCELLE
                                          setting_no = 1, ##BUNU ARGS A EKLE
                                          seqlen = 1200,
                                          smilen = 85,
                                          need_shuffle = False )

                    XD, XT, Y = dataset.parse_data(dataset_path, 1)

                    XD = np.asarray(XD)
                    XT = np.asarray(XT)
                    Y = np.asarray(Y)

                    test_set, outer_train_sets = dataset.read_sets(dataset_path, 1)

                    flat_list = [item for sublist in outer_train_sets for item in sublist]

                    label_row_inds, label_col_inds = np.where(np.isnan(Y)==False)
                    trrows = label_row_inds[flat_list]
                    trcol = label_col_inds[flat_list]
                    drug, targ, aff = prepare_interaction_pairs(XD, XT,  Y, trrows, trcol)
                    trrows = label_row_inds[test_set]
                    trcol = label_col_inds[test_set]
                    drug_test, targ_test, aff_test = prepare_interaction_pairs(XD, XT,  Y, trrows, trcol)

                    x_train=[np.array(drug), np.array(targ)]
                    y_train=np.array(aff)
                    x_test=[np.array(drug_test), np.array(targ_test)]
                    y_test=np.array(aff_test)
                elif arg == "mnist":
                    #print(arg)
                    # data loading
                    ((x_train, y_train), (x_test, y_test)) = mnist.load_data()
                    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
                    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))
                    x_train = x_train.astype("float32") / 255.0
                    x_test = x_test.astype("float32") / 255.0
                    y_train = tf.keras.utils.to_categorical(y_train, 10)
                    y_test = tf.keras.utils.to_categorical(y_test, 10)
                elif arg == "cifar10":
                    # data loading
                    num_classes = 10
                    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
                    y_train = tf.keras.utils.utils.to_categorical(y_train, num_classes)
                    y_test = tf.keras.utils.utils.to_categorical(y_test, num_classes)
                    x_train = x_train.astype('float32')
                    x_test = x_test.astype('float32')

                    # data preprocessing
                    x_train[:,:,:,0] = (x_train[:,:,:,0]-123.680)
                    x_train[:,:,:,1] = (x_train[:,:,:,1]-116.779)
                    x_train[:,:,:,2] = (x_train[:,:,:,2]-103.939)
                    x_test[:,:,:,0] = (x_test[:,:,:,0]-123.680)
                    x_test[:,:,:,1] = (x_test[:,:,:,1]-116.779)
                    x_test[:,:,:,2] = (x_test[:,:,:,2]-103.939)
                elif arg == "cifar100":
                    # data loading
                    num_classes = 100
                    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
                    x_train = x_train.astype('float32')
                    x_test = x_test.astype('float32')

                    mean = np.mean(x_train,axis=(0,1,2,3))
                    std = np.std(x_train, axis=(0, 1, 2, 3))
                    x_train = (x_train-mean)/(std+1e-7)
                    x_test = (x_test-mean)/(std+1e-7)

                    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
                    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

                    x_train = x_train.astype('float32')
                    x_test = x_test.astype('float32')
except getopt.GetoptError:
    # Print something useful
    print (string_error)
    sys.exit(2)

model = tf.keras.models.load_model(model_file)
original_acc = model.evaluate(x_test, y_test, verbose=0)
if isinstance(original_acc, list):
    original_acc = original_acc[1]


onlyfiles = [f for f in listdir(directory) if isfile(join(directory, f))]

pruning_l_h = []
ws_l_h = []
diff_acc_h = []
space_h = []
space_sh = []
time_h = []
time_h_p = []
time_h_p_cpp = []
nonzero_h = []
perf = []

pruning_l_sh = []
ws_l_sh = []
diff_acc_sh = []
space_sh = []
time_sh = []
nonzero_sh = []

        
for weights in sorted(onlyfiles):
        print("Applying compression")
        gc.collect()
        if weights[-3:] == ".h5":
            lw = pickle.load(open(directory+weights, "rb"))
            model.set_weights(lw)

            cnnIdx, denseIdx = list_of_cnn_and_dense_weights_indexes(lw)
            if type_compr == "also_cnn" or type_compr == "only_cnn":
                space_expanded_cnn = sum([cnn_space(lw[i]) for i in cnnIdx])
            else:
                space_expanded_cnn = 0
            # Estraggo quanti simboli sono usati effettivamente nei liv conv
            if type_compr == "also_cnn" or type_compr == "only_cnn":
                vect_weights =[np.hstack(lw[i]).reshape(-1,1) for i in cnnIdx]
                all_vect_weights = np.concatenate(vect_weights, axis=None).reshape(-1,1)
                uniques = np.unique(all_vect_weights)
                num_values = len(uniques)
                space_compr_cnn = (num_values*32 + math.ceil(np.log2(num_values)) * sum([lw[i].size for i in cnnIdx])) / 8
                #type_compr = "all"
            else:
                space_compr_cnn = space_expanded_cnn

            pr, ws, acc = split_filename(weights)
            ws_acc = float(acc[:-3])
            #print("{}% & {} --> {}".format(pr, ws, ws_acc))
            perf += [ws_acc]

            lodi = list_of_dense_indexes(model)
            lodwi = list_of_dense_weights_indexes(lw)
            # for l in lodwi:
            #     print(lw[l])
            #     #print(np.unique(lw[l]))
            #     print(len(np.nonzero(lw[l])[0]))
            #     print(lw[l].size)
            #     print()
            assert len(lodi) == len(lodwi)

            non_zero = []

            if type_compr == "ham":
                space_dense, space_huffman = make_huffman(model, lodi, lodwi, lw)
            elif type_compr == "sham":
                space_dense, space_shuffman = make_huffman_sparse_par(model, lodi, lodwi, lw)
            elif type_compr in ["all", "also_cnn"]:
                 space_dense, space_huffman = make_huffman(model, lodi, lodwi, lw)
                 space_dense, space_shuffman = make_huffman_sparse_par(model, lodi, lodwi, lw)
            
            pruning_l_h.append(pr)
            ws_l_h.append(ws)
            diff_acc_h.append(round(ws_acc-original_acc, 5))
            if type_compr in ["ham", "all", "also_cnn"]:
                space_h.append(round((space_compr_cnn + space_huffman)/(space_expanded_cnn + space_dense), 5)) # Tengo conto anche di cnn
            if type_compr in ["sham", "all", "also_cnn"]:
                space_sh.append(round((space_compr_cnn + space_shuffman)/(space_expanded_cnn + space_dense), 5))
            nonzero_h.append(non_zero)
            if type_compr in ["only_conv"]:
                space_h.append(round((space_compr_cnn)/(space_expanded_cnn)))
            
            
            print(f"\tCompressed performance test: {perf[-1]}")
            if type_compr == "ham":
                #print("{} {} acc1, space {} ".format(ws_l_h[-1], diff_acc_h[-1], space_h[-1]))
                print(f"\tSpace occupancy of compressed model with HAM: {space_h[-1]}")

            elif type_compr == "sham":
                ### Commentato per salvare solo i tempi, non i rapporti
                # print("{} {} acc1, space {}, time p {} time p cpp {} ".format(ws_l_h[-1], diff_acc_h[-1], space_h[-1], time_h_p[-1], time_h_p_cpp[-1]))
                #print("{} {} acc1, space {}".format(ws_l_h[-1], diff_acc_h[-1], space_sh[-1]))
                print(f"\tSpace occupancy of compressed model with sHAM: {space_sh[-1]}")
                ####
            elif type_compr in ["all", "also_cnn"] :
                #print("{} {} acc1, spaceh {}, spacesh {}".format(ws_l_h[-1], diff_acc_h[-1], space_h[-1], space_sh[-1], ))
                print(f"\tSpace occupancy of compressed model with HAM: {space_h[-1]}")
                print(f"\tSpace occupancy of compressed model with sHAM: {space_sh[-1]}")
            #elif type_compr == "only_conv":
                ### Commentato per salvare solo i tempi, non i rapporti
                # print("{} {} acc1, space {}, time p {} time p cpp {} ".format(ws_l_h[-1], diff_acc_h[-1], space_h[-1], time_h_p[-1], time_h_p_cpp[-1]))
                #print("{} {} acc1, space {}".format(ws_l_h[-1], diff_acc_h[-1], space_h[-1]))

if type_compr == "ham":
    str_res = "results/huffman_upq.txt" if pq else "results/huffman_pruws.txt"
    with open(str_res, "a+") as tex:
        tex.write(directory)
        tex.write("\npruning = {} \nclusters = {}\ndiff_acc = {}\nperf = {}\nspace = {} \n".format(pruning_l_h, ws_l_h, diff_acc_h, perf, space_h))

elif type_compr == "sham":
    str_res = "results/huffman_upq.txt" if pq else "results/huffman_pruws_sparse.txt"
    with open(str_res, "a+") as tex:
        tex.write(directory)
        tex.write("\npruning = {} \nclusters = {}\ndiff_acc = {}\nperf = {}\nspace = {}\n".format(pruning_l_h, ws_l_h, diff_acc_h, perf, space_h))

if type_compr in ["all", "also_cnn"]:
    str_res = "results/all_upq.txt" if pq else "results/all_pruws.txt"
    with open(str_res, "a+") as tex:
        tex.write(directory)
        tex.write("\npruning = {} \nclusters = {}\ndiff_acc = {}\nperf = {}\nspaceh = {}\nspacesh = {}\n".format(pruning_l_h, ws_l_h, diff_acc_h, perf, space_h, space_sh))

if type_compr == "only_conv":
    str_res = "results/all_upq.txt" if pq else "results/all_pruws.txt"
    with open(str_res, "a+") as tex:
        tex.write(directory)
        tex.write("\npruning = {} \nclusters = {}\ndiff_acc = {}\nperf = {}\nspaceh = {}\n".format(pruning_l_h, ws_l_h, diff_acc_h, perf, space_h))
