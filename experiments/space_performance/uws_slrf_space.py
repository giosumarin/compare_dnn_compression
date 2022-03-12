import os
from os import listdir
from os.path import isfile, join
import gc
import pickle
import click


import numpy as np
import tensorflow as tf
from datahelper_noflag import *
from tensorflow.keras.datasets import mnist, cifar10, cifar100


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

exec(open("../performance_eval/GPU.py").read())


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

def dense_space(npmatr2d):
    if npmatr2d.dtype == np.float64:
        byte = 64/8
    elif npmatr2d.dtype == np.float32:
        byte = 32/8
    else:
        byte = 32/8
    return npmatr2d.shape[0]*npmatr2d.shape[1]*byte

def cnn_space(npmatr2d):
    if npmatr2d.dtype == np.float64:
        byte = 64/8
    elif npmatr2d.dtype == np.float32:
        byte = 32/8
    else:
        byte = 32/8
    return npmatr2d.size * byte

def list_of_cnn_and_dense_weights_indexes(list_weights):
    # Deep DTA
    if len(list_weights) == 22:
        return [2, 4, 6, 8, 10, 12], [14, 16, 18, 20]
    # VGG19
    elif len(list_weights) == 114:
        return [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90], [96, 102, 108]



def split_filename(fn):
    c = fn.split(sep="-")
    if len(c) == 2:
        return 0, c[0], c[1]
    elif len(c) == 3:
        return c[0], c[1], c[2]
    elif len(c) == 5:
        #return c[0], c[2], c[4]
        return 0, c[3], c[4]

def sparse_SVD_wr(weights,U,S,V,keep,sr,rr):
    tU, tS, tV = U[:, 0:keep], S[0:keep], V[0:keep, :]

    # Input node selection
    iwm = np.sum(abs(weights),axis=1)
    imid = sorted(iwm)[int(weights.shape[0]*sr)]
    ipl = np.where(iwm<imid)[0]

    # Output node selection
    owm = np.sum(abs(weights),axis=0)
    omid = sorted(owm)[int(weights.shape[1]*sr)]
    opl = np.where(owm<omid)[0]

    # Masking the weights
    subrank = int(keep*rr)
    for ind in ipl:
        tU[ind,subrank:]=0

    for ind in opl:
        tV[subrank:,ind]=0

    return tU, tS, tV

def space_slrf(model, keep, sr, rr, x_test, y_test):
    lw = model.get_weights()
    dense_index = list_of_cnn_and_dense_weights_indexes(lw)[1]
    compr_space = 0
    original_space = 0
    for i in dense_index:
        U, S, V = np.linalg.svd(lw[i], full_matrices=False)
        tU, tS, tV = sparse_SVD_wr(lw[i],U,S,V,keep,sr,rr)
        lw[i] = np.matmul(np.matmul(tU, np.diag(tS)), tV)
        compr_space += (tU.size+tS.size+tV.size)*32/8
        original_space += dense_space(lw[i])
    model.set_weights(lw)
    score = model.evaluate(x_test, y_test)
    if type(score) == list:
        score = score[-1]
    
    return score, compr_space, original_space

@click.command()
@click.option('--compression', help='Type of compression')
@click.option('--net', help='original network datapath')
@click.option('--dataset', help='dataset')
@click.option('--directory', default=".", help='datapath compressed model')
@click.option('--keep', default=1, help='keep')
@click.option('--sr', default=1.0, help='sr')
@click.option('--rr', default=1.0, help='rr')


def main(compression, net, dataset, directory, keep,sr,rr):
    
    print(compression, net, dataset, directory, keep,sr,rr)

    # Load model
    model = tf.keras.models.load_model(net)
    
    if dataset == "kiba":
        # data loading
        dataset_path = '../performance_eval/DeepDTA/data_utils/kiba/'
        dataset1 = DataSet( fpath = dataset_path, ### BUNU ARGS DA GUNCELLE
                                setting_no = 1, ##BUNU ARGS A EKLE
                                seqlen = 1000,
                                smilen = 100,
                                need_shuffle = False )

        XD, XT, Y = dataset1.parse_data(dataset_path, 0)

        XD = np.asarray(XD)
        XT = np.asarray(XT)
        Y = np.asarray(Y)

        test_set, outer_train_sets = dataset1.read_sets(dataset_path, 1)

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
    elif dataset == "davis":
        # data loading
        dataset_path = '../performance_eval/DeepDTA/data_utils/davis/'
        dataset1 = DataSet( fpath = dataset_path, ### BUNU ARGS DA GUNCELLE
                                setting_no = 1, ##BUNU ARGS A EKLE
                                seqlen = 1200,
                                smilen = 85,
                                need_shuffle = False )

        XD, XT, Y = dataset1.parse_data(dataset_path, 1)

        XD = np.asarray(XD)
        XT = np.asarray(XT)
        Y = np.asarray(Y)

        test_set, outer_train_sets = dataset1.read_sets(dataset_path, 1)

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
    elif dataset == "mnist":
        # data loading
        ((x_train, y_train), (x_test, y_test)) = mnist.load_data()
        x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
        x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))
        x_train = x_train.astype("float32") / 255.0
        x_test = x_test.astype("float32") / 255.0
        y_train = tf.keras.utils.to_categorical(y_train, 10)
        y_test = tf.keras.utils.to_categorical(y_test, 10)
    elif dataset == "cifar10":
        # data loading
        num_classes = 10
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        y_train = tf.keras.utils.to_categorical(y_train, num_classes)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')

        # data preprocessing
        x_train[:,:,:,0] = (x_train[:,:,:,0]-123.680)
        x_train[:,:,:,1] = (x_train[:,:,:,1]-116.779)
        x_train[:,:,:,2] = (x_train[:,:,:,2]-103.939)
        x_test[:,:,:,0] = (x_test[:,:,:,0]-123.680)
        x_test[:,:,:,1] = (x_test[:,:,:,1]-116.779)
        x_test[:,:,:,2] = (x_test[:,:,:,2]-103.939)
    elif dataset == "cifar100":
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

    original_acc = model.evaluate(x_test, y_test)
    if isinstance(original_acc, list):
        original_acc = original_acc[1]
    
    if compression == "also_quant":
        perf = []
        compr_spaces = []
        wss = []

        onlyfiles = [f for f in listdir(directory) if isfile(join(directory, f))]
        for weights in sorted(onlyfiles):
            gc.collect()
            if weights[-3:] == ".h5":
                lw = pickle.load(open(directory+weights, "rb"))
                model.set_weights(lw)

                cnnIdx, _ = list_of_cnn_and_dense_weights_indexes(lw)
                space_expanded_cnn = sum([cnn_space(lw[i]) for i in cnnIdx])
                vect_weights =[np.hstack(lw[i]).reshape(-1,1) for i in cnnIdx]
                all_vect_weights = np.concatenate(vect_weights, axis=None).reshape(-1,1)
                uniques = np.unique(all_vect_weights)
                num_values = len(uniques)
                print(num_values)
                space_compr_cnn = (num_values*32 + math.ceil(np.log2(num_values)) * sum([lw[i].size for i in cnnIdx])) / 8
                pr, ws, acc = split_filename(weights)
                ws_acc = float(acc[:-3])
                print("{}% & {} --> {}".format(pr, ws, ws_acc))
                
                score, compr_space, original_space = space_slrf(model, keep, sr, rr, x_test, y_test)

                print(weights, score, compr_space, original_space)

                perf += [score]
                compr_spaces += [round((space_compr_cnn+compr_space)/(space_expanded_cnn+original_space), 5)]
                wss += [ws]


        file_res = f"results//res.txt"
        with open(file_res, "a+") as tex:
            tex.write(f"\n\n{keep} {sr} {rr}\n")
            tex.write(directory+"\n")
            tex.write(f"ws = {wss}\nperf = {perf}\ncompr_space={compr_spaces}")

    else:
        score, compr_space, original_space = space_slrf(model, keep, sr, rr, x_test, y_test)
        print(score, compr_space, original_space)


if __name__ == '__main__':
    main()
