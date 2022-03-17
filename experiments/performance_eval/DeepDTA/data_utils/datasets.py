from data_utils.datahelper_noflag import *
import tensorflow as tf
import numpy as np

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

def DAVIS(batch_size):

    max_seq_len = 1200
    max_smi_len = 85

    dataset_path = 'data_utils/davis/'
    problem_type = 1
    dataset = DataSet( fpath = dataset_path,
                          setting_no = problem_type,
                          seqlen = max_seq_len,
                          smilen = max_smi_len,
                          need_shuffle = False )

    charseqset_size = dataset.charseqset_size
    charsmiset_size = dataset.charsmiset_size

    XD, XT, Y = dataset.parse_data(dataset_path, 1)

    XD = np.asarray(XD)
    XT = np.asarray(XT)
    Y = np.asarray(Y)

    test_set, outer_train_sets = dataset.read_sets(dataset_path, problem_type)

    flat_list = [item for sublist in outer_train_sets for item in sublist]

    label_row_inds, label_col_inds = np.where(np.isnan(Y)==False)
    trrows = label_row_inds[flat_list]
    trcol = label_col_inds[flat_list]
    drug, targ, aff = prepare_interaction_pairs(XD, XT,  Y, trrows, trcol)
    trrows = label_row_inds[test_set]
    trcol = label_col_inds[test_set]
    drug_test, targ_test, aff_test = prepare_interaction_pairs(XD, XT,  Y, trrows, trcol)

    input_set = tf.data.Dataset.from_tensor_slices((np.array(drug), np.array(targ)))
    output_set = tf.data.Dataset.from_tensor_slices((np.array(aff)))
    dataset = tf.data.Dataset.zip((input_set, output_set))
    dataset = dataset.batch(batch_size)

    x_train=[np.array(drug), np.array(targ)]
    y_train=np.array(aff)
    x_test=[np.array(drug_test), np.array(targ_test)]
    y_test=np.array(aff_test)

    return dataset, x_train, y_train, x_test, y_test



def KIBA(batch_size):

    max_seq_len = 1000
    max_smi_len = 100

    dataset_path = 'data_utils/kiba/'
    problem_type = 1
    dataset = DataSet( fpath = dataset_path,
                          setting_no = problem_type,
                          seqlen = max_seq_len,
                          smilen = max_smi_len,
                          need_shuffle = False )

    charseqset_size = dataset.charseqset_size
    charsmiset_size = dataset.charsmiset_size

    XD, XT, Y = dataset.parse_data(dataset_path, 0)

    XD = np.asarray(XD)
    XT = np.asarray(XT)
    Y = np.asarray(Y)

    test_set, outer_train_sets = dataset.read_sets(dataset_path, problem_type)

    flat_list = [item for sublist in outer_train_sets for item in sublist]

    label_row_inds, label_col_inds = np.where(np.isnan(Y)==False)
    trrows = label_row_inds[flat_list]
    trcol = label_col_inds[flat_list]
    drug, targ, aff = prepare_interaction_pairs(XD, XT,  Y, trrows, trcol)
    trrows = label_row_inds[test_set]
    trcol = label_col_inds[test_set]
    drug_test, targ_test, aff_test = prepare_interaction_pairs(XD, XT,  Y, trrows, trcol)

    input_set = tf.data.Dataset.from_tensor_slices((np.array(drug), np.array(targ)))
    output_set = tf.data.Dataset.from_tensor_slices((np.array(aff)))
    dataset = tf.data.Dataset.zip((input_set, output_set))
    dataset = dataset.batch(batch_size)

    x_train=[np.array(drug), np.array(targ)]
    y_train=np.array(aff)
    x_test=[np.array(drug_test), np.array(targ_test)]
    y_test=np.array(aff_test)

    return dataset, x_train, y_train, x_test, y_test
