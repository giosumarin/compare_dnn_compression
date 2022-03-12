import time

from sHAM import pruning_uCWS
from sHAM import uPWS
from sHAM.uCWS import idx_matrix_to_matrix
from tensorflow.keras.layers import Conv1D, Conv2D, Conv3D, Dense


class pruning_uPWS(pruning_uCWS.pruning_uCWS, uPWS.uPWS):
    def __init__(self, model, perc_prun_for_dense, perc_prun_for_cnn, clusters_for_dense_layers, clusters_for_conv_layers):
        self.model = model
        self.perc_prun_for_dense = perc_prun_for_dense  # 0 disabilita pruning per livelli densi
        self.perc_prun_for_cnn = perc_prun_for_cnn      # 0 disabilita pruning per livelli convoluzionali
        self.clusters_fc = clusters_for_dense_layers
        self.clusters_cnn = clusters_for_conv_layers
        self.level_idxs = self.list_layer_idx()
        self.timestamped_filename = str(time.time()) + "_check.h5"

    def apply_pr_uPWS(self):
        self.apply_pruning()
        self.apply_uPWS()
        if self.perc_prun_for_dense > 0:
            self.recompose_weight_first(Dense, self.clusters_fc, self.centers_fc, self.idx_layers_fc)
        if self.perc_prun_for_cnn > 0:
            self.recompose_weight_first((Conv1D, Conv2D, Conv3D), self.clusters_cnn, self.centers_cnn, self.idx_layers_cnn)
        #print(self.model.get_weights())
    

