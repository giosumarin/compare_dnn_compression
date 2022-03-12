import time

from sHAM import pruning_uCWS
from sHAM import uECSQ
from tensorflow.keras.layers import Conv1D, Conv2D, Conv3D, Dense


class pruning_uECSQ(pruning_uCWS.pruning_uCWS, uECSQ.uECSQ):
    def __init__(self, model, perc_prun_for_dense, perc_prun_for_cnn, clusters_for_dense_layers, clusters_for_conv_layers, wanted_clusters_cnn, wanted_clusters_fc, tr=0.001, lamb=0.5):
        self.model = model
        self.perc_prun_for_dense = perc_prun_for_dense  # 0 disabilita pruning per livelli densi
        self.perc_prun_for_cnn = perc_prun_for_cnn      # 0 disabilita pruning per livelli convoluzionali
        self.clusters_fc = clusters_for_dense_layers # 0 disables Fully Connected clustering
        self.clusters_cnn = clusters_for_conv_layers # 0 disables CNN clustering
        self.wanted_clusters_fc = wanted_clusters_fc
        self.wanted_clusters_cnn = wanted_clusters_cnn
        self.level_idxs = self.list_layer_idx()
        self.timestamped_filename = str(time.time()) + "_check.h5"
        self.lamb_fc = lamb
        self.lamb_cnn = lamb
        self.tr = tr

    def apply_pr_uECSQ(self):
        self.apply_pruning()
        self.apply_uECSQ()
        if self.perc_prun_for_dense > 0:
            self.recompose_weight_first(Dense, self.clusters_fc, self.centers_fc, self.idx_layers_fc)
        if self.perc_prun_for_cnn > 0:
            self.recompose_weight_first((Conv1D, Conv2D, Conv3D), self.clusters_cnn, self.centers_cnn, self.idx_layers_cnn)

