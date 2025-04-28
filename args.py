import torch
import os


class Config(object):
    """args in model and trainer"""
    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.num_fold = 5
        self.num_classes = 5
        self.num_epochs = 200               # Because early stopping is used, this parameter can be relatively large
        self.batch_size = 448
        self.pad_size = 29                  # time dimension of TF image
        self.learning_rate = 2e-5
        self.dropout = 0.1                  # dropout rate in transformer encoder
        self.dim_model = 128                # frequency of TF image
        self.forward_hidden = 1024          # hidden units of transformer encoder
        self.fc_hidden = 1024               # hidden units of FC layers
        self.num_head = 8                   # number of heads in transformer encoder
        self.num_encoder = 12               # number of encoders in single-channel feature extraction block
        self.num_encoder_multi = 5          # number of encoders in multi-channel feature fusion block
        self.use_relative_pos = False        # whether to use relative positional encoding
        self.max_relative_position = 20     # maximum relative position for relative positional encoding
        self.lr_scheduler_t_max = 200      # 余弦退火周期长度，通常设置为总epoch数
        self.lr_scheduler_eta_min = 1e-7   # 学习率下限
        self.dynamic_lr_schedule = True    # 是否使用动态学习率调度（根据早停指标调整）
        self.lr_patience_ratio = 2.0       # 动态学习率周期与早停patience的比例


class Path(object):
    """path of files in this project"""
    def __init__(self):
        self.path_PSG = 'dataset/sleepEDF-78/sleep-cassette'
        self.path_hypnogram = 'dataset/sleepEDF-78/Hypnogram'
        self.path_raw_data = 'data/sleepEDF-78/data_array/raw_data'
        self.path_labels = 'data/sleepEDF-78/data_array/raw_data/labels'
        self.path_TF = 'data/sleepEDF-78/data_array/TF_data'
        self.path_figure = 'figures'

        if not os.path.exists(self.path_hypnogram):
            os.makedirs(self.path_hypnogram)

        if not os.path.exists(self.path_raw_data):
            os.makedirs(self.path_raw_data)

        if not os.path.exists(self.path_TF):
            os.makedirs(self.path_TF)

        if not os.path.exists(self.path_figure):
            os.makedirs(self.path_figure)
